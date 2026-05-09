import asyncio
import base64
import cgi
import logging
import os
import shutil
import socket
import uuid
import webbrowser
from asyncio import CancelledError
from pathlib import Path
from urllib.parse import quote
import typing as T

import anyio
from babeldoc import __version__ as babeldoc_version
from babeldoc.docvision.doclayout import OnnxModel
from fasthtml.common import *
import requests
from starlette.datastructures import UploadFile
from starlette.responses import FileResponse, Response
import tqdm

from pdf2zh import __version__
from pdf2zh.config import ConfigManager
from pdf2zh.translator import (
    AnythingLLMTranslator,
    ArgosTranslator,
    AzureOpenAITranslator,
    AzureTranslator,
    BaseTranslator,
    BingTranslator,
    DeepLTranslator,
    DeepLXTranslator,
    DeepseekTranslator,
    DifyTranslator,
    GeminiTranslator,
    GoogleTranslator,
    GrokTranslator,
    GroqTranslator,
    MiniMaxTranslator,
    ModelScopeTranslator,
    OllamaTranslator,
    OpenAITranslator,
    OpenAIlikedTranslator,
    QwenMtTranslator,
    SiliconTranslator,
    TencentTranslator,
    X302AITranslator,
    XinferenceTranslator,
    ZhipuTranslator,
)

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("pdf2zh_files")


class GuiError(RuntimeError):
    """User-facing GUI error."""


class _LazyModel:
    """Defers model loading until first access so the GUI starts instantly."""

    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            self._model = OnnxModel.load_available()

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        self._ensure_loaded()
        return getattr(self._model, name)


BABELDOC_MODEL = _LazyModel()

service_map: dict[str, BaseTranslator] = {
    "Google": GoogleTranslator,
    "Bing": BingTranslator,
    "DeepL": DeepLTranslator,
    "DeepLX": DeepLXTranslator,
    "Ollama": OllamaTranslator,
    "Xinference": XinferenceTranslator,
    "AzureOpenAI": AzureOpenAITranslator,
    "OpenAI": OpenAITranslator,
    "Zhipu": ZhipuTranslator,
    "ModelScope": ModelScopeTranslator,
    "Silicon": SiliconTranslator,
    "Gemini": GeminiTranslator,
    "Azure": AzureTranslator,
    "Tencent": TencentTranslator,
    "Dify": DifyTranslator,
    "AnythingLLM": AnythingLLMTranslator,
    "Argos Translate": ArgosTranslator,
    "Grok": GrokTranslator,
    "Groq": GroqTranslator,
    "DeepSeek": DeepseekTranslator,
    "MiniMax": MiniMaxTranslator,
    "OpenAI-liked": OpenAIlikedTranslator,
    "Ali Qwen-Translation": QwenMtTranslator,
    "302.AI": X302AITranslator,
}

lang_map = {
    "Simplified Chinese": "zh",
    "Traditional Chinese": "zh-TW",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Spanish": "es",
    "Italian": "it",
}

page_map = {
    "All": None,
    "First": [0],
    "First 5 pages": list(range(0, 5)),
    "Others": None,
}

flag_demo = False
if ConfigManager.get("PDF2ZH_DEMO"):
    flag_demo = True
    service_map = {"Google": GoogleTranslator}
    page_map = {"First": [0], "First 20 pages": list(range(0, 20))}
    client_key = ConfigManager.get("PDF2ZH_CLIENT_KEY")
    server_key = ConfigManager.get("PDF2ZH_SERVER_KEY")

enabled_services: T.Optional[T.List[str]] = ConfigManager.get("ENABLED_SERVICES")
if isinstance(enabled_services, list):
    default_services = ["Google", "Bing"]
    enabled_services_names = [str(_).lower().strip() for _ in enabled_services]
    enabled_services = [
        k
        for k in service_map.keys()
        if str(k).lower().strip() in enabled_services_names
    ]
    if len(enabled_services) == 0:
        raise RuntimeError("No services available.")
    enabled_services = default_services + enabled_services
else:
    enabled_services = list(service_map.keys())

hidden_secret_details: bool = bool(ConfigManager.get("HIDDEN_GRADIO_DETAILS"))
cancellation_event_map = {}


def verify_recaptcha(response):
    recaptcha_url = "https://www.google.com/recaptcha/api/siteverify"
    data = {"secret": server_key, "response": response}
    result = requests.post(recaptcha_url, data=data).json()
    return result.get("success")


def download_with_limit(url: str, save_path: Path, size_limit: int | None) -> Path:
    chunk_size = 1024
    total_size = 0
    with requests.get(url, stream=True, timeout=10) as response:
        response.raise_for_status()
        content = response.headers.get("Content-Disposition")
        try:
            _, params = cgi.parse_header(content)
            filename = params["filename"]
        except Exception:
            filename = os.path.basename(url)
        filename = os.path.splitext(os.path.basename(filename))[0] + ".pdf"
        path = save_path / filename
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                total_size += len(chunk)
                if size_limit and total_size > size_limit:
                    raise GuiError("Exceeds file size limit")
                file.write(chunk)
    return path


def stop_translate_file(session_id: str | None) -> None:
    if session_id and session_id in cancellation_event_map:
        logger.info("Stopping translation for session %s", session_id)
        cancellation_event_map[session_id].set()


def _selected_pages(page_range: str, page_input: str) -> list[int] | None:
    if page_range != "Others":
        return page_map[page_range]
    selected_page = []
    for p in page_input.split(","):
        p = p.strip()
        if not p:
            continue
        if "-" in p:
            start, end = p.split("-")
            selected_page.extend(range(int(start) - 1, int(end)))
        else:
            selected_page.append(int(p) - 1)
    return selected_page


def translate_file(
    file_type,
    file_input,
    link_input,
    service,
    lang_from,
    lang_to,
    page_range,
    page_input,
    prompt,
    threads,
    skip_subset_fonts,
    ignore_cache,
    vfont,
    mode_choice,
    recaptcha_response,
    session_id,
    *envs,
):
    session_id = session_id or str(uuid.uuid4())
    cancellation_event_map[session_id] = asyncio.Event()
    if flag_demo and not verify_recaptcha(recaptcha_response):
        raise GuiError("reCAPTCHA fail")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if file_type == "File":
        if not file_input:
            raise GuiError("No input")
        source_path = Path(file_input)
        destination_path = OUTPUT_DIR / source_path.name
        if source_path.resolve() == destination_path.resolve():
            file_path = source_path
        else:
            file_path = Path(shutil.copy(source_path, OUTPUT_DIR))
    else:
        if not link_input:
            raise GuiError("No input")
        file_path = download_with_limit(
            link_input,
            OUTPUT_DIR,
            5 * 1024 * 1024 if flag_demo else None,
        )

    filename = os.path.splitext(os.path.basename(file_path))[0]
    file_raw = OUTPUT_DIR / f"{filename}.pdf"
    file_mono = OUTPUT_DIR / f"{filename}-mono.pdf"
    file_dual = OUTPUT_DIR / f"{filename}-dual.pdf"

    translator = service_map[service]
    selected_page = _selected_pages(page_range, page_input)
    lang_from = lang_map[lang_from]
    lang_to = lang_map[lang_to]

    _envs = {}
    for i, env in enumerate(translator.envs.items()):
        _envs[env[0]] = envs[i] if i < len(envs) else env[1]
    for k, v in _envs.items():
        if str(k).upper().endswith("API_KEY") and str(v) == "***":
            _envs[k] = ConfigManager.get_env_by_translatername(translator, k, None)

    def progress_bar(t: tqdm.tqdm):
        desc = getattr(t, "desc", "Translating...") or "Translating..."
        total = getattr(t, "total", 0) or 1
        logger.info("%s %.0f%%", desc, 100 * t.n / total)

    try:
        threads = int(threads)
    except ValueError:
        threads = 1

    try:
        from pdf2zh.kernel import KernelRegistry
        from pdf2zh.kernel.protocol import TranslateRequest

        KernelRegistry.switch(mode_choice)
        kernel = KernelRegistry.get()
        request = TranslateRequest(
            files=[str(file_raw)],
            output=str(OUTPUT_DIR),
            pages=selected_page,
            lang_in=lang_from,
            lang_out=lang_to,
            service=f"{translator.name}",
            thread=int(threads),
            envs=_envs,
            prompt=str(prompt) if prompt else None,
            skip_subset_fonts=skip_subset_fonts,
            ignore_cache=ignore_cache,
            vfont=vfont,
        )
        kernel.translate(
            request,
            callback=progress_bar,
            cancellation_event=cancellation_event_map[session_id],
        )
    except CancelledError as exc:
        raise GuiError("Translation cancelled") from exc
    finally:
        cancellation_event_map.pop(session_id, None)

    if not file_mono.exists() or not file_dual.exists():
        raise GuiError("No output")
    return str(file_mono), str(file_dual)


def parse_user_passwd(file_path: str) -> tuple:
    tuple_list = []
    content = ""
    if not file_path:
        return tuple_list, content
    if len(file_path) == 2:
        try:
            with open(file_path[1], "r", encoding="utf-8") as file:
                content = file.read()
        except FileNotFoundError:
            print(f"Error: File '{file_path[1]}' not found.")
    try:
        with open(file_path[0], "r", encoding="utf-8") as file:
            tuple_list = [
                tuple(line.strip().split(",")) for line in file if line.strip()
            ]
    except FileNotFoundError:
        print(f"Error: File '{file_path[0]}' not found.")
    return tuple_list, content


def _has_ipv6() -> bool:
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.close()
        return True
    except OSError:
        return False


def _option(label: str, selected: str | None = None):
    return Option(label, value=label, selected=label == selected)


def _field(label: str, child):
    return Label(Span(label), child)


def _checkbox(label: str, name: str, checked: bool = False):
    return Label(Input(type="checkbox", name=name, value="true", checked=checked), label)


def _service_env_fields(service: str):
    translator = service_map[service]
    fields = [Input(type="hidden", name=f"env_{i}", value="") for i in range(4)]
    for i, env in enumerate(translator.envs.items()):
        label = env[0]
        value = ConfigManager.get_env_by_translatername(translator, env[0], env[1])
        input_type = "password" if "API_KEY" in label.upper() else "text"
        if hidden_secret_details and "MODEL" not in str(label).upper() and value:
            value = "***" if "API_KEY" in label.upper() else value
        fields[i] = _field(
            label,
            Input(type=input_type, name=f"env_{i}", value=value or "", autocomplete="off"),
        )
    if translator.CustomPrompt:
        fields[-1] = _field("Custom Prompt for llm", Textarea("", name="prompt", rows=5))
    else:
        fields.append(Input(type="hidden", name="prompt", value=""))
    return Div(*fields, id="env-fields", cls="stack")


def _result_panel(mono: str | None = None, dual: str | None = None, error: str | None = None):
    if error:
        return Div(H2("Translation failed"), P(error), cls="result error", id="result")
    if not mono or not dual:
        return Div(H2("Translated"), P("Run a translation to create output files."), id="result")
    mono_name = Path(mono).name
    dual_name = Path(dual).name
    mono_url = f"/file?name={quote(mono_name)}"
    dual_url = f"/file?name={quote(dual_name)}"
    return Div(
        H2("Translated"),
        Div(
            Label(
                Input(
                    type="radio",
                    name="translated_view",
                    value="mono",
                    checked=True,
                    data_url=mono_url,
                ),
                "Mono",
            ),
            Label(
                Input(
                    type="radio",
                    name="translated_view",
                    value="dual",
                    data_url=dual_url,
                ),
                "Dual",
            ),
            cls="radio-row",
        ),
        Div(
            A(
                "Download Translation (Mono)",
                href=mono_url,
                cls="button",
            ),
            A(
                "Download Translation (Dual)",
                href=dual_url,
                cls="button secondary",
            ),
            cls="actions",
        ),
        Iframe(id="translated-frame", src=mono_url, title="Translated Document"),
        Script(
            """
            document.querySelectorAll('input[name="translated_view"]').forEach((input) => {
                input.addEventListener('change', (event) => {
                    document.getElementById('translated-frame').src = event.target.dataset.url;
                });
            });
            """
        ),
        id="result",
        cls="result",
    )


def _preview_panel(filename: str | None = None):
    src = f"/file?name={quote(filename)}" if filename else ""
    return Div(
        H2("Preview"),
        Iframe(src=src, title="Document Preview"),
        id="preview-panel",
        cls="preview",
    )


def _auth_response(message: str):
    return Response(
        message or "Authentication required",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="PDFMathTranslate"'},
        media_type="text/html",
    )


def _authorized(req, user_list: list[tuple[str, str]], auth_message: str):
    if not user_list:
        return None
    header = req.headers.get("authorization", "")
    if not header.startswith("Basic "):
        return _auth_response(auth_message)
    try:
        raw = base64.b64decode(header.split(" ", 1)[1]).decode("utf-8")
        username, password = raw.split(":", 1)
    except Exception:
        return _auth_response(auth_message)
    if (username, password) not in user_list:
        return _auth_response(auth_message)
    return None


def _page(*children):
    recaptcha = []
    if flag_demo:
        recaptcha = [
            Script(
                f"""
                function pdf2zhRecaptchaOnload() {{
                    const box = document.getElementById('recaptcha-box');
                    if (!box || box.dataset.rendered) return;
                    box.dataset.rendered = 'true';
                    grecaptcha.render('recaptcha-box', {{
                        sitekey: '{client_key}',
                        callback: function(token) {{
                            document.getElementById('recaptcha-response').value = token;
                        }}
                    }});
                }}
                """
            ),
            Script(
                src=(
                    "https://www.google.com/recaptcha/api.js"
                    "?onload=pdf2zhRecaptchaOnload&render=explicit"
                ),
                async_=True,
                defer=True,
            ),
        ]
    return (
        Title("PDFMathTranslate - PDF Translation with preserved formats"),
        *recaptcha,
        Main(
            Header(
                H1(A("PDFMathTranslate @ GitHub", href="https://github.com/Byaidu/PDFMathTranslate")),
                P("PDF translation with preserved formats"),
            ),
            *children,
            cls="app-shell",
        ),
    )


def create_app(user_list: list[tuple[str, str]] | None = None, auth_message: str = ""):
    user_list = user_list or []
    app, rt = fast_app(
        pico=True,
        secret_key=os.environ.get("PDF2ZH_SESSION_SECRET", "pdf2zh-fasthtml"),
        hdrs=(
            Style(
                """
                :root { --pico-border-radius: 6px; }
                body { background: #f7f8fb; }
                .app-shell { max-width: 1440px; }
                header { margin-bottom: 1.25rem; }
                header h1 { font-size: 1.8rem; margin-bottom: .25rem; }
                .layout { display: grid; grid-template-columns: minmax(300px, 420px) 1fr; gap: 1.25rem; align-items: start; }
                .panel { background: #fff; border: 1px solid #dfe3ea; border-radius: 8px; padding: 1rem; }
                .stack { display: grid; gap: .75rem; }
                .split { display: grid; grid-template-columns: 1fr 1fr; gap: .75rem; }
                .actions { display: flex; flex-wrap: wrap; gap: .75rem; align-items: center; }
                .radio-row { display: flex; gap: 1rem; align-items: center; margin-bottom: .75rem; }
                .radio-row label { display: inline-flex; width: auto; gap: .35rem; align-items: center; margin: 0; }
                .radio-row input { margin: 0; }
                .secondary { background: #eef2f7; color: #243042; border-color: #d8dee8; }
                .muted { color: #687386; font-size: .9rem; }
                .error { border-color: #d33; color: #9b1c1c; }
                iframe { width: 100%; height: min(78vh, 1100px); border: 1px solid #dfe3ea; border-radius: 8px; background: #fff; }
                details { margin-top: .75rem; }
                @media (max-width: 900px) {
                    .layout, .split { grid-template-columns: 1fr; }
                    iframe { height: 70vh; }
                }
                """
            ),
        ),
    )

    @rt("/")
    def index(req):
        auth = _authorized(req, user_list, auth_message)
        if auth:
            return auth
        session_id = str(uuid.uuid4())
        default_service = enabled_services[0]
        form = Form(
            Input(type="hidden", name="session_id", value=session_id),
            _field(
                "Type",
                Select(_option("File", "File"), _option("Link", "File"), name="file_type"),
            ),
            Div(
                _field(
                    "File" + (" | < 5 MB" if flag_demo else ""),
                    Input(
                        type="file",
                        name="file_input",
                        accept=".pdf,.doc,.docx",
                        hx_post="/preview",
                        hx_trigger="change",
                        hx_target="#preview-panel",
                        hx_swap="outerHTML",
                        hx_encoding="multipart/form-data",
                    ),
                ),
                _field("Link", Input(type="url", name="link_input", placeholder="https://...")),
                cls="stack",
            ),
            H2("Option"),
            _field(
                "Service",
                Select(
                    *[_option(service, default_service) for service in enabled_services],
                    name="service",
                    hx_get="/service-fields",
                    hx_target="#env-fields",
                    hx_trigger="change",
                    hx_include="[name='service']",
                ),
            ),
            _service_env_fields(default_service),
            Div(
                _field(
                    "Translate from",
                    Select(
                        *[
                            _option(lang, ConfigManager.get("PDF2ZH_LANG_FROM", "English"))
                            for lang in lang_map.keys()
                        ],
                        name="lang_from",
                    ),
                ),
                _field(
                    "Translate to",
                    Select(
                        *[
                            _option(
                                lang,
                                ConfigManager.get("PDF2ZH_LANG_TO", "Simplified Chinese"),
                            )
                            for lang in lang_map.keys()
                        ],
                        name="lang_to",
                    ),
                ),
                cls="split",
            ),
            _field(
                "Pages",
                Select(*[_option(page, list(page_map.keys())[0]) for page in page_map.keys()], name="page_range"),
            ),
            _field("Page range", Input(type="text", name="page_input", placeholder="1,3,5-7")),
            Details(
                Summary("More experimental options"),
                Div(
                    _field("number of threads", Input(type="number", min="1", step="1", name="threads", value="4")),
                    _checkbox("Skip font subsetting", "skip_subset_fonts"),
                    _checkbox("Ignore cache", "ignore_cache"),
                    _field(
                        "Custom formula font regex (vfont)",
                        Input(type="text", name="vfont", value=ConfigManager.get("PDF2ZH_VFONT", "")),
                    ),
                    _field(
                        "Translation Mode",
                        Select(_option("fast", "fast"), _option("precise", "fast"), name="mode_choice"),
                    ),
                    cls="stack",
                ),
            ),
            Input(type="hidden", name="recaptcha_response", id="recaptcha-response", value=""),
            Div(id="recaptcha-box"),
            Div(
                Button("Translate", type="submit"),
                Button(
                    "Cancel",
                    type="button",
                    hx_post="/cancel",
                    hx_include="[name='session_id']",
                    hx_target="#cancel-status",
                    cls="secondary",
                ),
                Span(id="cancel-status", cls="muted"),
                cls="actions",
            ),
            method="post",
            action="/translate",
            enctype="multipart/form-data",
            cls="stack",
        )
        details = Details(
            Summary("Technical details"),
            P(A("GitHub: Byaidu/PDFMathTranslate", href="https://github.com/Byaidu/PDFMathTranslate")),
            P(A("BabelDOC: funstory-ai/BabelDOC", href="https://github.com/funstory-ai/BabelDOC")),
            P(f"pdf2zh Version: {__version__}"),
            P(f"BabelDOC Version: {babeldoc_version}"),
            cls="muted",
        )
        return _page(
            Div(
                Div(H2("File"), form, details, cls="panel"),
                Div(_preview_panel(), _result_panel(), cls="stack"),
                cls="layout",
            )
        )

    @rt("/service-fields")
    def service_fields(req, service: str):
        auth = _authorized(req, user_list, auth_message)
        if auth:
            return auth
        if service not in service_map:
            service = enabled_services[0]
        return _service_env_fields(service)

    @rt("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    @rt("/cancel")
    async def cancel(req):
        auth = _authorized(req, user_list, auth_message)
        if auth:
            return auth
        form = await req.form()
        stop_translate_file(form.get("session_id"))
        return "Cancellation requested"

    @rt("/preview")
    async def preview(req):
        auth = _authorized(req, user_list, auth_message)
        if auth:
            return auth
        form = await req.form()
        upload = form.get("file_input")
        if not isinstance(upload, UploadFile) or not upload.filename:
            return _preview_panel()
        safe_name = os.path.basename(upload.filename)
        suffix = Path(safe_name).suffix.lower()
        if suffix != ".pdf":
            return _preview_panel()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        preview_path = OUTPUT_DIR / f"{uuid.uuid4()}-{safe_name}"
        preview_path.write_bytes(await upload.read())
        return _preview_panel(preview_path.name)

    @rt("/translate")
    async def translate(req):
        auth = _authorized(req, user_list, auth_message)
        if auth:
            return auth
        form = await req.form()
        upload = form.get("file_input")
        uploaded_path = None
        if isinstance(upload, UploadFile) and upload.filename:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = os.path.basename(upload.filename)
            uploaded_path = OUTPUT_DIR / f"{uuid.uuid4()}-{safe_name}"
            uploaded_path.write_bytes(await upload.read())

        def run_translation():
            try:
                mono, dual = translate_file(
                    form.get("file_type", "File"),
                    str(uploaded_path) if uploaded_path else "",
                    form.get("link_input", ""),
                    form.get("service", enabled_services[0]),
                    form.get("lang_from", "English"),
                    form.get("lang_to", "Simplified Chinese"),
                    form.get("page_range", "All"),
                    form.get("page_input", ""),
                    form.get("prompt", ""),
                    form.get("threads", "4"),
                    bool(form.get("skip_subset_fonts")),
                    bool(form.get("ignore_cache")),
                    form.get("vfont", ""),
                    form.get("mode_choice", "fast"),
                    form.get("recaptcha_response", ""),
                    form.get("session_id", ""),
                    form.get("env_0", ""),
                    form.get("env_1", ""),
                    form.get("env_2", ""),
                    form.get("env_3", ""),
                )
                return _page(
                    Div(A("Start another translation", href="/", cls="button secondary"), cls="actions"),
                    _result_panel(mono, dual),
                )
            except Exception as exc:
                logger.exception("GUI translation failed")
                return _page(
                    Div(A("Back", href="/", cls="button secondary"), cls="actions"),
                    _result_panel(error=str(exc) or exc.__class__.__name__),
                )

        return await anyio.to_thread.run_sync(run_translation)

    @rt("/file")
    def file(req, name: str):
        auth = _authorized(req, user_list, auth_message)
        if auth:
            return auth
        name = os.path.basename(name)
        path = (OUTPUT_DIR / name).resolve()
        root = OUTPUT_DIR.resolve()
        if root not in path.parents and path != root:
            return Response("Not found", status_code=404)
        if not path.exists() or not path.is_file():
            return Response("Not found", status_code=404)
        return FileResponse(path)

    return app


def setup_gui(
    share: bool = False, auth_file: list = ["", ""], server_port=7860
) -> None:
    user_list, html = parse_user_passwd(auth_file)
    app = create_app(user_list, html)

    if share:
        print("FastHTML does not provide a Gradio-style share tunnel.")

    import uvicorn

    bind_addresses = ["0.0.0.0", "127.0.0.1"]
    if _has_ipv6():
        bind_addresses.append("::")

    for addr in bind_addresses:
        try:
            print(f"Starting FastHTML GUI on http://{addr}:{server_port}")
            print(f"Open locally at http://127.0.0.1:{server_port}")
            webbrowser.open(f"http://127.0.0.1:{server_port}")
            uvicorn.run(app, host=addr, port=server_port)
            return
        except Exception:
            print(
                f"Error launching GUI using {addr}.\n"
                "This may be caused by global mode of proxy software."
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    setup_gui()
