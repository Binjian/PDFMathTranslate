import asyncio
import base64
import cgi
import json
import logging
import multiprocessing
import os
import shutil
import socket
import threading
import time
import uuid
import webbrowser
from asyncio import CancelledError
from pathlib import Path
from urllib.parse import quote
import typing as T

import anyio
from fasthtml.common import *
import requests
from starlette.datastructures import UploadFile
from starlette.responses import FileResponse, JSONResponse, Response
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
GUI_BACKEND = "auto"
GUI_ONNX: str | None = None

try:
    from babeldoc import __version__ as babeldoc_version
except Exception:
    babeldoc_version = "unknown"


class GuiError(RuntimeError):
    """User-facing GUI error."""


class _LazyModel:
    """Defers model loading until first access so the GUI starts instantly."""

    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from babeldoc.docvision.doclayout import OnnxModel

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
translation_jobs: dict[str, dict[str, T.Any]] = {}
GUI_LAST_SETTINGS_KEY = "PDF2ZH_GUI_LAST_SETTINGS"

UI_TEXT = {
    "en": {
        "title": "PDFMathTranslate - PDF Translation with preserved formats",
        "subtitle": "PDF translation with preserved formats",
        "show_controls": "Show controls",
        "language": "Interface",
        "english": "English",
        "chinese": "中文",
        "translation_failed": "Translation failed",
        "translated": "Translated",
        "no_result": "Run a translation to create output files.",
        "autohide": "Autohide",
        "mono": "Mono",
        "dual": "Dual",
        "download_mono": "Download Translation (Mono)",
        "download_dual": "Download Translation (Dual)",
        "translated_document": "Translated Document",
        "preview": "Preview",
        "document_preview": "Document Preview",
        "file_section": "File",
        "file_limited": "File | < 5 MB",
        "type": "Type",
        "file_choice": "File",
        "link_choice": "Link",
        "link": "Link",
        "option": "Option",
        "service": "Service",
        "translate_from": "Translate from",
        "translate_to": "Translate to",
        "pages": "Pages",
        "page_range": "Page range",
        "experimental_options": "More experimental options",
        "threads": "number of threads",
        "skip_subset_fonts": "Skip font subsetting",
        "ignore_cache": "Ignore cache",
        "vfont": "Custom formula font regex (vfont)",
        "translation_mode": "Translation Mode",
        "translate": "Translate",
        "cancel": "Cancel",
        "technical_details": "Technical details",
        "start_another": "Start another translation",
        "back": "Back",
        "custom_prompt": "Custom Prompt for llm",
        "progress_title": "Translating",
        "progress_starting": "Starting translation...",
        "progress_cancel": "Cancel translation",
        "progress_wait": "Preparing progress...",
        "run_settings": "Settings used for this translation",
        "elapsed_time": "Elapsed",
        "time_spent": "Time spent",
        "source": "Source",
        "yes": "Yes",
        "no": "No",
    },
    "zh": {
        "title": "PDFMathTranslate - 保留格式的 PDF 翻译",
        "subtitle": "保留格式的 PDF 翻译",
        "show_controls": "显示控制项",
        "language": "界面语言",
        "english": "English",
        "chinese": "中文",
        "translation_failed": "翻译失败",
        "translated": "译文",
        "no_result": "翻译完成后会在这里显示输出文件。",
        "autohide": "自动隐藏",
        "mono": "单页",
        "dual": "双页",
        "download_mono": "下载译文（单页）",
        "download_dual": "下载译文（双页）",
        "translated_document": "译文预览",
        "preview": "预览",
        "document_preview": "文档预览",
        "file_section": "文件",
        "file_limited": "文件 | < 5 MB",
        "type": "类型",
        "file_choice": "文件",
        "link_choice": "链接",
        "link": "链接",
        "option": "选项",
        "service": "服务",
        "translate_from": "源语言",
        "translate_to": "目标语言",
        "pages": "页面",
        "page_range": "页码范围",
        "experimental_options": "更多实验选项",
        "threads": "线程数",
        "skip_subset_fonts": "跳过字体子集化",
        "ignore_cache": "忽略缓存",
        "vfont": "自定义公式字体正则 (vfont)",
        "translation_mode": "翻译模式",
        "translate": "翻译",
        "cancel": "取消",
        "technical_details": "技术详情",
        "start_another": "继续翻译",
        "back": "返回",
        "custom_prompt": "大模型自定义提示词",
        "progress_title": "正在翻译",
        "progress_starting": "正在开始翻译...",
        "progress_cancel": "取消翻译",
        "progress_wait": "正在准备进度...",
        "run_settings": "本次翻译设置",
        "elapsed_time": "已用时间",
        "time_spent": "总用时",
        "source": "来源",
        "yes": "是",
        "no": "否",
    },
}


def _ui_lang(lang: str | None) -> str:
    lang = str(lang or "zh").lower()
    return "en" if lang.startswith("en") else "zh"


def _t(lang: str | None, key: str) -> str:
    lang = _ui_lang(lang)
    return UI_TEXT[lang].get(key, UI_TEXT["en"].get(key, key))


LANG_LABELS_ZH = {
    "Simplified Chinese": "简体中文",
    "Traditional Chinese": "繁体中文",
    "English": "英语",
    "French": "法语",
    "German": "德语",
    "Japanese": "日语",
    "Korean": "韩语",
    "Russian": "俄语",
    "Spanish": "西班牙语",
    "Italian": "意大利语",
}

PAGE_LABELS_ZH = {
    "All": "全部",
    "First": "第一页",
    "First 5 pages": "前 5 页",
    "First 20 pages": "前 20 页",
    "Others": "其他",
}

MODE_LABELS_ZH = {
    "fast": "快速",
    "precise": "精确",
}

OLLAMA_MODEL_OPTIONS = [
    "qwen3-coder:latest",
    "gemma4:latest",
    "qwen3.6:latest",
    "qwen3.5:latest",
    "deepseek-r1:32b",
    "qwen3-embedding:latest",
    "bge-m3:latest",
    "deepseek-r1:7b",
    "deepseek-r1:14b",
    "deepseek-r1:70b",
    "gemma3:27b",
    "deepseek-r1:1.5b",
]

OLLAMA_MODEL_FALLBACK_OPTIONS = OLLAMA_MODEL_OPTIONS
OLLAMA_HOST_OPTIONS = [
    "127.0.0.1:11434",
    "172.27.74.16:11434",
    "172.27.74.49:11434",
]


def _normalize_ollama_host(host: str | None) -> str:
    host = (host or "").strip()
    if host and "://" not in host:
        host = f"http://{host}"
    return host.rstrip("/")


def _ollama_model_options(host: str | None, selected: str | None = None) -> list[str]:
    models: list[str] = []
    host = _normalize_ollama_host(host)
    if host:
        try:
            response = requests.get(f"{host}/api/tags", timeout=2)
            response.raise_for_status()
            data = response.json()
            for model in data.get("models", []):
                name = model.get("name") if isinstance(model, dict) else None
                if name:
                    models.append(name)
        except Exception as exc:
            logger.info("Unable to query Ollama models from %s: %s", host, exc)
    if not models:
        models = list(OLLAMA_MODEL_FALLBACK_OPTIONS)
    if selected and selected not in models:
        models.insert(0, selected)
    return models


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


class _ProgressPipe:
    def __init__(self, conn):
        self.conn = conn

    def put(self, event: dict[str, T.Any]) -> None:
        try:
            self.conn.send(event)
        except (BrokenPipeError, EOFError, OSError):
            pass

    def close(self) -> None:
        try:
            self.conn.close()
        except OSError:
            pass


def stop_translate_file(session_id: str | None) -> None:
    if session_id and session_id in cancellation_event_map:
        logger.info("Stopping translation for session %s", session_id)
        cancellation_event_map[session_id].set()
    if session_id and session_id in translation_jobs:
        process = translation_jobs[session_id].get("process")
        if process and process.is_alive():
            process.terminate()
        translation_jobs[session_id]["message"] = "Cancellation requested"
        translation_jobs[session_id]["status"] = "error"
        translation_jobs[session_id]["error"] = "Translation cancelled"
        translation_jobs[session_id]["finished_at"] = time.time()


def shutdown_translation_jobs() -> None:
    for session_id, job in list(translation_jobs.items()):
        process = job.get("process")
        if not process:
            continue
        logger.info("Stopping translation worker for session %s", session_id)
        if process.is_alive():
            process.terminate()
            process.join(timeout=3)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
        job["status"] = "error"
        job["progress"] = 1.0
        job["message"] = "Server stopped"
        job["error"] = "Server stopped"
        job["finished_at"] = time.time()
        conn = job.pop("progress_conn", None)
        if conn:
            try:
                conn.close()
            except OSError:
                pass
        job.pop("process", None)
    cancellation_event_map.clear()


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
    progress_queue=None,
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
        if k == "OLLAMA_HOST" and v:
            _envs[k] = _normalize_ollama_host(str(v))
        if str(k).upper().endswith("API_KEY") and str(v) == "***":
            _envs[k] = ConfigManager.get_env_by_translatername(translator, k, None)

    def progress_bar(t: tqdm.tqdm):
        desc = getattr(t, "desc", "Translating...") or "Translating..."
        total = getattr(t, "total", 0) or 1
        if session_id in translation_jobs:
            translation_jobs[session_id].update(
                {
                    "progress": min(0.99, max(0.0, t.n / total)),
                    "message": desc,
                }
            )
        if progress_queue is not None:
            progress_queue.put(
                {
                    "type": "progress",
                    "progress": min(0.99, max(0.0, t.n / total)),
                    "message": desc,
                }
            )
        logger.info("%s %.0f%%", desc, 100 * t.n / total)

    try:
        threads = int(threads)
    except ValueError:
        threads = 1

    try:
        from pdf2zh.kernel import KernelRegistry
        from pdf2zh.kernel.protocol import TranslateRequest
        from pdf2zh.doclayout import ModelInstance, OnnxModel, set_backend

        set_backend(GUI_BACKEND)
        if GUI_ONNX and ModelInstance.value is None:
            ModelInstance.value = OnnxModel(GUI_ONNX)
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


def _translate_file_process(params: dict, progress_queue) -> None:
    try:
        mono, dual = translate_file(
            params["file_type"],
            params["file_input"],
            params["link_input"],
            params["service"],
            params["lang_from"],
            params["lang_to"],
            params["page_range"],
            params["page_input"],
            params["prompt"],
            params["threads"],
            params["skip_subset_fonts"],
            params["ignore_cache"],
            params["vfont"],
            params["mode_choice"],
            params["recaptcha_response"],
            params["session_id"],
            params["env_0"],
            params["env_1"],
            params["env_2"],
            params["env_3"],
            progress_queue=progress_queue,
        )
        progress_queue.put({"type": "done", "mono": mono, "dual": dual})
    except BaseException as exc:
        progress_queue.put(
            {
                "type": "error",
                "message": str(exc) or exc.__class__.__name__,
            }
        )
    finally:
        progress_queue.close()


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


def _value_option(value: str, label: str, selected: str | None = None):
    return Option(label, value=value, selected=value == selected)


def _lang_options(ui_lang: str, selected: str):
    labels = LANG_LABELS_ZH if _ui_lang(ui_lang) == "zh" else {}
    return [
        _value_option(lang, labels.get(lang, lang), selected)
        for lang in lang_map.keys()
    ]


def _page_options(ui_lang: str, selected: str):
    labels = PAGE_LABELS_ZH if _ui_lang(ui_lang) == "zh" else {}
    return [
        _value_option(page, labels.get(page, page), selected)
        for page in page_map.keys()
    ]


def _mode_options(ui_lang: str, selected: str):
    labels = MODE_LABELS_ZH if _ui_lang(ui_lang) == "zh" else {}
    return [
        _value_option(mode, labels.get(mode, mode), selected)
        for mode in ["fast", "precise"]
    ]


def _ollama_host_label(host: str) -> str:
    return host.removeprefix("http://").removeprefix("https://")


def _ollama_host_input(name: str, value: str | None, ui_lang: str):
    normalized_value = _normalize_ollama_host(value)
    options = [
        (_normalize_ollama_host(host), _ollama_host_label(host))
        for host in OLLAMA_HOST_OPTIONS
    ]
    option_values = {host for host, _ in options}
    manual_value = value or ""
    selected = normalized_value if normalized_value in option_values else "__manual__"
    if selected != "__manual__":
        manual_value = ""

    return Div(
        Input(
            type="hidden",
            name=name,
            id="ollama-host-value",
            value=normalized_value,
            hx_get="/ollama-models",
            hx_trigger="change delay:300ms",
            hx_target="#ollama-model-field",
            hx_include="[name='env_0'],[name='env_1']",
        ),
        Select(
            *[_value_option(host, label, selected) for host, label in options],
            Option(
                "手动输入" if _ui_lang(ui_lang) == "zh" else "Manual input",
                value="__manual__",
                selected=selected == "__manual__",
            ),
            id="ollama-host-preset",
            onchange=(
                "const hidden=document.getElementById('ollama-host-value');"
                "const manual=document.getElementById('ollama-host-manual');"
                "if(this.value==='__manual__'){"
                "manual.style.display='block'; hidden.value=manual.value;"
                "}else{"
                "manual.style.display='none'; hidden.value=this.value;"
                "}"
                "hidden.dispatchEvent(new Event('change',{bubbles:true}));"
            ),
        ),
        Input(
            type="text",
            id="ollama-host-manual",
            value=manual_value,
            placeholder="172.27.74.16:11434",
            autocomplete="off",
            style="" if selected == "__manual__" else "display:none",
            oninput=(
                "const hidden=document.getElementById('ollama-host-value');"
                "hidden.value=this.value;"
                "hidden.dispatchEvent(new Event('change',{bubbles:true}));"
            ),
        ),
        cls="ollama-host-field",
    )


def _field(label: str, child):
    return Label(Span(label), child)


def _checkbox(label: str, name: str, checked: bool = False):
    return Label(Input(type="checkbox", name=name, value="true", checked=checked), label)


def _service_env_fields(
    service: str,
    ui_lang: str = "zh",
    env_overrides: dict[str, str] | None = None,
    prompt_value: str = "",
):
    env_overrides = env_overrides or {}
    translator = service_map[service]
    fields = [Input(type="hidden", name=f"env_{i}", value="") for i in range(4)]
    env_values: dict[str, str] = {}
    for i, env in enumerate(translator.envs.items()):
        label = env[0]
        try:
            configured_value = ConfigManager.get_env_by_translatername(
                translator,
                env[0],
                env[1],
            )
        except (KeyError, TypeError):
            configured_value = env[1]
        value = env_overrides.get(f"env_{i}", configured_value)
        env_values[label] = value
        input_type = "password" if "API_KEY" in label.upper() else "text"
        if hidden_secret_details and "MODEL" not in str(label).upper() and value:
            value = "***" if "API_KEY" in label.upper() else value
        if service == "Ollama" and label == "OLLAMA_HOST":
            child = _ollama_host_input(f"env_{i}", value, ui_lang)
            fields[i] = _field(label, child)
            continue
        if service == "Ollama" and label == "OLLAMA_MODEL":
            choices = _ollama_model_options(env_values.get("OLLAMA_HOST"), value)
            child = Select(
                *[_value_option(choice, choice, value) for choice in choices],
                name=f"env_{i}",
            )
            fields[i] = Div(_field(label, child), id="ollama-model-field")
            continue
        else:
            child = Input(
                type=input_type,
                name=f"env_{i}",
                value=value or "",
                autocomplete="off",
            )
        fields[i] = _field(label, child)
    if translator.CustomPrompt:
        fields.append(
            _field(
                _t(ui_lang, "custom_prompt"),
                Textarea(prompt_value or "", name="prompt", rows=5),
            )
        )
    else:
        fields.append(Input(type="hidden", name="prompt", value=""))
    return Div(*fields, id="env-fields", cls="stack")


def _output_file_path(name: str) -> Path | None:
    name = os.path.basename(name)
    path = (OUTPUT_DIR / name).resolve()
    root = OUTPUT_DIR.resolve()
    if root not in path.parents and path != root:
        return None
    if not path.exists() or not path.is_file():
        return None
    return path


def _translated_download_name(name: str, variant: str) -> str:
    stem = Path(os.path.basename(name)).stem
    suffix = f"-{variant}"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    parts = stem.split("-", 5)
    if len(parts) == 6:
        try:
            uuid.UUID("-".join(parts[:5]))
            stem = parts[5]
        except ValueError:
            pass
    return f"{stem}_{variant}.pdf"


def _format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        seconds = 0
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def _job_elapsed_seconds(job: dict[str, T.Any]) -> float:
    started_at = job.get("started_at")
    if not started_at:
        return 0
    finished_at = job.get("finished_at") or time.time()
    return max(0, float(finished_at) - float(started_at))


def _result_panel(
    mono: str | None = None,
    dual: str | None = None,
    error: str | None = None,
    autohide: bool = False,
    ui_lang: str = "zh",
    elapsed_seconds: float | None = None,
):
    if error:
        return Div(H2(_t(ui_lang, "translation_failed")), P(error), cls="result error", id="result")
    if not mono or not dual:
        return Div(H2(_t(ui_lang, "translated")), P(_t(ui_lang, "no_result")), id="result")
    mono_name = Path(mono).name
    dual_name = Path(dual).name
    mono_url = f"/file?name={quote(mono_name)}"
    dual_url = f"/file?name={quote(dual_name)}"
    mono_view_url = f"{mono_url}#view=FitH"
    dual_view_url = f"/pdf-viewer?name={quote(dual_name)}&view=facing"
    mono_download_name = _translated_download_name(mono_name, "mono")
    dual_download_name = _translated_download_name(dual_name, "dual")
    mono_download_url = f"/download?name={quote(mono_name)}&variant=mono"
    dual_download_url = f"/download?name={quote(dual_name)}&variant=dual"
    return Div(
        Div(
            H2(_t(ui_lang, "translated")),
            Span(
                f"{_t(ui_lang, 'time_spent')}: {_format_duration(elapsed_seconds)}",
                cls="muted elapsed-time",
            ),
            Div(
                Label(
                    Input(
                        type="checkbox",
                        id="result-autohide-toggle",
                        checked=autohide,
                    ),
                    _t(ui_lang, "autohide"),
                ),
                cls="toggle-row",
            ),
            Div(
                Label(
                    Input(
                        type="radio",
                        name="translated_view",
                        value="mono",
                        data_url=mono_view_url,
                    ),
                    _t(ui_lang, "mono"),
                ),
                Label(
                    Input(
                        type="radio",
                        name="translated_view",
                        value="dual",
                        checked=True,
                        data_url=dual_view_url,
                    ),
                    _t(ui_lang, "dual"),
                ),
                cls="radio-row",
            ),
            Div(
                A(
                    _t(ui_lang, "download_mono"),
                    href=mono_download_url,
                    download=mono_download_name,
                    cls="button",
                ),
                A(
                    _t(ui_lang, "download_dual"),
                    href=dual_download_url,
                    download=dual_download_name,
                    cls="button secondary",
                ),
                cls="actions",
            ),
            cls="result-toolbar",
        ),
        Iframe(id="translated-frame", src=dual_view_url, title=_t(ui_lang, "translated_document")),
        Script(
            """
            document.querySelectorAll('input[name="translated_view"]').forEach((input) => {
                input.addEventListener('change', (event) => {
                    document.getElementById('translated-frame').src = event.target.dataset.url;
                });
            });
            document.getElementById('result-autohide-toggle')?.addEventListener('change', (event) => {
                document.querySelector('.app-shell')?.classList.toggle('autohide', event.target.checked);
            });
            """
        ),
        id="result",
        cls="result",
    )


def _preview_panel(filename: str | None = None, autohide: bool = False, ui_lang: str = "zh"):
    src = f"/file?name={quote(filename)}" if filename else ""
    return Div(
        H2(_t(ui_lang, "preview")),
        Iframe(src=src, title=_t(ui_lang, "document_preview")),
        id="preview-panel",
        cls="preview",
        data_autohide="true" if autohide else "false",
        data_ui_lang=_ui_lang(ui_lang),
    )


def _bool_label(value: bool, ui_lang: str) -> str:
    return _t(ui_lang, "yes" if value else "no")


def _run_settings(params: dict[str, T.Any], ui_lang: str) -> list[tuple[str, str]]:
    service = str(params.get("service") or enabled_services[0])
    file_type = str(params.get("file_type") or "File")
    if file_type == "File":
        source = Path(str(params.get("file_input") or "")).name
    else:
        source = str(params.get("link_input") or "")

    page_range = str(params.get("page_range") or "All")
    pages = (PAGE_LABELS_ZH.get(page_range, page_range) if _ui_lang(ui_lang) == "zh" else page_range)
    if page_range == "Others" and params.get("page_input"):
        pages = f"{pages}: {params['page_input']}"

    mode = str(params.get("mode_choice") or "fast")
    if _ui_lang(ui_lang) == "zh":
        mode = MODE_LABELS_ZH.get(mode, mode)

    rows = [
        (_t(ui_lang, "source"), source or "-"),
        (_t(ui_lang, "service"), service),
        (_t(ui_lang, "translate_from"), str(params.get("lang_from") or "-")),
        (_t(ui_lang, "translate_to"), str(params.get("lang_to") or "-")),
        (_t(ui_lang, "pages"), pages),
        (_t(ui_lang, "threads"), str(params.get("threads") or "-")),
        (_t(ui_lang, "translation_mode"), mode),
        (_t(ui_lang, "skip_subset_fonts"), _bool_label(bool(params.get("skip_subset_fonts")), ui_lang)),
        (_t(ui_lang, "ignore_cache"), _bool_label(bool(params.get("ignore_cache")), ui_lang)),
    ]
    if params.get("vfont"):
        rows.append((_t(ui_lang, "vfont"), str(params["vfont"])))
    if params.get("prompt"):
        rows.append((_t(ui_lang, "custom_prompt"), str(params["prompt"])))

    translator = service_map.get(service)
    if translator:
        for i, env_name in enumerate(translator.envs.keys()):
            if str(env_name).upper().endswith("API_KEY"):
                continue
            value = str(params.get(f"env_{i}") or "").strip()
            if value:
                rows.append((env_name, value))
    return rows


def _run_settings_panel(settings: list[tuple[str, str]], ui_lang: str):
    if not settings:
        return ""
    return Details(
        Summary(_t(ui_lang, "run_settings")),
        Table(
            Tbody(
                *[
                    Tr(Th(label, scope="row"), Td(value))
                    for label, value in settings
                ]
            )
        ),
        open=True,
        cls="run-settings",
    )


def _sanitize_saved_params(params: dict[str, T.Any]) -> dict[str, T.Any]:
    saved_keys = {
        "file_type",
        "link_input",
        "service",
        "lang_from",
        "lang_to",
        "page_range",
        "page_input",
        "prompt",
        "threads",
        "skip_subset_fonts",
        "ignore_cache",
        "vfont",
        "mode_choice",
    }
    saved = {key: params.get(key, "") for key in saved_keys}
    saved["file_input"] = ""
    translator = service_map.get(str(params.get("service") or ""))
    if translator:
        for i, env_name in enumerate(translator.envs.keys()):
            if str(env_name).upper().endswith("API_KEY"):
                continue
            saved[f"env_{i}"] = params.get(f"env_{i}", "")
    return saved


def _save_last_gui_settings(
    params: dict[str, T.Any],
    ui_lang: str,
    autohide: bool,
) -> None:
    try:
        ConfigManager.set(
            GUI_LAST_SETTINGS_KEY,
            {
                "params": _sanitize_saved_params(params),
                "ui_lang": _ui_lang(ui_lang),
                "autohide": bool(autohide),
            },
        )
    except Exception as exc:
        logger.warning("Unable to save GUI settings: %s", exc)


def _load_last_gui_settings() -> dict[str, T.Any]:
    try:
        settings = ConfigManager.get(GUI_LAST_SETTINGS_KEY)
    except Exception as exc:
        logger.warning("Unable to load GUI settings: %s", exc)
        return {}
    return settings if isinstance(settings, dict) else {}


def _progress_page(session_id: str, ui_lang: str = "zh", autohide: bool = False):
    job = translation_jobs.get(session_id, {})
    settings = job.get("settings", [])
    return _page(
        Div(
            H2(_t(ui_lang, "progress_title")),
            Progress(id="translation-progress", value="0", max="100"),
            P(_t(ui_lang, "progress_wait"), id="translation-progress-text", cls="muted"),
            P(
                f"{_t(ui_lang, 'elapsed_time')}: {_format_duration(_job_elapsed_seconds(job))}",
                id="translation-elapsed-time",
                cls="muted elapsed-time",
            ),
            _run_settings_panel(settings, ui_lang),
            Form(
                Input(type="hidden", name="session_id", value=session_id),
                Button(
                    _t(ui_lang, "progress_cancel"),
                    type="button",
                    hx_post="/cancel",
                    hx_include="closest form",
                    hx_target="#translation-progress-text",
                    cls="secondary",
                ),
            ),
            Script(
                f"""
                const progressBar = document.getElementById('translation-progress');
                const progressText = document.getElementById('translation-progress-text');
                const elapsedTime = document.getElementById('translation-elapsed-time');
                async function pollTranslationProgress() {{
                    const response = await fetch('/progress/{session_id}');
                    const state = await response.json();
                    progressBar.value = Math.round((state.progress || 0) * 100);
                    progressText.textContent = state.message || '';
                    elapsedTime.textContent = `{_t(ui_lang, 'elapsed_time')}: ${{state.elapsed_text || '0:00'}}`;
                    if (state.status === 'done' || state.status === 'error') {{
                        window.location = '/result/{session_id}?ui_lang={_ui_lang(ui_lang)}';
                        return;
                    }}
                    setTimeout(pollTranslationProgress, 1000);
                }}
                pollTranslationProgress();
                """
            ),
            cls="result",
        ),
        autohide=autohide,
        ui_lang=ui_lang,
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


def _page(*children, autohide: bool = False, ui_lang: str = "zh"):
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
        Title(_t(ui_lang, "title")),
        *recaptcha,
        Main(
            Header(
                H1(A("PDFMathTranslate on FastHTML", href="https://github.com/Binjian/PDFMathTranslate")),
                P(_t(ui_lang, "subtitle")),
                cls="page-header",
            ),
            Button(
                _t(ui_lang, "show_controls"),
                type="button",
                cls="autohide-exit secondary",
                onclick=(
                    "document.querySelector('.app-shell')?.classList.remove('autohide');"
                    "const toggle=document.getElementById('autohide-toggle');"
                    "if(toggle) toggle.checked=false;"
                ),
            ),
            *children,
            cls=f"app-shell{' autohide' if autohide else ''}",
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
                html { font-size: clamp(13px, .78vw + 8px, 16px); }
                body { background: #f7f8fb; }
                .app-shell { width: 100%; max-width: none; padding: clamp(.4rem, 1.1vw, 1rem); }
                header { margin-bottom: clamp(.35rem, 1vh, .9rem); }
                header h1 { font-size: clamp(1.15rem, 1.2vw + .85rem, 1.7rem); margin-bottom: .15rem; line-height: 1.15; }
                header p { margin-bottom: 0; font-size: .9rem; }
                h2 { font-size: clamp(1rem, .7vw + .85rem, 1.25rem); margin-bottom: .45rem; }
                label, summary, input, select, textarea, button, .button { font-size: .9rem; }
                input, select, textarea { min-height: 2rem; padding: .35rem .5rem; margin-bottom: .45rem; }
                input[type="checkbox"], input[type="radio"] { min-height: 0; }
                button, .button, [role="button"] { padding: .4rem .7rem; margin-bottom: 0; }
                .layout { display: grid; grid-template-columns: clamp(280px, 25vw, 420px) minmax(0, 1fr); gap: clamp(.5rem, 1vw, 1rem); align-items: start; }
                .control-panel, .panel { background: #fff; border: 1px solid #dfe3ea; border-radius: 8px; padding: clamp(.45rem, .75vw, .7rem); }
                .control-panel { max-height: calc(100dvh - 4.75rem); overflow-y: auto; overflow-x: hidden; scrollbar-gutter: stable; }
                .control-panel h2 { margin: 0 0 .3rem; font-size: .98rem; line-height: 1.15; }
                .control-panel form { display: grid; gap: .25rem; }
                .control-panel form > script { display: none; }
                .control-panel label { margin: 0; line-height: 1.18; }
                .control-panel label > span { display: block; margin-bottom: .1rem; color: #526071; font-size: .76rem; font-weight: 600; }
                .control-panel input, .control-panel select, .control-panel textarea { width: 100%; min-height: 1.65rem; padding: .22rem .4rem; margin: 0; line-height: 1.2; }
                .control-panel input[type="file"] { min-height: 1.8rem; padding: .18rem .35rem; font-size: .78rem; }
                .control-panel input[type="checkbox"], .control-panel input[type="radio"] { width: auto; min-height: 0; }
                .control-panel button, .control-panel .button, .control-panel [role="button"] { padding: .28rem .55rem; font-size: .84rem; line-height: 1.2; }
                .control-panel details { margin-top: .25rem; }
                .control-panel summary { padding: .15rem 0; line-height: 1.2; }
                .control-panel details > div { margin-top: .25rem; }
                .control-panel .muted { font-size: .75rem; }
                .preview, .result { width: 100%; }
                .result { grid-column: 1 / -1; }
                .stack { display: grid; gap: .3rem; }
                .ollama-host-field { display: grid; gap: .25rem; }
                .split { display: grid; grid-template-columns: 1fr 1fr; gap: .35rem; }
                .run-settings { margin-top: .75rem; }
                .run-settings summary { font-weight: 600; }
                .run-settings table { margin: .35rem 0 .75rem; }
                .run-settings th, .run-settings td { padding: .25rem .4rem; vertical-align: top; }
                .run-settings th { width: 12rem; color: #526071; }
                .actions { display: flex; flex-wrap: wrap; gap: .45rem; align-items: center; }
                .result-toolbar { display: flex; flex-wrap: wrap; gap: .75rem 1rem; align-items: center; margin-bottom: .75rem; }
                .result-toolbar h2 { margin: 0; }
                .elapsed-time { white-space: nowrap; }
                .result-toolbar .toggle-row, .result-toolbar .radio-row, .result-toolbar .actions { margin: 0; }
                .toggle-row { display: flex; align-items: center; gap: .5rem; margin-bottom: .25rem; }
                .toggle-row label { display: inline-flex; width: auto; gap: .35rem; align-items: center; margin: 0; }
                .autohide-exit { display: none; position: fixed; top: .5rem; right: 1rem; z-index: 10; width: auto; padding: .35rem .65rem; }
                .radio-row { display: flex; gap: 1rem; align-items: center; margin-bottom: .45rem; }
                .radio-row label { display: inline-flex; width: auto; gap: .35rem; align-items: center; margin: 0; }
                .radio-row input, .toggle-row input { margin: 0; }
                .secondary { background: #eef2f7; color: #243042; border-color: #d8dee8; }
                .muted { color: #687386; font-size: .85rem; }
                .error { border-color: #d33; color: #9b1c1c; }
                iframe { display: block; width: 100%; height: calc(100vh - clamp(7rem, 12vh, 10rem)); min-height: 24rem; border: 1px solid #dfe3ea; border-radius: 8px; background: #fff; }
                .autohide { padding-top: .5rem; }
                .autohide .page-header, .autohide .control-panel { display: none; }
                .autohide .autohide-exit { display: inline-flex; }
                .autohide .layout { display: block; }
                .autohide .preview h2, .autohide .result h2 { margin-bottom: .5rem; }
                .autohide iframe { height: calc(100vh - 6.5rem); }
                .autohide .result-toolbar { flex-wrap: nowrap; overflow-x: auto; white-space: nowrap; margin-right: 8.5rem; margin-bottom: .35rem; padding-bottom: .1rem; }
                .autohide .result-toolbar .actions { flex-wrap: nowrap; }
                .autohide .result-toolbar h2 { font-size: 1rem; line-height: 1.2; margin: 0; }
                .autohide .result-toolbar a.button, .autohide .result-toolbar button, .autohide .result-toolbar [role="button"] { width: auto; padding: .25rem .55rem; margin: 0; font-size: .875rem; }
                .autohide .result #translated-frame { height: calc(100vh - 3.25rem); }
                details { margin-top: .5rem; }
                details > div { margin-top: .35rem; }
                @media (min-width: 1500px) {
                    .layout { grid-template-columns: clamp(320px, 22vw, 460px) minmax(0, 1fr); }
                    iframe { height: calc(100vh - 8.25rem); }
                }
                @media (max-width: 1200px) {
                    .layout { grid-template-columns: minmax(260px, 34vw) minmax(0, 1fr); }
                    .control-panel { padding: .5rem; max-height: calc(100dvh - 4rem); }
                    iframe { height: calc(100vh - 7.5rem); min-height: 22rem; }
                }
                @media (max-width: 900px) {
                    .layout, .split { grid-template-columns: 1fr; }
                    .control-panel { max-width: none; max-height: none; overflow: visible; }
                    iframe { height: 62vh; min-height: 20rem; }
                    header { margin-bottom: .75rem; }
                    header h1 { font-size: 1.45rem; }
                }
                @media (max-width: 560px) {
                    .app-shell { padding: .5rem; }
                    .control-panel, .panel { padding: .45rem; }
                    iframe { height: 56vh; min-height: 18rem; }
                    .actions { gap: .5rem; }
                }
                @media (max-height: 760px) and (min-width: 901px) {
                    html { font-size: 13px; }
                    header { margin-bottom: .35rem; }
                    header p { display: none; }
                    h2 { margin-bottom: .3rem; }
                    .control-panel { max-height: calc(100dvh - 3rem); }
                    .control-panel form { gap: .18rem; }
                    .control-panel .stack { gap: .2rem; }
                    .control-panel input, .control-panel select, .control-panel textarea { min-height: 1.45rem; padding-top: .16rem; padding-bottom: .16rem; }
                    .control-panel h2 { font-size: .9rem; margin-bottom: .2rem; }
                    .control-panel label > span { font-size: .7rem; }
                    iframe { height: calc(100vh - 4.25rem); min-height: 18rem; }
                }
                """
            ),
        ),
    )

    @rt("/")
    def index(req, ui_lang: str = "zh", settings_from: str = ""):
        auth = _authorized(req, user_list, auth_message)
        if auth:
            return auth
        ui_lang = _ui_lang(ui_lang)
        session_id = str(uuid.uuid4())
        seed_job = translation_jobs.get(settings_from) if settings_from else None
        saved_settings = {} if seed_job else _load_last_gui_settings()
        seed_params = dict(
            seed_job.get("params", {})
            if seed_job
            else saved_settings.get("params", {})
        )
        default_service = str(seed_params.get("service") or enabled_services[0])
        if default_service not in enabled_services:
            default_service = enabled_services[0]
        file_type = str(seed_params.get("file_type") or "File")
        lang_from = str(seed_params.get("lang_from") or ConfigManager.get("PDF2ZH_LANG_FROM", "English"))
        lang_to = str(seed_params.get("lang_to") or ConfigManager.get("PDF2ZH_LANG_TO", "Simplified Chinese"))
        page_range = str(seed_params.get("page_range") or list(page_map.keys())[0])
        if page_range not in page_map:
            page_range = list(page_map.keys())[0]
        autohide_checked = bool(
            seed_job.get("autohide") if seed_job else saved_settings.get("autohide")
        )
        env_overrides = {
            f"env_{i}": str(seed_params.get(f"env_{i}") or "")
            for i in range(4)
            if seed_params.get(f"env_{i}")
        }
        settings_query = f"&settings_from={quote(settings_from)}" if seed_job else ""
        form = Form(
            Input(type="hidden", name="session_id", value=session_id),
            Input(type="hidden", name="ui_lang", value=ui_lang),
            _field(
                _t(ui_lang, "language"),
                Select(
                    Option(_t(ui_lang, "english"), value="en", selected=ui_lang == "en"),
                    Option(_t(ui_lang, "chinese"), value="zh", selected=ui_lang == "zh"),
                    name="ui_lang_selector",
                    onchange=f"window.location='/?ui_lang=' + this.value + '{settings_query}'",
                ),
            ),
            Div(
                Label(
                    Input(
                        type="checkbox",
                        name="autohide",
                        value="true",
                        id="autohide-toggle",
                        checked=autohide_checked,
                    ),
                    _t(ui_lang, "autohide"),
                ),
                cls="toggle-row",
            ),
            Script(
                """
                document.getElementById('autohide-toggle')?.addEventListener('change', (event) => {
                    document.querySelector('.app-shell')?.classList.toggle('autohide', event.target.checked);
                });
                document.body.addEventListener('htmx:afterSwap', (event) => {
                    if (event.detail.target?.id === 'preview-panel') {
                        const panel = event.detail.target;
                        const enabled = panel.dataset.autohide === 'true';
                        document.getElementById('autohide-toggle').checked = enabled;
                        document.querySelector('.app-shell')?.classList.toggle('autohide', enabled);
                    }
                });
                """
            ),
            _field(
                _t(ui_lang, "type"),
                Select(
                    Option(_t(ui_lang, "file_choice"), value="File", selected=file_type == "File"),
                    Option(_t(ui_lang, "link_choice"), value="Link", selected=file_type == "Link"),
                    name="file_type",
                ),
            ),
            Div(
                _field(
                    _t(ui_lang, "file_limited" if flag_demo else "file_section"),
                    Input(
                        type="file",
                        name="file_input",
                        accept=".pdf,.doc,.docx",
                        hx_post="/preview",
                        hx_trigger="change",
                        hx_target="#preview-panel",
                        hx_swap="outerHTML",
                        hx_encoding="multipart/form-data",
                        hx_include="[name='autohide'],[name='ui_lang']",
                    ),
                ),
                _field(
                    _t(ui_lang, "link"),
                    Input(
                        type="url",
                        name="link_input",
                        value=str(seed_params.get("link_input") or ""),
                        placeholder="https://...",
                    ),
                ),
                cls="stack",
            ),
            H2(_t(ui_lang, "option")),
            _field(
                _t(ui_lang, "service"),
                Select(
                    *[_option(service, default_service) for service in enabled_services],
                    name="service",
                    hx_get="/service-fields",
                    hx_target="#env-fields",
                    hx_trigger="change",
                    hx_include="[name='service'],[name='ui_lang']",
                ),
            ),
            _service_env_fields(
                default_service,
                ui_lang,
                env_overrides=env_overrides,
                prompt_value=str(seed_params.get("prompt") or ""),
            ),
            Div(
                _field(
                    _t(ui_lang, "translate_from"),
                    Select(
                        *_lang_options(ui_lang, lang_from),
                        name="lang_from",
                    ),
                ),
                _field(
                    _t(ui_lang, "translate_to"),
                    Select(
                        *_lang_options(ui_lang, lang_to),
                        name="lang_to",
                    ),
                ),
                cls="split",
            ),
            _field(
                _t(ui_lang, "pages"),
                Select(
                    *_page_options(ui_lang, page_range),
                    name="page_range",
                ),
            ),
            _field(
                _t(ui_lang, "page_range"),
                Input(
                    type="text",
                    name="page_input",
                    value=str(seed_params.get("page_input") or ""),
                    placeholder="1,3,5-7",
                ),
            ),
            Details(
                Summary(_t(ui_lang, "experimental_options")),
                Div(
                    _field(
                        _t(ui_lang, "threads"),
                        Input(
                            type="number",
                            min="1",
                            step="1",
                            name="threads",
                            value=str(seed_params.get("threads") or "4"),
                        ),
                    ),
                    _checkbox(
                        _t(ui_lang, "skip_subset_fonts"),
                        "skip_subset_fonts",
                        checked=bool(seed_params.get("skip_subset_fonts")),
                    ),
                    _checkbox(
                        _t(ui_lang, "ignore_cache"),
                        "ignore_cache",
                        checked=bool(seed_params.get("ignore_cache")),
                    ),
                    _field(
                        _t(ui_lang, "vfont"),
                        Input(
                            type="text",
                            name="vfont",
                            value=str(
                                seed_params.get("vfont")
                                if seed_params.get("vfont") is not None
                                else ConfigManager.get("PDF2ZH_VFONT", "")
                            ),
                        ),
                    ),
                    _field(
                        _t(ui_lang, "translation_mode"),
                        Select(
                            *_mode_options(
                                ui_lang,
                                str(seed_params.get("mode_choice") or "fast"),
                            ),
                            name="mode_choice",
                        ),
                    ),
                    cls="stack",
                ),
            ),
            Input(type="hidden", name="recaptcha_response", id="recaptcha-response", value=""),
            Div(id="recaptcha-box"),
            Div(
                Button(_t(ui_lang, "translate"), type="submit"),
                Button(
                    _t(ui_lang, "cancel"),
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
            Summary(_t(ui_lang, "technical_details")),
            P(A("GitHub: Byaidu/PDFMathTranslate", href="https://github.com/Byaidu/PDFMathTranslate")),
            P(A("BabelDOC: funstory-ai/BabelDOC", href="https://github.com/funstory-ai/BabelDOC")),
            P(f"pdf2zh Version: {__version__}"),
            P(f"BabelDOC Version: {babeldoc_version}"),
            cls="muted",
        )
        return _page(
            Div(
                Div(H2(_t(ui_lang, "file_section")), form, details, cls="control-panel"),
                Div(_preview_panel(ui_lang=ui_lang), cls="stack"),
                cls="layout",
            ),
            ui_lang=ui_lang,
        )

    @rt("/service-fields")
    def service_fields(req, service: str, ui_lang: str = "zh"):
        auth = _authorized(req, user_list, auth_message)
        if auth:
            return auth
        if service not in service_map:
            service = enabled_services[0]
        return _service_env_fields(service, ui_lang)

    @rt("/ollama-models")
    def ollama_models(req, env_0: str = "", env_1: str = ""):
        auth = _authorized(req, user_list, auth_message)
        if auth:
            return auth
        host = env_0 or service_map["Ollama"].envs["OLLAMA_HOST"]
        selected = env_1 or service_map["Ollama"].envs["OLLAMA_MODEL"]
        choices = _ollama_model_options(host, selected)
        return Div(
            _field(
                "OLLAMA_MODEL",
                Select(
                    *[_value_option(choice, choice, selected) for choice in choices],
                    name="env_1",
                ),
            ),
            id="ollama-model-field",
        )

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
        autohide = bool(form.get("autohide"))
        ui_lang = _ui_lang(form.get("ui_lang"))
        upload = form.get("file_input")
        if not isinstance(upload, UploadFile) or not upload.filename:
            return _preview_panel(autohide=autohide, ui_lang=ui_lang)
        safe_name = os.path.basename(upload.filename)
        suffix = Path(safe_name).suffix.lower()
        if suffix != ".pdf":
            return _preview_panel(autohide=autohide, ui_lang=ui_lang)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        preview_path = OUTPUT_DIR / f"{uuid.uuid4()}-{safe_name}"
        preview_path.write_bytes(await upload.read())
        return _preview_panel(preview_path.name, autohide=autohide, ui_lang=ui_lang)

    @rt("/translate")
    async def translate(req):
        auth = _authorized(req, user_list, auth_message)
        if auth:
            return auth
        form = await req.form()
        autohide = bool(form.get("autohide"))
        ui_lang = _ui_lang(form.get("ui_lang"))
        upload = form.get("file_input")
        uploaded_path = None
        if isinstance(upload, UploadFile) and upload.filename:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = os.path.basename(upload.filename)
            uploaded_path = OUTPUT_DIR / f"{uuid.uuid4()}-{safe_name}"
            uploaded_path.write_bytes(await upload.read())
        session_id = form.get("session_id") or str(uuid.uuid4())
        translation_jobs[session_id] = {
            "status": "running",
            "progress": 0.0,
            "message": _t(ui_lang, "progress_starting"),
            "autohide": autohide,
            "ui_lang": ui_lang,
            "started_at": time.time(),
        }
        params = {
            "file_type": form.get("file_type", "File"),
            "file_input": str(uploaded_path) if uploaded_path else "",
            "link_input": form.get("link_input", ""),
            "service": form.get("service", enabled_services[0]),
            "lang_from": form.get("lang_from", "English"),
            "lang_to": form.get("lang_to", "Simplified Chinese"),
            "page_range": form.get("page_range", "All"),
            "page_input": form.get("page_input", ""),
            "prompt": form.get("prompt", ""),
            "threads": form.get("threads", "4"),
            "skip_subset_fonts": bool(form.get("skip_subset_fonts")),
            "ignore_cache": bool(form.get("ignore_cache")),
            "vfont": form.get("vfont", ""),
            "mode_choice": form.get("mode_choice", "fast"),
            "recaptcha_response": form.get("recaptcha_response", ""),
            "session_id": session_id,
            "env_0": form.get("env_0", ""),
            "env_1": form.get("env_1", ""),
            "env_2": form.get("env_2", ""),
            "env_3": form.get("env_3", ""),
        }
        translation_jobs[session_id]["params"] = dict(params)
        translation_jobs[session_id]["settings"] = _run_settings(params, ui_lang)
        _save_last_gui_settings(params, ui_lang, autohide)

        def run_translation_job():
            ctx = multiprocessing.get_context("spawn")
            progress_conn, worker_conn = ctx.Pipe(duplex=False)
            progress_pipe = _ProgressPipe(worker_conn)
            process = ctx.Process(
                target=_translate_file_process,
                args=(params, progress_pipe),
            )
            translation_jobs[session_id]["process"] = process
            translation_jobs[session_id]["progress_conn"] = progress_conn
            process.start()
            worker_conn.close()
            while True:
                event = None
                try:
                    if progress_conn.poll(0.5):
                        event = progress_conn.recv()
                except (EOFError, OSError):
                    event = None

                if event:
                    event_type = event.get("type")
                    if event_type == "progress":
                        translation_jobs[session_id].update(
                            {
                                "progress": event.get("progress", 0.0),
                                "message": event.get("message", ""),
                            }
                        )
                    elif event_type == "done":
                        finished_at = time.time()
                        translation_jobs[session_id].update(
                            {
                                "status": "done",
                                "progress": 1.0,
                                "message": _t(ui_lang, "translated"),
                                "mono": event["mono"],
                                "dual": event["dual"],
                                "finished_at": finished_at,
                            }
                        )
                        break
                    elif event_type == "error":
                        finished_at = time.time()
                        translation_jobs[session_id].update(
                            {
                                "status": "error",
                                "progress": 1.0,
                                "message": event.get("message", "Unknown error"),
                                "error": event.get("message", "Unknown error"),
                                "finished_at": finished_at,
                            }
                        )
                        break

                if not process.is_alive():
                    exitcode = process.exitcode
                    if translation_jobs[session_id].get("status") in {"done", "error"}:
                        break
                    if exitcode == 0:
                        message = "Translation worker finished without returning a result."
                    else:
                        message = (
                            f"Translation worker crashed with exit code {exitcode}. "
                            "The GUI is still running; try another model or reduce concurrency."
                        )
                    translation_jobs[session_id].update(
                        {
                            "status": "error",
                            "progress": 1.0,
                            "message": message,
                            "error": message,
                            "finished_at": time.time(),
                        }
                    )
                    break

            process.join(timeout=1)
            translation_jobs[session_id].pop("process", None)
            translation_jobs[session_id].pop("progress_conn", None)
            try:
                progress_conn.close()
            except OSError:
                pass

        threading.Thread(target=run_translation_job, daemon=True).start()
        return _progress_page(session_id, ui_lang=ui_lang, autohide=autohide)

    @rt("/progress/{session_id}")
    def progress(req, session_id: str):
        auth = _authorized(req, user_list, auth_message)
        if auth:
            return auth
        job = translation_jobs.get(session_id)
        if not job:
            return JSONResponse(
                {"status": "error", "progress": 1.0, "message": "Unknown job"}
            )
        elapsed_seconds = _job_elapsed_seconds(job)
        return JSONResponse(
            {
                "status": job.get("status", "running"),
                "progress": job.get("progress", 0.0),
                "message": job.get("message", ""),
                "elapsed_seconds": elapsed_seconds,
                "elapsed_text": _format_duration(elapsed_seconds),
            }
        )

    @rt("/result/{session_id}")
    def result(req, session_id: str, ui_lang: str = "zh"):
        auth = _authorized(req, user_list, auth_message)
        if auth:
            return auth
        job = translation_jobs.get(session_id)
        ui_lang = _ui_lang(job.get("ui_lang") if job else ui_lang)
        autohide = bool(job.get("autohide")) if job else False
        if not job:
            return _page(
                Div(A(_t(ui_lang, "back"), href=f"/?ui_lang={ui_lang}", cls="button secondary"), cls="actions"),
                _result_panel(error="Unknown job", ui_lang=ui_lang),
                ui_lang=ui_lang,
            )
        if job.get("status") == "done":
            return _page(
                Div(
                    A(
                        _t(ui_lang, "start_another"),
                        href=f"/?ui_lang={ui_lang}&settings_from={quote(session_id)}",
                        cls="button secondary",
                    ),
                    cls="actions",
                ),
                _result_panel(
                    job.get("mono"),
                    job.get("dual"),
                    autohide=autohide,
                    ui_lang=ui_lang,
                    elapsed_seconds=_job_elapsed_seconds(job),
                ),
                autohide=autohide,
                ui_lang=ui_lang,
            )
        if job.get("status") == "error":
            return _page(
                Div(A(_t(ui_lang, "back"), href=f"/?ui_lang={ui_lang}", cls="button secondary"), cls="actions"),
                _result_panel(error=job.get("error", "Unknown error"), ui_lang=ui_lang),
                ui_lang=ui_lang,
            )
        return _progress_page(session_id, ui_lang=ui_lang, autohide=autohide)

    @rt("/file")
    def file(req, name: str):
        auth = _authorized(req, user_list, auth_message)
        if auth:
            return auth
        path = _output_file_path(name)
        if path is None:
            return Response("Not found", status_code=404)
        return FileResponse(path)

    @rt("/download")
    def download(req, name: str, variant: str = "mono"):
        auth = _authorized(req, user_list, auth_message)
        if auth:
            return auth
        if variant not in {"mono", "dual"}:
            return Response("Not found", status_code=404)
        path = _output_file_path(name)
        if path is None:
            return Response("Not found", status_code=404)
        return FileResponse(
            path,
            media_type="application/pdf",
            filename=_translated_download_name(path.name, variant),
        )

    @rt("/pdf-viewer")
    def pdf_viewer(req, name: str, view: str = "facing"):
        auth = _authorized(req, user_list, auth_message)
        if auth:
            return auth
        path = _output_file_path(name)
        if path is None:
            return Response("Not found", status_code=404)
        name = path.name

        pdf_url = f"/file?name={quote(name)}"
        facing = view == "facing"
        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{name}</title>
  <style>
    html, body {{
      margin: 0;
      min-height: 100%;
      background: #eef2f7;
      color: #243042;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    #status {{
      position: sticky;
      top: 0;
      z-index: 2;
      padding: .45rem .75rem;
      background: #f8fafc;
      border-bottom: 1px solid #d8dee8;
      font-size: .9rem;
    }}
    #viewer {{
      display: grid;
      gap: 1rem;
      padding: 1rem;
      box-sizing: border-box;
      width: 100%;
    }}
    .spread {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 1rem;
      align-items: start;
      justify-items: center;
      width: 100%;
    }}
    .spread.single {{
      grid-template-columns: minmax(0, 1fr);
    }}
    canvas {{
      max-width: 100%;
      height: auto;
      background: white;
      box-shadow: 0 1px 5px rgba(15, 23, 42, .18);
    }}
    @media (max-width: 900px) {{
      #viewer {{ padding: .5rem; gap: .5rem; }}
      .spread {{ grid-template-columns: minmax(0, 1fr); gap: .5rem; }}
    }}
  </style>
</head>
<body>
  <div id="status">Loading PDF...</div>
  <main id="viewer" data-facing="{str(facing).lower()}"></main>
  <script type="module">
    import * as pdfjsLib from "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.mjs";

    pdfjsLib.GlobalWorkerOptions.workerSrc =
      "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.worker.mjs";

    const pdfUrl = {json.dumps(pdf_url)};
    const viewer = document.getElementById("viewer");
    const status = document.getElementById("status");
    const facing = viewer.dataset.facing === "true";
    const pageGap = 16;

    async function renderPage(pdf, pageNumber, container, scale) {{
      const page = await pdf.getPage(pageNumber);
      const viewport = page.getViewport({{ scale }});
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.floor(viewport.width * ratio);
      canvas.height = Math.floor(viewport.height * ratio);
      canvas.style.width = `${{viewport.width}}px`;
      canvas.style.height = `${{viewport.height}}px`;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      container.appendChild(canvas);
      await page.render({{ canvasContext: context, viewport }}).promise;
    }}

    async function renderDocument() {{
      const pdf = await pdfjsLib.getDocument(pdfUrl).promise;
      status.textContent = `${{pdf.numPages}} pages`;
      const firstPage = await pdf.getPage(1);
      const baseViewport = firstPage.getViewport({{ scale: 1 }});
      const pagesPerRow = facing && window.innerWidth > 900 ? 2 : 1;
      const availableWidth = viewer.clientWidth - pageGap * (pagesPerRow - 1);
      const scale = Math.max(.25, Math.min(2.5, availableWidth / pagesPerRow / baseViewport.width));

      for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += pagesPerRow) {{
        const spread = document.createElement("section");
        spread.className = pagesPerRow === 2 && pageNumber < pdf.numPages ? "spread" : "spread single";
        viewer.appendChild(spread);
        await renderPage(pdf, pageNumber, spread, scale);
        if (pagesPerRow === 2 && pageNumber + 1 <= pdf.numPages) {{
          await renderPage(pdf, pageNumber + 1, spread, scale);
        }}
      }}
      status.remove();
    }}

    renderDocument().catch((error) => {{
      console.error(error);
      status.textContent = "Unable to load PDF";
    }});
  </script>
</body>
</html>"""
        return Response(html, media_type="text/html")

    return app


def setup_gui(
    share: bool = False,
    auth_file: list = ["", ""],
    server_port=7860,
    backend: str = "auto",
    onnx: str | None = None,
) -> None:
    global GUI_BACKEND, GUI_ONNX
    GUI_BACKEND = backend
    GUI_ONNX = onnx

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
            config = uvicorn.Config(app, host=addr, port=server_port)
            server = uvicorn.Server(config)
            server.run()
            print("FastHTML GUI stopped.")
            return
        except KeyboardInterrupt:
            print("\nShutting down FastHTML GUI...")
            return
        except Exception:
            print(
                f"Error launching GUI using {addr}.\n"
                "This may be caused by global mode of proxy software."
            )
        finally:
            shutdown_translation_jobs()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    setup_gui()
