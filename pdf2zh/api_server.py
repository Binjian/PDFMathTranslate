"""FastAPI translation backend — standalone service wrapping the pdf2zh engine.

Start with:
    uvicorn pdf2zh.api_server:app --host 0.0.0.0 --port 7861

Or via python:
    python -m pdf2zh.api_server

Environment variables:
    PDF2ZH_API_OUTPUT    Output directory (default: pdf2zh_api_files)
    PDF2ZH_API_JOB_LOG   Job log Markdown table (default: <output>/job_log.md)
    PDF2ZH_API_HOST     Bind host for run_api_server() (default: 0.0.0.0)
    PDF2ZH_API_PORT     Port for run_api_server() (default: 7861)
"""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import queue
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional

import requests as _requests
import tqdm
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

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

OUTPUT_DIR = Path(ConfigManager.get("PDF2ZH_API_OUTPUT", "pdf2zh_api_files"))
JOB_LOG = Path(
    ConfigManager.get("PDF2ZH_API_JOB_LOG", str(OUTPUT_DIR / "job_log.md"))
)

# ── Translator / language / page lookup tables ────────────────────────────────

SERVICE_MAP: dict[str, type[BaseTranslator]] = {
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

LANG_MAP: dict[str, str] = {
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

PAGE_MAP: dict[str, Optional[list[int]]] = {
    "All": None,
    "First": [0],
    "First 5 pages": list(range(0, 5)),
}

# ── In-memory job store ───────────────────────────────────────────────────────

_jobs: dict[str, dict] = {}
_job_log_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _job_log_cell(value) -> str:
    if value is None:
        text = ""
    elif isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    return text.replace("\r", " ").replace("\n", " ").replace("|", "\\|")


def _job_file_names(job: dict) -> list[str]:
    files: list[str] = []
    for key in ("source", "mono", "dual"):
        path_str = job.get(key)
        if path_str:
            name = Path(path_str).name
            if name not in files:
                files.append(name)
    for name in job.get("removed_files") or []:
        if name not in files:
            files.append(name)
    return files


def _job_elapsed_seconds(job: dict) -> float | None:
    started_at = job.get("started_at")
    if started_at is None:
        return None
    finished_at = job.get("finished_at") or time.time()
    return max(0.0, finished_at - started_at)


def _format_elapsed_time(seconds: float | None) -> str:
    if seconds is None:
        return ""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _format_llm_duration_ns(nanoseconds: int | float | None) -> str:
    try:
        ns = int(nanoseconds or 0)
    except (TypeError, ValueError):
        return ""
    if ns <= 0:
        return ""
    seconds = ns / 1_000_000_000
    if seconds < 1:
        return f"{max(1, round(ns / 1_000_000))}ms"
    return _format_elapsed_time(seconds)


def _format_llm_usage(usage: dict | None) -> str:
    if not usage:
        return ""
    parts: list[str] = []
    requests = int(usage.get("requests") or 0)
    prompt_count = int(usage.get("prompt_eval_count") or 0)
    eval_count = int(usage.get("eval_count") or 0)
    prompt_duration = _format_llm_duration_ns(usage.get("prompt_eval_duration"))
    eval_duration = _format_llm_duration_ns(usage.get("eval_duration"))
    total_duration = _format_llm_duration_ns(usage.get("total_duration"))
    load_duration = _format_llm_duration_ns(usage.get("load_duration"))
    if requests:
        parts.append(f"requests: {requests}")
    if prompt_count or prompt_duration:
        detail = f"{prompt_count:,} tok" if prompt_count else ""
        if prompt_duration:
            detail = f"{detail} in {prompt_duration}" if detail else prompt_duration
        parts.append(f"prompt: {detail}")
    if eval_count or eval_duration:
        detail = f"{eval_count:,} tok" if eval_count else ""
        if eval_duration:
            detail = f"{detail} in {eval_duration}" if detail else eval_duration
        parts.append(f"completion: {detail}")
    if total_duration:
        parts.append(f"total: {total_duration}")
    if load_duration:
        parts.append(f"load: {load_duration}")
    return "; ".join(parts)


def _append_job_log(job_id: str, job: dict, response: dict) -> None:
    """Append a human-readable Markdown table row for a job event."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    elapsed = _job_elapsed_seconds(job)
    row = [
        timestamp,
        job_id,
        job.get("service", ""),
        ", ".join(_job_file_names(job)),
        _format_elapsed_time(elapsed),
        _format_llm_usage(job.get("llm_usage")),
        response,
    ]
    try:
        with _job_log_lock:
            JOB_LOG.parent.mkdir(parents=True, exist_ok=True)
            header = "| timestamp | job_id | service | files | elapsed_time | llm_usage | response |\n"
            separator = "|---|---|---|---|---:|---|---|\n"
            old_tables = (
                (
                    "| timestamp | job_id | service | files | response |\n",
                    "|---|---|---|---|---|\n",
                ),
                (
                    "| timestamp | job_id | service | files | elapsed_seconds | response |\n",
                    "|---|---|---|---|---:|---|\n",
                ),
                (
                    "| timestamp | job_id | service | files | elapsed_time | response |\n",
                    "|---|---|---|---|---:|---|\n",
                ),
            )
            if JOB_LOG.exists() and JOB_LOG.stat().st_size > 0:
                content = JOB_LOG.read_text(encoding="utf-8")
                matched_old_table = next(
                    (
                        old_header + old_separator
                        for old_header, old_separator in old_tables
                        if content.startswith(old_header + old_separator)
                    ),
                    None,
                )
                if matched_old_table:
                    migrated = [header.rstrip("\n"), separator.rstrip("\n")]
                    for line in content.splitlines()[2:]:
                        parts = line.split(" | ")
                        if len(parts) == 5:
                            parts.insert(4, "")
                            parts.insert(5, "")
                            line = " | ".join(parts)
                        elif len(parts) == 6:
                            parts.insert(5, "")
                            line = " | ".join(parts)
                        migrated.append(line)
                    JOB_LOG.write_text("\n".join(migrated) + "\n", encoding="utf-8")
            if not JOB_LOG.exists() or JOB_LOG.stat().st_size == 0:
                JOB_LOG.write_text(header + separator, encoding="utf-8")
            with JOB_LOG.open("a", encoding="utf-8") as handle:
                handle.write("| " + " | ".join(_job_log_cell(value) for value in row) + " |\n")
    except Exception:
        logger.exception("Unable to update API job log at %s", JOB_LOG)


def _selected_pages(page_range: str, page_input: str) -> Optional[list[int]]:
    if page_range in PAGE_MAP:
        return PAGE_MAP[page_range]
    selected: list[int] = []
    for part in page_input.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            selected.extend(range(int(start) - 1, int(end)))
        else:
            selected.append(int(part) - 1)
    return selected or None


def _normalize_ollama_host(host: str) -> str:
    host = host.strip()
    if host and "://" not in host:
        host = f"http://{host}"
    return host.rstrip("/")


def _resolve_translator_envs(
    service: str, submitted: list[str]
) -> dict[str, str | None]:
    """Resolve request-scoped translator settings for a submitted API job."""
    translator = SERVICE_MAP[service]
    envs: dict[str, str | None] = {}
    for i, (key, default) in enumerate(translator.envs.items()):
        value = submitted[i] if i < len(submitted) else ""
        envs[key] = value if value != "" else default
    if service == "Ollama":
        envs["OLLAMA_HOST"] = _normalize_ollama_host(str(envs["OLLAMA_HOST"]))
    for key, value in envs.items():
        if key.upper().endswith("API_KEY") and value == "***":
            envs[key] = ConfigManager.get_env_by_translatername(translator, key, None)
    return envs


def _ollama_model_names(host: str, timeout: float = 2) -> list[str]:
    """Return installed Ollama model names as seen from the API backend."""
    host = _normalize_ollama_host(host)
    if not host:
        raise ValueError("Ollama host must not be empty")
    with _requests.Session() as session:
        session.trust_env = False
        response = session.get(f"{host}/api/tags", timeout=timeout)
    response.raise_for_status()
    return [
        model["name"]
        for model in response.json().get("models", [])
        if isinstance(model, dict) and model.get("name")
    ]


def _validate_ollama_envs(envs: dict[str, str | None]) -> None:
    """Fail fast when the submitted Ollama host/model cannot run the job."""
    host = str(envs.get("OLLAMA_HOST") or "")
    model = str(envs.get("OLLAMA_MODEL") or "")
    if not model:
        raise HTTPException(400, "Ollama model must not be empty")
    try:
        models = _ollama_model_names(host)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            502, f"Unable to query Ollama models from {_normalize_ollama_host(host)}"
        ) from exc
    if model not in models:
        available = ", ".join(models) if models else "none"
        raise HTTPException(
            400,
            f"Ollama model '{model}' is not installed on {host}. "
            f"Available models: {available}",
        )


def _cleanup_job_artifacts(job_id: str, job: dict) -> dict:
    """Remove PDF files for a completed job."""
    if job.get("status") == "running":
        raise HTTPException(409, "Cannot remove artifacts while job is running")

    job_dir = (OUTPUT_DIR / job_id).resolve()
    output_root = OUTPUT_DIR.resolve()
    if output_root not in (job_dir, *job_dir.parents):
        raise HTTPException(400, "Invalid job output directory")

    candidates: set[Path] = set()
    for variant in ("mono", "dual"):
        path_str = job.get(variant)
        if path_str:
            candidates.add(Path(path_str))
    if job_dir.exists():
        candidates.update(job_dir.glob("*.pdf"))

    removed: list[str] = []
    for path in sorted(candidates):
        path = path.resolve()
        if output_root not in (path, *path.parents):
            continue
        if path.is_file():
            path.unlink()
            removed.append(path.name)

    response = {
        "job_id": job_id,
        "status": "artifacts_removed",
        "removed_files": removed,
    }
    job.update(
        {
            "mono": None,
            "dual": None,
            "message": "Artifacts removed",
            "removed_files": removed,
        }
    )
    _append_job_log(job_id, job, response)
    return response

# ── Translation subprocess ────────────────────────────────────────────────────

def _translate_process(params: dict, progress_queue: multiprocessing.Queue) -> None:
    """Spawned in a child process; sends progress / done / error events."""
    try:
        from pdf2zh.doclayout import ModelInstance, OnnxModel, set_backend
        from pdf2zh.kernel import KernelRegistry
        from pdf2zh.kernel.protocol import TranslateRequest

        set_backend(params.get("backend", "auto"))
        onnx = params.get("onnx")
        if onnx and ModelInstance.value is None:
            ModelInstance.value = OnnxModel(onnx)

        if params.get("service_name") == "ollama":
            OllamaTranslator.reset_usage()

        KernelRegistry.switch(params["mode_choice"])
        kernel = KernelRegistry.get()

        def _cb(t: tqdm.tqdm) -> None:
            total = max(getattr(t, "total", 0) or 0, 1)
            progress_queue.put(
                {
                    "type": "progress",
                    "progress": min(0.99, max(0.0, t.n / total)),
                    "message": getattr(t, "desc", "") or "Translating...",
                }
            )

        request = TranslateRequest(
            files=[params["file_path"]],
            output=params["output_dir"],
            pages=params["pages"],
            lang_in=params["lang_in"],
            lang_out=params["lang_out"],
            service=params["service_name"],
            thread=params["threads"],
            envs=params["envs"],
            prompt=params["prompt"],
            skip_subset_fonts=params["skip_subset_fonts"],
            ignore_cache=params["ignore_cache"],
            vfont=params["vfont"],
        )
        kernel.translate(request, callback=_cb)
        progress_queue.put(
            {"type": "progress", "progress": 0.99, "message": "Collecting output files..."}
        )

        stem = Path(params["file_path"]).stem
        out = Path(params["output_dir"])
        mono = str(out / f"{stem}-mono.pdf")
        dual = str(out / f"{stem}-dual.pdf")

        if not Path(mono).exists() or not Path(dual).exists():
            error_event = {"type": "error", "message": "Translation produced no output files"}
            if params.get("service_name") == "ollama":
                error_event["llm_usage"] = OllamaTranslator.usage_snapshot()
            progress_queue.put(error_event)
            return

        done_event = {"type": "done", "mono": mono, "dual": dual}
        if params.get("service_name") == "ollama":
            done_event["llm_usage"] = OllamaTranslator.usage_snapshot()
        progress_queue.put(done_event)
    except BaseException as exc:
        message = str(exc) or type(exc).__name__
        if params.get("service_name") == "ollama":
            envs = params.get("envs") or {}
            message = (
                "Ollama translation failed "
                f"(host={envs.get('OLLAMA_HOST')}, "
                f"model={envs.get('OLLAMA_MODEL')}): {message}"
            )
        error_event = {"type": "error", "message": message}
        if params.get("service_name") == "ollama":
            error_event["llm_usage"] = OllamaTranslator.usage_snapshot()
        progress_queue.put(error_event)


def _monitor_job(
    job_id: str,
    process: multiprocessing.Process,
    pq: multiprocessing.Queue,
) -> None:
    """Daemon thread: drains the progress queue and keeps _jobs up to date."""
    while True:
        try:
            event = pq.get(timeout=0.5)
        except queue.Empty:
            event = None

        if event:
            etype = event.get("type")
            if etype == "progress":
                _jobs[job_id].update(
                    {
                        "progress": event.get("progress", 0.0),
                        "message": event.get("message", ""),
                    }
                )
            elif etype == "done":
                _jobs[job_id].update(
                    {
                        "status": "done",
                        "progress": 1.0,
                        "message": "Translation complete",
                        "mono": event["mono"],
                        "dual": event["dual"],
                        "llm_usage": event.get("llm_usage"),
                        "finished_at": time.time(),
                    }
                )
                _append_job_log(
                    job_id,
                    _jobs[job_id],
                    {
                        "status": "done",
                        "message": "Translation complete",
                        "mono": Path(event["mono"]).name,
                        "dual": Path(event["dual"]).name,
                    },
                )
                logger.info("Translation job %s completed", job_id)
                break
            elif etype == "error":
                msg = event.get("message", "Unknown error")
                _jobs[job_id].update(
                    {
                        "status": "error",
                        "progress": 1.0,
                        "message": msg,
                        "error": msg,
                        "llm_usage": event.get("llm_usage"),
                        "finished_at": time.time(),
                    }
                )
                _append_job_log(
                    job_id,
                    _jobs[job_id],
                    {"status": "error", "message": msg},
                )
                logger.error("Translation job %s failed: %s", job_id, msg)
                break

        if not process.is_alive():
            if _jobs[job_id].get("status") in {"done", "error"}:
                break
            code = process.exitcode
            msg = (
                "Worker finished without output."
                if code == 0
                else f"Worker crashed (exit code {code})."
            )
            _jobs[job_id].update(
                {
                    "status": "error",
                    "progress": 1.0,
                    "message": msg,
                    "error": msg,
                    "finished_at": time.time(),
                }
            )
            _append_job_log(
                job_id,
                _jobs[job_id],
                {"status": "error", "message": msg, "exit_code": code},
            )
            logger.error("Translation job %s failed: %s", job_id, msg)
            break

    process.join(timeout=1)
    _jobs[job_id].pop("process", None)
    try:
        pq.close()
        pq.join_thread()
    except Exception:
        pass


# ── FastAPI application ───────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(application: FastAPI):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="pdf2zh Translation API", version="1.0.0", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}


@app.get("/v1/ollama/models")
def ollama_models(host: str = OllamaTranslator.envs["OLLAMA_HOST"]) -> dict:
    """List models from Ollama as reachable by the translation backend."""
    host = _normalize_ollama_host(host)
    try:
        return {"models": _ollama_model_names(host)}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            502, f"Unable to query Ollama models from {host}"
        ) from exc


@app.post("/v1/translate", status_code=202)
async def create_translate_job(
    file: Optional[UploadFile] = File(None),
    link: Annotated[str, Form()] = "",
    service: Annotated[str, Form()] = "Google",
    lang_from: Annotated[str, Form()] = "English",
    lang_to: Annotated[str, Form()] = "Simplified Chinese",
    page_range: Annotated[str, Form()] = "All",
    page_input: Annotated[str, Form()] = "",
    prompt: Annotated[str, Form()] = "",
    threads: Annotated[int, Form()] = 4,
    skip_subset_fonts: Annotated[bool, Form()] = False,
    ignore_cache: Annotated[bool, Form()] = False,
    vfont: Annotated[str, Form()] = "",
    mode_choice: Annotated[str, Form()] = "fast",
    env_0: Annotated[str, Form()] = "",
    env_1: Annotated[str, Form()] = "",
    env_2: Annotated[str, Form()] = "",
    env_3: Annotated[str, Form()] = "",
) -> dict:
    """Submit a translation job.

    Returns 202 Accepted with ``{"job_id": "<uuid>"}`` immediately.
    Poll ``GET /v1/translate/{job_id}`` for status.
    """
    if service not in SERVICE_MAP:
        raise HTTPException(400, f"Unknown service '{service}'. "
                            f"Valid values: {sorted(SERVICE_MAP)}")

    translator = SERVICE_MAP[service]
    envs = _resolve_translator_envs(service, [env_0, env_1, env_2, env_3])
    if service == "Ollama":
        _validate_ollama_envs(envs)
        logger.info(
            "Submitting Ollama translation using OLLAMA_HOST=%s", envs["OLLAMA_HOST"]
        )

    job_id = str(uuid.uuid4())
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # ── Persist the source PDF ─────────────────────────────────────────────
    if file and file.filename:
        safe_name = os.path.basename(file.filename)
        file_path = job_dir / safe_name
        file_path.write_bytes(await file.read())
    elif link:
        try:
            resp = _requests.get(link, allow_redirects=True, timeout=60)
            resp.raise_for_status()
        except Exception as exc:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(400, f"Could not download file: {exc}") from exc
        raw_name = os.path.basename(link.split("?")[0].rstrip("/")) or "document"
        if not raw_name.lower().endswith(".pdf"):
            raw_name += ".pdf"
        file_path = job_dir / raw_name
        file_path.write_bytes(resp.content)
    else:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, "Provide a file upload or a link")

    lang_in = LANG_MAP.get(lang_from, lang_from)
    lang_out = LANG_MAP.get(lang_to, lang_to)
    pages = _selected_pages(page_range, page_input)

    _jobs[job_id] = {
        "status": "running",
        "progress": 0.0,
        "message": "Starting translation...",
        "service": service,
        "source": str(file_path),
        "mono": None,
        "dual": None,
        "error": None,
        "llm_usage": None,
        "started_at": time.time(),
        "finished_at": None,
    }

    params = {
        "file_path": str(file_path),
        "output_dir": str(job_dir),
        "service_name": translator.name,
        "lang_in": lang_in,
        "lang_out": lang_out,
        "pages": pages,
        "threads": threads,
        "envs": envs,
        "prompt": prompt or None,
        "skip_subset_fonts": skip_subset_fonts,
        "ignore_cache": ignore_cache,
        "vfont": vfont,
        "mode_choice": mode_choice,
        "backend": "auto",
        "onnx": None,
    }

    ctx = multiprocessing.get_context("spawn")
    pq = ctx.Queue()
    process = ctx.Process(target=_translate_process, args=(params, pq))
    _jobs[job_id]["process"] = process
    process.start()

    threading.Thread(
        target=_monitor_job, args=(job_id, process, pq), daemon=True
    ).start()

    return {"job_id": job_id}


@app.get("/v1/translate/{job_id}")
def get_job_status(job_id: str) -> dict:
    """Return current status and progress of a translation job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    elapsed = time.time() - job["started_at"]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0.0),
        "message": job.get("message", ""),
        "error": job.get("error"),
        "elapsed_seconds": elapsed,
    }


@app.delete("/v1/translate/{job_id}", status_code=200)
def cancel_job(job_id: str) -> dict:
    """Terminate a running translation job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    process = job.get("process")
    if process and process.is_alive():
        process.terminate()
    response = {"status": "cancelled"}
    job.update(
        {
            "status": "error",
            "message": "Cancelled",
            "error": "Cancelled",
            "finished_at": time.time(),
        }
    )
    _append_job_log(job_id, job, response)
    return response


@app.delete("/v1/translate/{job_id}/artifacts", status_code=200)
@app.delete("/v1/translate/{job_id}/artefacts", status_code=200)
def remove_job_artifacts(job_id: str) -> dict:
    """Remove PDF files generated or stored for a job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return _cleanup_job_artifacts(job_id, job)


@app.get("/v1/translate/{job_id}/{variant}")
def download_result(job_id: str, variant: str) -> FileResponse:
    """Download the translated PDF.  ``variant`` is ``mono`` or ``dual``."""
    if variant not in {"mono", "dual"}:
        raise HTTPException(400, "variant must be 'mono' or 'dual'")
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        raise HTTPException(409, f"Job not finished (status: {job['status']})")
    path_str = job.get(variant)
    if not path_str or not Path(path_str).exists():
        raise HTTPException(404, f"{variant} file not found")
    return FileResponse(
        path_str,
        media_type="application/pdf",
        filename=Path(path_str).name,
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

def run_api_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> None:
    """Start the translation API server.

    ``host`` and ``port`` can be overridden by the caller; otherwise they are
    read from ``PDF2ZH_API_HOST`` / ``PDF2ZH_API_PORT`` env vars at call time,
    falling back to ``0.0.0.0:7861``.  Using ``0.0.0.0`` is required when the
    server must accept connections from other hosts.
    """
    import uvicorn

    _host = host or (ConfigManager.get("PDF2ZH_API_HOST") or "0.0.0.0")
    _port = port or int(ConfigManager.get("PDF2ZH_API_PORT") or "7861")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Starting pdf2zh API server on http://%s:%s", _host, _port)
    uvicorn.run(app, host=_host, port=_port)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    run_api_server()
