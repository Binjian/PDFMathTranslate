"""FastAPI translation backend — standalone service wrapping the pdf2zh engine.

Start with:
    uvicorn pdf2zh.api_server:app --host 0.0.0.0 --port 7861

Or via python:
    python -m pdf2zh.api_server

Environment variables:
    PDF2ZH_API_OUTPUT              Output directory (default: pdf2zh_api_files)
    PDF2ZH_API_JOB_LOG             Job log Markdown table (default: <output>/job_log.md)
    PDF2ZH_API_FRONTEND_METRICS    Frontend metrics Markdown table (default: <output>/frontend_metrics.md)
    PDF2ZH_API_HOST     Bind host for run_api_server() (default: 0.0.0.0)
    PDF2ZH_API_PORT     Port for run_api_server() (default: 7861)
    PDF2ZH_API_MONGODB_URI         MongoDB URI for durable job-artefact storage
                                   (default: mongodb://localhost:27017)
    PDF2ZH_API_MONGODB_DB          MongoDB database name (default: pdf2zh)
    PDF2ZH_API_MONGODB_COLLECTION  MongoDB collection name (default: job_artifacts)
"""
from __future__ import annotations

import io
import json
import logging
import multiprocessing
import os
import queue
import shutil
import threading
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional

import requests as _requests
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from pdf2zh.config import ConfigManager
from pdf2zh.kernel import KernelRegistry
from pdf2zh.mongo_store import JobArtifactStore
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
FRONTEND_METRICS = Path(
    ConfigManager.get("PDF2ZH_API_FRONTEND_METRICS") or str(OUTPUT_DIR / "frontend_metrics.md")
)

# MongoDB persistence of job artefacts. Defaults to mongodb://localhost:27017;
# override with PDF2ZH_API_MONGODB_URI. See pdf2zh.mongo_store.
_artifact_store = JobArtifactStore()

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
_MODE_SETUP_HINTS = {
    "precise": (
        "Kernel 'precise' is not available on the API server. "
        "Initialize the v2 submodule and run pdf2zh-setup-precise on the API host."
    )
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    real_ip = request.headers.get("x-real-ip", "").strip()
    if real_ip:
        return real_ip
    return request.client.host if request.client else ""


def _validate_mode_choice(mode_choice: str) -> str:
    mode = (mode_choice or "fast").strip().lower()
    try:
        kernel = KernelRegistry.get(mode)
    except KeyError as exc:
        raise HTTPException(
            400,
            f"Unknown mode_choice '{mode_choice}'. Valid values: fast, precise",
        ) from exc

    if not kernel.is_available():
        raise HTTPException(
            400,
            _MODE_SETUP_HINTS.get(
                mode,
                f"Kernel '{mode}' is not available on the API server.",
            ),
        )
    return mode


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


def _llm_usage_translator(service_name: str):
    return {
        "ollama": OllamaTranslator,
        "openailiked": OpenAIlikedTranslator,
        "qwen-mt": QwenMtTranslator,
    }.get(service_name)


def _reset_llm_usage(service_name: str) -> None:
    translator = _llm_usage_translator(service_name)
    if translator is not None:
        translator.reset_usage()


def _llm_usage_snapshot(service_name: str) -> dict | None:
    translator = _llm_usage_translator(service_name)
    if translator is None:
        return None
    return translator.usage_snapshot()


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


def _llm_generated_tokens(usage: dict | None) -> int | None:
    if not usage:
        return None
    try:
        return int(usage.get("eval_count") or usage.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        return None


def _format_llm_usage(usage: dict | None) -> str:
    if not usage:
        return ""
    parts: list[str] = []
    requests = int(usage.get("requests") or 0)
    prompt_count = int(
        usage.get("prompt_eval_count") or usage.get("prompt_tokens") or 0
    )
    eval_count = int(
        usage.get("eval_count") or usage.get("completion_tokens") or 0
    )
    total_count = int(usage.get("total_tokens") or 0)
    prompt_duration = _format_llm_duration_ns(usage.get("prompt_eval_duration"))
    eval_duration = _format_llm_duration_ns(usage.get("eval_duration"))
    total_duration = _format_llm_duration_ns(usage.get("total_duration"))
    request_duration = _format_elapsed_time(usage.get("request_duration"))
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
    if total_count:
        parts.append(f"tokens: {total_count:,}")
    if total_duration:
        parts.append(f"total: {total_duration}")
    elif request_duration:
        parts.append(f"api time: {request_duration}")
    if load_duration:
        parts.append(f"load: {load_duration}")
    if not parts and "requests" in usage:
        return "requests: 0"
    return "; ".join(parts)


def _format_llm_duration(usage: dict | None) -> str:
    if not usage:
        return ""
    total_duration = _format_llm_duration_ns(usage.get("total_duration"))
    if total_duration:
        return total_duration
    try:
        request_duration = float(usage.get("request_duration") or 0)
    except (TypeError, ValueError):
        return ""
    return _format_elapsed_time(request_duration) if request_duration > 0 else ""


def _format_llm_generated_tokens(usage: dict | None) -> str:
    generated_tokens = _llm_generated_tokens(usage)
    if generated_tokens is None:
        return ""
    return f"{generated_tokens:,}"


def _llm_requests(usage: dict | None) -> int:
    if not usage:
        return 0
    try:
        return int(usage.get("requests") or 0)
    except (TypeError, ValueError):
        return 0


def _llm_prompt_tokens(usage: dict | None) -> int:
    if not usage:
        return 0
    try:
        return int(usage.get("prompt_eval_count") or usage.get("prompt_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def _llm_total_tokens(usage: dict | None) -> int:
    if not usage:
        return 0
    try:
        total = int(usage.get("total_tokens") or 0)
        if total:
            return total
        prompt = int(usage.get("prompt_eval_count") or usage.get("prompt_tokens") or 0)
        completion = int(usage.get("eval_count") or usage.get("completion_tokens") or 0)
        return prompt + completion
    except (TypeError, ValueError):
        return 0


def _append_job_log(job_id: str, job: dict, response: dict) -> None:
    """Append a human-readable Markdown table row for a job event."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    elapsed = _job_elapsed_seconds(job)
    usage = job.get("llm_usage")

    requests_n = _llm_requests(usage)
    prompt_n = _llm_prompt_tokens(usage)
    completion_n = _llm_generated_tokens(usage) or 0
    total_n = _llm_total_tokens(usage)
    api_time_s = _format_llm_duration(usage) or "0"

    row = [
        timestamp,
        job_id,
        f"**{response.get('status', '')}**",
        job.get("client_ip", ""),
        job.get("service", ""),
        ", ".join(_job_file_names(job)),
        _format_elapsed_time(elapsed),
        f"**{requests_n:,}**",
        f"{prompt_n:,}" if prompt_n else "0",
        f"{completion_n:,}" if completion_n else "0",
        f"**{total_n:,}**" if total_n else "**0**",
        f"**{api_time_s}**",
        response,
    ]
    try:
        with _job_log_lock:
            JOB_LOG.parent.mkdir(parents=True, exist_ok=True)
            header = "| timestamp | job_id | status | client_ip | service | files | elapsed_time | requests | prompt | completion | total_tokens | api_time | response |\n"
            separator = "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|\n"
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
                (
                    "| timestamp | job_id | service | files | elapsed_time | llm_usage | response |\n",
                    "|---|---|---|---|---:|---|---|\n",
                ),
                (
                    "| timestamp | job_id | client_ip | service | files | elapsed_time | llm_usage | response |\n",
                    "|---|---|---|---|---|---:|---|---|\n",
                ),
                (
                    "| timestamp | job_id | client_ip | service | files | elapsed_time | llm_duration | generated_tokens | response |\n",
                    "|---|---|---|---|---|---:|---:|---:|---|\n",
                ),
                (
                    "| timestamp | job_id | client_ip | service | files | elapsed_time | llm_usage | llm_duration | generated_tokens | response |\n",
                    "|---|---|---|---|---|---:|---|---:|---:|---|\n",
                ),
                (
                    "| timestamp | job_id | client_ip | service | files | elapsed_time | requests | prompt | completion | total_tokens | api_time | response |\n",
                    "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|\n",
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
                        n = len(parts)
                        if n == 5:
                            # ts, jid, svc, files, resp
                            parts.insert(2, "")   # status
                            parts.insert(3, "")   # ip
                            parts.insert(6, "")   # elapsed
                            parts[7:7] = ["**0**", "0", "0", "**0**", "**0**"]
                        elif n == 6:
                            # ts, jid, svc, files, elapsed, resp
                            parts.insert(2, "")   # status
                            parts.insert(3, "")   # ip
                            parts[7:7] = ["**0**", "0", "0", "**0**", "**0**"]
                        elif n == 7:
                            # ts, jid, svc, files, elapsed, llm_usage, resp
                            parts.insert(2, "")   # status
                            parts.insert(3, "")   # ip
                            parts[7:8] = ["**0**", "0", "0", "**0**", "**0**"]
                        elif n == 8:
                            # ts, jid, ip, svc, files, elapsed, llm_usage, resp
                            parts.insert(2, "")   # status
                            parts[7:8] = ["**0**", "0", "0", "**0**", "**0**"]
                        elif n == 9:
                            # ts, jid, ip, svc, files, elapsed, llm_dur, gen_tok, resp
                            parts.insert(2, "")   # status
                            old_dur = parts[7].strip()
                            old_tok = parts[8].strip()
                            parts[7:9] = ["**0**", "0", old_tok, "**0**", f"**{old_dur or '0'}**"]
                        elif n == 10:
                            # ts, jid, ip, svc, files, elapsed, llm_usage, llm_dur, gen_tok, resp
                            parts.insert(2, "")   # status
                            old_dur = parts[8].strip()
                            old_tok = parts[9].strip()
                            parts[7:10] = ["**0**", "0", old_tok, "**0**", f"**{old_dur or '0'}**"]
                        elif n == 12:
                            # ts, jid, ip, svc, files, elapsed, requests, prompt, completion, total, api, resp
                            parts.insert(2, "")   # status
                        line = " | ".join(parts)
                        migrated.append(line)
                    JOB_LOG.write_text("\n".join(migrated) + "\n", encoding="utf-8")
            if not JOB_LOG.exists() or JOB_LOG.stat().st_size == 0:
                JOB_LOG.write_text(header + separator, encoding="utf-8")
            with JOB_LOG.open("a", encoding="utf-8") as handle:
                handle.write("| " + " | ".join(_job_log_cell(value) for value in row) + " |\n")
            fm_header = "| timestamp | api_time | completion | response |\n"
            fm_separator = "|---|---:|---:|---|\n"
            fm_row = [timestamp, api_time_s, f"{completion_n:,}" if completion_n else "0", response]
            FRONTEND_METRICS.parent.mkdir(parents=True, exist_ok=True)
            if FRONTEND_METRICS.exists() and FRONTEND_METRICS.stat().st_size > 0:
                fm_text = FRONTEND_METRICS.read_text(encoding="utf-8")
                if fm_text.startswith("| timestamp | llm_duration |"):
                    FRONTEND_METRICS.write_text(
                        fm_text.replace(
                            "| timestamp | llm_duration | generated_tokens | response |",
                            "| timestamp | api_time | completion | response |",
                            1,
                        ),
                        encoding="utf-8",
                    )
            if not FRONTEND_METRICS.exists() or FRONTEND_METRICS.stat().st_size == 0:
                FRONTEND_METRICS.write_text(fm_header + fm_separator, encoding="utf-8")
            with FRONTEND_METRICS.open("a", encoding="utf-8") as handle:
                handle.write("| " + " | ".join(_job_log_cell(v) for v in fm_row) + " |\n")
    except Exception:
        logger.exception("Unable to update API job log at %s", JOB_LOG)

    # Mirror the event into MongoDB when configured (no-op otherwise). Kept in
    # its own try/except so file logging and Mongo persistence stay independent.
    try:
        document = {
            "status": response.get("status", ""),
            "client_ip": job.get("client_ip", ""),
            "service": job.get("service", ""),
            "files": _job_file_names(job),
            "source": job.get("source"),
            "mono": job.get("mono"),
            "dual": job.get("dual"),
            "removed_files": job.get("removed_files"),
            "elapsed_seconds": elapsed,
            "llm_usage": usage,
            "llm_requests": requests_n,
            "llm_prompt_tokens": prompt_n,
            "llm_completion_tokens": completion_n,
            "llm_total_tokens": total_n,
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
        }
        event = {
            "timestamp": timestamp,
            "status": response.get("status", ""),
            "response": response,
        }
        _artifact_store.record(job_id, document, event)
    except Exception:
        logger.exception("Unable to persist job %s artefacts to MongoDB", job_id)


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


# OpenAI-liked credentials are backend-only: whatever a client submits is
# discarded and the value is resolved from this server's environment (.env).
OPENAILIKED_BACKEND_ONLY_ENVS = {
    "OPENAILIKED_BASE_URL": "DASHSCOPE_API_URL",
    "OPENAILIKED_API_KEY": "DASHSCOPE_API_KEY",
}


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
    for key, alias in OPENAILIKED_BACKEND_ONLY_ENVS.items():
        if key in envs:
            envs[key] = os.environ.get(key) or os.environ.get(alias) or None
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


def _ingest_job_files(job_id: str, job: dict) -> dict[str, str]:
    """Upload a finished job's PDFs to GridFS (the source of truth).

    Reads source/mono/dual from the on-disk scratch folder and stores each blob
    keyed by ``job_id`` + ``variant``. Only when every present file was stored
    successfully is the on-disk job folder removed — a Mongo outage leaves the
    scratch files in place so nothing is lost.
    """
    stored: dict[str, str] = {}
    all_ok = True
    for variant in ("source", "mono", "dual"):
        path_str = job.get(variant)
        if not path_str:
            continue
        path = Path(path_str)
        if not path.is_file():
            all_ok = False
            continue
        try:
            data = path.read_bytes()
        except OSError:
            logger.exception("Unable to read %s for job %s", path, job_id)
            all_ok = False
            continue
        file_id = _artifact_store.put_file(
            data, path.name, job_id=job_id, variant=variant
        )
        if file_id is None:
            all_ok = False
        else:
            stored[variant] = path.name

    if all_ok and stored:
        job_dir = (OUTPUT_DIR / job_id).resolve()
        if OUTPUT_DIR.resolve() in job_dir.parents:
            shutil.rmtree(job_dir, ignore_errors=True)
    return stored


def _cleanup_job_artifacts(job_id: str, job: dict) -> dict:
    """Remove a job's PDF artefacts from MongoDB and any on-disk scratch."""
    if job.get("status") == "running":
        raise HTTPException(409, "Cannot remove artifacts while job is running")

    removed = _artifact_store.delete_files({"job_id": job_id})

    job_dir = (OUTPUT_DIR / job_id).resolve()
    output_root = OUTPUT_DIR.resolve()
    if output_root not in (job_dir, *job_dir.parents):
        raise HTTPException(400, "Invalid job output directory")
    if job_dir.exists():
        for path in sorted(job_dir.glob("*.pdf")):
            path = path.resolve()
            if output_root not in path.parents:
                continue
            if path.is_file():
                path.unlink()
                if path.name not in removed:
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

        _reset_llm_usage(params.get("service_name", ""))

        KernelRegistry.switch(params["mode_choice"])
        kernel = KernelRegistry.get()

        def _cb(t) -> None:
            if isinstance(t, dict):
                event_type = t.get("type", "")
                if event_type not in ("progress_start", "progress_update", "progress_end"):
                    return
                progress = float(t.get("overall_progress") or t.get("stage_progress") or 0.0)
                message = t.get("stage", "") or "Translating..."
            else:
                total = max(getattr(t, "total", 0) or 0, 1)
                progress = min(0.99, max(0.0, t.n / total))
                message = getattr(t, "desc", "") or "Translating..."
            progress_queue.put(
                {
                    "type": "progress",
                    "progress": min(0.99, max(0.0, progress)),
                    "message": message,
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
        results = kernel.translate(request, callback=_cb)
        progress_queue.put(
            {"type": "progress", "progress": 0.99, "message": "Collecting output files..."}
        )

        stem = Path(params["file_path"]).stem
        out = Path(params["output_dir"])

        # Use paths from the kernel result when available (precise kernel names files
        # differently from fast kernel, e.g. {stem}.zh.mono.pdf vs {stem}-mono.pdf).
        if results and results[0].mono_pdf and results[0].dual_pdf:
            mono = str(results[0].mono_pdf)
            dual = str(results[0].dual_pdf)
        else:
            mono = str(out / f"{stem}-mono.pdf")
            dual = str(out / f"{stem}-dual.pdf")

        if not Path(mono).exists() or not Path(dual).exists():
            error_event = {"type": "error", "message": "Translation produced no output files"}
            error_event["llm_usage"] = _llm_usage_snapshot(params.get("service_name", "")) or {"requests": 0}
            progress_queue.put(error_event)
            return

        done_event = {"type": "done", "mono": mono, "dual": dual}
        done_event["llm_usage"] = _llm_usage_snapshot(params.get("service_name", "")) or {"requests": 0}
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
        llm_usage = _llm_usage_snapshot(params.get("service_name", ""))
        if llm_usage is not None:
            error_event["llm_usage"] = llm_usage
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
                # Ingest the produced PDFs into MongoDB (the source of truth)
                # before logging, so the metadata reflects what was stored.
                _ingest_job_files(job_id, _jobs[job_id])
                _append_job_log(
                    job_id,
                    _jobs[job_id],
                    {
                        "job_id": job_id,
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
                    {"job_id": job_id, "status": "error", "message": msg},
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
    try:
        yield
    finally:
        _artifact_store.close()


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


# OpenAI-liked model chosen by the `service` knob on POST /v1/service/translate.
_SERVICE_OPENAILIKED_MODEL = {
    "fast": "qwen3.6-flash",
    "precise": "qwen3.6-plus",
}

# When `use_ollama=True` on POST /v1/service/translate, the job is routed to the
# Ollama translator and `service` selects the OLLAMA_MODEL instead of the
# OpenAI-liked qwen models.
_SERVICE_OLLAMA_MODEL = {
    "fast": "gemma4:e4b",
    "precise": "qwen3.6:35b",
}


async def _submit_translate_job(
    request: Request,
    *,
    file: Optional[UploadFile],
    link: str,
    service: str,
    lang_from: str,
    lang_to: str,
    page_range: str,
    page_input: str,
    prompt: str,
    threads: int,
    skip_subset_fonts: bool,
    ignore_cache: bool,
    vfont: str,
    mode_choice: str,
    env_overrides: dict[str, str] | None = None,
) -> dict:
    """Core translation-job submission shared by the public translate endpoints.

    ``env_overrides`` force-sets resolved translator envs (e.g. selecting an
    OpenAI-liked model) after the per-request envs are resolved.
    """
    mode_choice = _validate_mode_choice(mode_choice)

    if service not in SERVICE_MAP:
        raise HTTPException(400, f"Unknown service '{service}'. "
                            f"Valid values: {sorted(SERVICE_MAP)}")

    translator = SERVICE_MAP[service]
    # Translator settings arrive as env_0..env_N form fields; the count varies
    # per translator, so collect them dynamically instead of fixed parameters.
    form = await request.form()
    submitted_envs: list[str] = []
    while (value := form.get(f"env_{len(submitted_envs)}")) is not None:
        submitted_envs.append(str(value))
    envs = _resolve_translator_envs(service, submitted_envs)
    if env_overrides:
        envs.update(env_overrides)
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
        "client_ip": _client_ip(request),
        "source": str(file_path),
        "mono": None,
        "dual": None,
        "error": None,
        "llm_usage": None,
        "started_at": time.time(),
        "finished_at": None,
    }

    # Persist the initial job snapshot so MongoDB has the record even if the
    # worker crashes before any lifecycle event is logged (no-op when disabled).
    try:
        _artifact_store.record(
            job_id,
            {
                "status": "running",
                "client_ip": _jobs[job_id]["client_ip"],
                "service": service,
                "files": _job_file_names(_jobs[job_id]),
                "source": str(file_path),
                "started_at": _jobs[job_id]["started_at"],
            },
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "status": "submitted",
                "response": {"job_id": job_id, "status": "running"},
            },
        )
    except Exception:
        logger.exception("Unable to persist job %s submission to MongoDB", job_id)

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


@app.post("/v1/translate", status_code=202)
async def create_translate_job(
    request: Request,
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
) -> dict:
    """Submit a translation job.

    Returns 202 Accepted with ``{"job_id": "<uuid>"}`` immediately.
    Poll ``GET /v1/translate/{job_id}`` for status.
    """
    return await _submit_translate_job(
        request,
        file=file,
        link=link,
        service=service,
        lang_from=lang_from,
        lang_to=lang_to,
        page_range=page_range,
        page_input=page_input,
        prompt=prompt,
        threads=threads,
        skip_subset_fonts=skip_subset_fonts,
        ignore_cache=ignore_cache,
        vfont=vfont,
        mode_choice=mode_choice,
    )


@app.post("/v1/service/translate", status_code=202)
async def create_translate_job_via_service(
    request: Request,
    file: Optional[UploadFile] = File(None),
    link: Annotated[str, Form()] = "",
    service: Annotated[str, Form()] = "fast",
    lang_from: Annotated[str, Form()] = "English",
    lang_to: Annotated[str, Form()] = "Simplified Chinese",
    page_range: Annotated[str, Form()] = "All",
    page_input: Annotated[str, Form()] = "",
    prompt: Annotated[str, Form()] = "",
    threads: Annotated[int, Form()] = 4,
    skip_subset_fonts: Annotated[bool, Form()] = False,
    ignore_cache: Annotated[bool, Form()] = False,
    vfont: Annotated[str, Form()] = "",
    use_ollama: Annotated[bool, Form()] = False,
) -> dict:
    """Submit a translation job driven by ``service`` (``fast`` / ``precise``).

    Both services run on the v1 (``fast``) kernel and only ``OpenAI-liked``;
    ``service`` selects the model: ``fast`` -> ``qwen3.6-flash``,
    ``precise`` -> ``qwen3.6-plus``.

    When ``use_ollama=True`` the job is instead routed to the ``Ollama``
    translator; ``service`` still selects the model: ``fast`` -> ``gemma4:e4b``,
    ``precise`` -> ``qwen3.6:35b``.

    Returns 202 Accepted with ``{"job_id": "<uuid>"}`` immediately.
    """
    mode = (service or "fast").strip().lower()

    if use_ollama:
        ollama_model = _SERVICE_OLLAMA_MODEL.get(mode)
        if ollama_model is None:
            raise HTTPException(
                400,
                f"Unknown service '{service}'. "
                f"Valid values: {sorted(_SERVICE_OLLAMA_MODEL)}",
            )
        return await _submit_translate_job(
            request,
            file=file,
            link=link,
            service="Ollama",
            lang_from=lang_from,
            lang_to=lang_to,
            page_range=page_range,
            page_input=page_input,
            prompt=prompt,
            threads=threads,
            skip_subset_fonts=skip_subset_fonts,
            ignore_cache=ignore_cache,
            vfont=vfont,
            mode_choice="fast",
            env_overrides={"OLLAMA_MODEL": ollama_model},
        )

    model = _SERVICE_OPENAILIKED_MODEL.get(mode)
    if model is None:
        raise HTTPException(
            400,
            f"Unknown service '{service}'. "
            f"Valid values: {sorted(_SERVICE_OPENAILIKED_MODEL)}",
        )
    return await _submit_translate_job(
        request,
        file=file,
        link=link,
        service="OpenAI-liked",
        lang_from=lang_from,
        lang_to=lang_to,
        page_range=page_range,
        page_input=page_input,
        prompt=prompt,
        threads=threads,
        skip_subset_fonts=skip_subset_fonts,
        ignore_cache=ignore_cache,
        vfont=vfont,
        # Both "fast" and "precise" use the v1 kernel; only the model differs.
        mode_choice="fast",
        env_overrides={"OPENAILIKED_MODEL": model},
    )


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


@app.get("/v1/translate/{job_id}/both")
def download_both_results(
    job_id: str,
    as_zip: Annotated[bool, Query(alias="zip")] = False,
) -> Response:
    """Download both translated PDFs (mono + dual) in a single response.

    By default the two PDFs are returned **unzipped** as a ``multipart/mixed``
    response. Pass ``?zip=true`` to receive them bundled as a ZIP archive.

    Declared before the ``{variant}`` route so the literal ``both`` path wins
    matching. Returns 404 unless both variants are present in MongoDB.
    """
    # A running job has nothing to serve yet; report that distinctly.
    job = _jobs.get(job_id)
    if job and job["status"] == "running":
        raise HTTPException(409, f"Job not finished (status: {job['status']})")
    if not _artifact_store.available():
        raise HTTPException(503, "Artifact storage (MongoDB) is unavailable")

    files: list[tuple[str, bytes]] = []
    for variant in ("mono", "dual"):
        result = _artifact_store.get_file({"job_id": job_id, "variant": variant})
        if result is None:
            raise HTTPException(404, f"{variant} file not found")
        data, filename = result
        files.append((filename, data))

    if as_zip:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for filename, data in files:
                archive.writestr(filename, data)
        return Response(
            content=buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{job_id}.zip"'},
        )

    # Default: return both PDFs unzipped as a multipart/mixed payload.
    boundary = uuid.uuid4().hex
    chunks: list[bytes] = []
    for filename, data in files:
        chunks.append(
            (
                f"--{boundary}\r\n"
                f"Content-Type: application/pdf\r\n"
                f'Content-Disposition: attachment; filename="{filename}"\r\n\r\n'
            ).encode("utf-8")
        )
        chunks.append(data)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return Response(
        content=b"".join(chunks),
        media_type=f"multipart/mixed; boundary={boundary}",
    )


@app.get("/v1/translate/{job_id}/record")
def get_job_record(job_id: str) -> dict:
    """Return the persisted job-artefact metadata document from MongoDB.

    This is the same document written in ``_append_job_log`` (status, service,
    client IP, file names, LLM usage, timings) plus the ``events`` audit trail.
    Declared before the ``{variant}`` route so the literal ``record`` path wins
    matching.
    """
    if not _artifact_store.available():
        raise HTTPException(503, "Artifact storage (MongoDB) is unavailable")
    record = _artifact_store.get(job_id)
    if record is None:
        raise HTTPException(404, "Job record not found")
    return record


@app.get("/v1/translate/{job_id}/{variant}")
def download_result(job_id: str, variant: str) -> Response:
    """Download the translated PDF from MongoDB.  ``variant`` is ``mono``/``dual``."""
    if variant not in {"mono", "dual"}:
        raise HTTPException(400, "variant must be 'mono' or 'dual'")
    # A running job has nothing to serve yet; report that distinctly.
    job = _jobs.get(job_id)
    if job and job["status"] == "running":
        raise HTTPException(409, f"Job not finished (status: {job['status']})")
    if not _artifact_store.available():
        raise HTTPException(503, "Artifact storage (MongoDB) is unavailable")
    result = _artifact_store.get_file({"job_id": job_id, "variant": variant})
    if result is None:
        raise HTTPException(404, f"{variant} file not found")
    data, filename = result
    return Response(
        content=data,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/v1/metrics/frontend")
def get_frontend_metrics() -> FileResponse:
    """Return the frontend_metrics.md log file."""
    if not FRONTEND_METRICS.exists() or FRONTEND_METRICS.stat().st_size == 0:
        raise HTTPException(404, "No frontend metrics recorded yet")
    return FileResponse(
        str(FRONTEND_METRICS),
        media_type="text/markdown",
        filename=FRONTEND_METRICS.name,
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
