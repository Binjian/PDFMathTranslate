[**Documentation**](https://github.com/Byaidu/PDFMathTranslate) > **FastAPI Backend** _(current)_

<h2 id="toc">Table of Contents</h2>

- [Overview](#overview)
- [Architecture](#architecture)
- [Starting the API server](#start)
- [Quick translation with curl](#curl-translate)
- [Environment variables](#env)
- [API reference](#api)
  - [Health check](#health)
  - [Submit a translation job](#submit)
  - [Poll job status](#status)
  - [Cancel a job](#cancel)
  - [Remove generated artifacts](#artifacts)
  - [Download results](#download)
- [Using the FastHTML GUI as a client](#gui-client)
- [Running both together](#together)
- [Code changes summary](#changes)
- [Cross-host fix: Connection refused](#crosshost)

---

<h2 id="overview">Overview</h2>

`pdf2zh` ships a lightweight **FastAPI translation backend** (`pdf2zh/api_server.py`) that exposes the full translation engine over HTTP — no Redis or Celery required.

It is an alternative to the existing Flask/Celery backend (`pdf2zh --flask`).  Key differences:

| | FastAPI backend | Flask/Celery backend |
|---|---|---|
| Extra services needed | None | Redis + Celery worker |
| Install extras | None | `pip install pdf2zh[backend]` |
| Framework | FastAPI + uvicorn | Flask + Celery |
| Progress format | `{"progress": 0.0–1.0, "message": "..."}` | `{"n": N, "total": T}` |
| Port (default) | 7861 | 11008 |

[⬆️ Back to top](#toc)

---

<h2 id="architecture">Architecture</h2>

```
┌────────────────────────┐        HTTP (httpx)       ┌────────────────────────┐
│  FastHTML GUI          │ ──── POST /v1/translate ──▶│  FastAPI backend       │
│  pdf2zh --gui          │ ──── GET  /v1/translate/…──▶│  pdf2zh.api_server     │
│  (port 7860)           │ ◀─── progress / files ────│  (port 7861)           │
└────────────────────────┘                            └────────────────────────┘
        ▲                                                       │
        │ browser                                    spawns subprocess
        └── user                                               │
                                                    ┌──────────────────────┐
                                                    │  translation engine  │
                                                    │  KernelRegistry      │
                                                    └──────────────────────┘
```

When `PDF2ZH_API_BASE_URL` is set the FastHTML GUI acts as a thin client:
it uploads the PDF, polls for progress, and downloads the finished files — the
heavy lifting happens entirely inside the API process.

Without `PDF2ZH_API_BASE_URL` the GUI behaves exactly as before (local
multiprocessing, no API involved).

[⬆️ Back to top](#toc)

---

<h2 id="start">Starting the API server</h2>

**Recommended — via the `pdf2zh` CLI (binds to `0.0.0.0` by default):**

```bash
pdf2zh --api                          # 0.0.0.0:7861
pdf2zh --api --api-port 8000          # custom port
pdf2zh --api --api-host 127.0.0.1    # localhost only
```

**Via Python module (also binds to `0.0.0.0` by default):**

```bash
python -m pdf2zh.api_server
```

**Via uvicorn directly (defaults to `127.0.0.1` — add `--host 0.0.0.0` for cross-host access):**

```bash
uvicorn pdf2zh.api_server:app --host 0.0.0.0 --port 7861
```

> **Warning:** omitting `--host 0.0.0.0` from a plain `uvicorn` command binds
> the server to `127.0.0.1` only, making it unreachable from other hosts and
> producing `[Errno 111] Connection refused` on the GUI side.  Use
> `pdf2zh --api` or `python -m pdf2zh.api_server` to avoid this.

**Programmatically:**

```python
from pdf2zh.api_server import run_api_server

run_api_server(host="0.0.0.0", port=7861)
```

The interactive docs (Swagger UI) are available at `http://127.0.0.1:7861/docs`
once the server is running.

[⬆️ Back to top](#toc)

---

<h2 id="curl-translate">Quick translation with curl</h2>

With the API server running, translate a local PDF from English to Simplified
Chinese using the default Google service:

```bash
curl http://127.0.0.1:7861/v1/translate \
  -F "file=@paper.pdf" \
  -F "service=Google" \
  -F "lang_from=English" \
  -F "lang_to=Simplified Chinese"
```

The request starts translation asynchronously and returns a job ID:

```json
{"job_id": "d9894125-2f4e-45ea-9d93-1a9068d2045a"}
```

Use that ID with the status and download endpoints documented below to retrieve
the translated PDF when the job is complete.

[⬆️ Back to top](#toc)

---

<h2 id="env">Environment variables</h2>

| Variable | Default | Description |
|---|---|---|
| `PDF2ZH_API_OUTPUT` | `pdf2zh_api_files` | Directory where translated PDFs are stored by the API server |
| `PDF2ZH_API_JOB_LOG` | `<PDF2ZH_API_OUTPUT>/job_log.md` | Markdown table recording job files, human-readable elapsed time, and completed, failed, cancelled, or cleanup responses |
| `PDF2ZH_API_HOST` | `0.0.0.0` | Bind address used by `run_api_server()` |
| `PDF2ZH_API_PORT` | `7861` | Port used by `run_api_server()` |
| `PDF2ZH_API_BASE_URL` | _(empty)_ | **GUI only** — URL of the FastAPI backend. When set, the FastHTML GUI delegates all translation to that server. Example: `http://127.0.0.1:7861` |

[⬆️ Back to top](#toc)

---

<h2 id="api">API reference</h2>

Base URL: `http://127.0.0.1:7861`

---

<h3 id="health">GET /health</h3>

Liveness check.

```bash
curl http://127.0.0.1:7861/health
```

```json
{"status": "ok", "version": "1.0.0"}
```

---

<h3 id="submit">POST /v1/translate</h3>

Submit a translation job.  Accepts `multipart/form-data`.  Returns `202 Accepted`
immediately with the job ID — translation runs asynchronously.

**File upload:**

```bash
curl http://127.0.0.1:7861/v1/translate \
  -F "file=@paper.pdf" \
  -F "service=Google" \
  -F "lang_from=English" \
  -F "lang_to=Simplified Chinese"
```

**Local Ollama model (`qwen3.6:latest`):**

Please refer to the test script in [test/test_translate_service.sh](../test/test_translate_service.sh).

**Form parameters:**

| Parameter | Default | Description |
|---|---|---|
| `file` | — | PDF file upload (mutually exclusive with `link`) |
| `link` | `""` | URL to a PDF (mutually exclusive with `file`) |
| `service` | `Google` | Translator service name (see table below) |
| `lang_from` | `English` | Source language display name |
| `lang_to` | `Simplified Chinese` | Target language display name |
| `page_range` | `All` | `All`, `First`, `First 5 pages`, or `Others` |
| `page_input` | `""` | Custom page range when `page_range=Others`, e.g. `1-3,5,7-9` |
| `prompt` | `""` | Custom prompt template for LLM-based translators |
| `threads` | `4` | Number of translation threads |
| `skip_subset_fonts` | `false` | Skip font subsetting step |
| `ignore_cache` | `false` | Bypass translation cache |
| `vfont` | `""` | Vertical font regex |
| `mode_choice` | `fast` | Kernel mode: `fast` or `precise` |
| `env_0`–`env_3` | `""` | Translator-specific config (API keys, endpoints, model names) |

**Supported service names:**

`Google`, `Bing`, `DeepL`, `DeepLX`, `Ollama`, `Xinference`, `AzureOpenAI`,
`OpenAI`, `Zhipu`, `ModelScope`, `Silicon`, `Gemini`, `Azure`, `Tencent`,
`Dify`, `AnythingLLM`, `Argos Translate`, `Grok`, `Groq`, `DeepSeek`,
`MiniMax`, `OpenAI-liked`, `Ali Qwen-Translation`, `302.AI`

**Language display names:**

`English`, `Simplified Chinese`, `Traditional Chinese`, `French`, `German`,
`Japanese`, `Korean`, `Russian`, `Spanish`, `Italian`

---

<h3 id="status">GET /v1/translate/{job_id}</h3>

Poll the status and progress of a job.

```bash
curl http://127.0.0.1:7861/v1/translate/d9894125-2f4e-45ea-9d93-1a9068d2045a
```

**While running:**

```json
{
  "job_id": "d9894125-2f4e-45ea-9d93-1a9068d2045a",
  "status": "running",
  "progress": 0.42,
  "message": "Translating page 21 / 50",
  "error": null,
  "elapsed_seconds": 18.3
}
```

**On success:**

```json
{
  "job_id": "d9894125-2f4e-45ea-9d93-1a9068d2045a",
  "status": "done",
  "progress": 1.0,
  "message": "Translation complete",
  "error": null,
  "elapsed_seconds": 42.1
}
```

**On failure:**

```json
{
  "job_id": "d9894125-2f4e-45ea-9d93-1a9068d2045a",
  "status": "error",
  "progress": 1.0,
  "message": "Worker crashed (exit code 1).",
  "error": "Worker crashed (exit code 1).",
  "elapsed_seconds": 5.0
}
```

`status` is one of `running`, `done`, or `error`.

---

<h3 id="cancel">DELETE /v1/translate/{job_id}</h3>

Terminate a running job.

```bash
curl -X DELETE \
  http://127.0.0.1:7861/v1/translate/d9894125-2f4e-45ea-9d93-1a9068d2045a
```

```json
{"status": "cancelled"}
```

---

<h3 id="artifacts">DELETE /v1/translate/{job_id}/artifacts</h3>

Remove the PDF files generated or stored for a job, including the uploaded source PDF. The alias `/v1/translate/{job_id}/artefacts` is also supported. Returns `409` if the job is still running.

```bash
curl -X DELETE \
  http://127.0.0.1:7861/v1/translate/d9894125-2f4e-45ea-9d93-1a9068d2045a/artifacts
```

```json
{
  "job_id": "d9894125-2f4e-45ea-9d93-1a9068d2045a",
  "status": "artifacts_removed",
  "removed_files": ["paper-dual.pdf", "paper-mono.pdf", "paper.pdf"]
}
```

---

<h3 id="download">GET /v1/translate/{job_id}/{variant}</h3>

Download a translated PDF.  `variant` is `mono` (translated language only) or
`dual` (original and translated pages interleaved).  Returns `409` if the job
is not yet finished.

```bash
# Monolingual output
curl http://127.0.0.1:7861/v1/translate/d9894125-2f4e-45ea-9d93-1a9068d2045a/mono \
  --output paper-mono.pdf

# Bilingual output
curl http://127.0.0.1:7861/v1/translate/d9894125-2f4e-45ea-9d93-1a9068d2045a/dual \
  --output paper-dual.pdf
```

**Python client example:**

```python
import time
import httpx

API = "http://127.0.0.1:7861"

# Submit
resp = httpx.post(f"{API}/v1/translate",
    data={"service": "Google", "lang_from": "English", "lang_to": "Simplified Chinese"},
    files={"file": ("paper.pdf", open("paper.pdf", "rb"), "application/pdf")},
)
resp.raise_for_status()
job_id = resp.json()["job_id"]

# Poll
while True:
    status = httpx.get(f"{API}/v1/translate/{job_id}").json()
    print(f"[{status['progress']:.0%}] {status['message']}")
    if status["status"] in ("done", "error"):
        break
    time.sleep(1)

# Download
if status["status"] == "done":
    for variant in ("mono", "dual"):
        data = httpx.get(f"{API}/v1/translate/{job_id}/{variant}").content
        open(f"paper-{variant}.pdf", "wb").write(data)
```

[⬆️ Back to top](#toc)

---

<h2 id="gui-client">Using the FastHTML GUI as a client</h2>

Set `PDF2ZH_API_BASE_URL` before starting the GUI:

```bash
# Terminal 1 — API backend (binds to 0.0.0.0 by default)
pdf2zh --api

# Terminal 2 — FastHTML GUI (acts as API client)
PDF2ZH_API_BASE_URL=http://127.0.0.1:7861 pdf2zh --gui
```

With this configuration:

1. The GUI uploads the user's PDF to `POST /v1/translate` via `httpx`.
2. A background thread polls `GET /v1/translate/{job_id}` every 500 ms and
   mirrors progress into the GUI's job store — the progress bar and status
   messages work exactly as in standalone mode.
3. When the job finishes, the GUI fetches the mono and dual PDFs from the API
   and saves them to its own `pdf2zh_files/` directory so the built-in
   `/file` and `/download` routes serve them without change.
4. Clicking **Cancel** in the GUI calls `DELETE /v1/translate/{job_id}` on the
   API to terminate the subprocess there.

No changes to the GUI's URL scheme or result pages are needed — the API
integration is transparent to the browser.

[⬆️ Back to top](#toc)

---

<h2 id="together">Running both together</h2>

**Docker Compose example:**

```yaml
services:
  api:
    image: pdf2zh
    command: python -m pdf2zh.api_server
    environment:
      PDF2ZH_API_OUTPUT: /data/api
    volumes:
      - pdf2zh_data:/data
    ports:
      - "7861:7861"

  gui:
    image: pdf2zh
    command: pdf2zh --gui
    environment:
      PDF2ZH_API_BASE_URL: http://api:7861
    volumes:
      - pdf2zh_data:/data
    ports:
      - "7860:7860"
    depends_on:
      - api

volumes:
  pdf2zh_data:
```

**Standalone (single machine):**

```bash
PDF2ZH_API_OUTPUT=./api-output pdf2zh --api &

PDF2ZH_API_BASE_URL=http://127.0.0.1:7861 \
  pdf2zh --gui --port 7860
```

[⬆️ Back to top](#toc)

---

<h2 id="changes">Code changes summary</h2>

### New file: `pdf2zh/api_server.py`

Self-contained FastAPI application.  Depends only on existing `pdf2zh`
internals (`KernelRegistry`, `TranslateRequest`, translator classes,
`ConfigManager`) — no new runtime dependencies beyond `fastapi` and `uvicorn`,
which were already required by the FastHTML GUI.

Key implementation details:

- Each job gets its own subdirectory under `PDF2ZH_API_OUTPUT/<job_id>/` so
  concurrent jobs never clobber each other's files.
- Translation runs in a **spawned subprocess** (same isolation model as the
  GUI) to keep the API process responsive and to allow clean `SIGTERM`
  cancellation.
- A daemon **monitor thread** per job drains the inter-process progress queue
  and updates the in-memory `_jobs` dict that the status endpoint reads.
- The job store is in-memory; restart the server and existing job IDs are lost.
  Files on disk remain until manually deleted.

### Changes to `pdf2zh/gui_fasthtml.py`

| Location | Change |
|---|---|
| Top-level imports | Added `import httpx` |
| After `GUI_ONNX` | Added `API_BASE_URL` constant read from `PDF2ZH_API_BASE_URL` env var |
| `stop_translate_file()` | Extended to call `DELETE /v1/translate/{api_job_id}` when an API-backed job is active |
| After `_translate_file_process()` | Added `_run_api_translation_job(session_id, params)` — the API client coroutine that runs in a daemon thread |
| `/translate` endpoint | Branches on `API_BASE_URL`: starts `_run_api_translation_job` thread (API mode) or original `run_translation_job` thread (local mode) |

### `pyproject.toml`

Added `httpx` to the main dependencies list (it was already transitively
installed but is now an explicit requirement).

[⬆️ Back to top](#toc)

---

<h2 id="crosshost">Cross-host fix: Connection refused</h2>

### Problem

When the FastAPI server and the FastHTML GUI run on different machines,
translation fails immediately with:

```
[Errno 111] Connection refused
```

Three issues combined to cause this:

| # | Root cause | Effect |
|---|-----------|--------|
| 1 | `uvicorn pdf2zh.api_server:app` binds to `127.0.0.1` by default | All TCP connections from a remote GUI host are refused at the OS level |
| 2 | `run_api_server()` evaluated `ConfigManager.get(...)` at **import time** as default parameter values | Host/port read from whichever machine imported the module first; could freeze the wrong value |
| 3 | `_run_api_translation_job` used a single combined `timeout=30` for all phases | Large PDF uploads timed out mid-transfer; `ConnectError` was surfaced as a raw unguided exception |

### Fix

**`pdf2zh/api_server.py` — compute host/port at call time**

`run_api_server()` no longer uses `ConfigManager.get(...)` as a default
argument (evaluated at import time). The values are now resolved inside the
function body each time it is called:

```python
# before — evaluated once at module import
def run_api_server(
    host: str = ConfigManager.get("PDF2ZH_API_HOST", "0.0.0.0"),
    ...

# after — read fresh on every call
def run_api_server(host: Optional[str] = None, port: Optional[int] = None):
    _host = host or (ConfigManager.get("PDF2ZH_API_HOST") or "0.0.0.0")
    _port = port or int(ConfigManager.get("PDF2ZH_API_PORT") or "7861")
    uvicorn.run(app, host=_host, port=_port)
```

**`pdf2zh/pdf2zh.py` — new `--api` CLI flag**

The `pdf2zh --api` command starts the API server via `run_api_server()`,
which defaults to `0.0.0.0` and therefore accepts connections from any host:

```bash
pdf2zh --api                          # 0.0.0.0:7861
pdf2zh --api --api-host 0.0.0.0      # explicit
pdf2zh --api --api-port 8000          # custom port
```

Three new argparse arguments were added to `create_parser()`:
`--api`, `--api-host`, `--api-port`.  The handler in `main()` calls
`run_api_server()` and returns before any model loading, matching the pattern
of `--flask` and `--interactive`.

**`pdf2zh/gui_fasthtml.py` — preflight check and per-phase timeouts**

`_run_api_translation_job` now:

1. **Preflight `GET /health`** before uploading the PDF.  On
   `httpx.ConnectError` it fails immediately with a message that names both
   the problem and the fix:

   ```
   Cannot connect to API server at http://…
   Make sure it is running and bound to 0.0.0.0, not 127.0.0.1.
   Start it with: pdf2zh --api  (or: python -m pdf2zh.api_server)
   ```

2. **Per-phase `httpx.Timeout`** instead of a single combined value:

   | Phase | Timeout |
   |-------|---------|
   | Connect | 10 s |
   | Write (PDF upload) | unlimited |
   | Read | 60 s (poll) / 300 s (download) |
   | Pool | 10 s |

   The unlimited write timeout prevents large PDFs from being cut off
   mid-upload on a slow link.

3. **`httpx.ConnectError` caught explicitly** at every call site (submit,
   poll, download) so every failure path produces a message that mentions the
   `0.0.0.0` bind requirement rather than a bare errno string.

### Quick reference

```bash
# Server (remote host or same machine — 0.0.0.0 required for cross-host)
pdf2zh --api --api-host 0.0.0.0 --api-port 7861

# GUI (any host)
PDF2ZH_API_BASE_URL=http://<server-ip>:7861 pdf2zh --gui
```

[⬆️ Back to top](#toc)
