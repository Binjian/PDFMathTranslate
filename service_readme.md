# pdf2zh API — New Endpoints

This document covers the endpoints added to [`pdf2zh/api_server.py`](pdf2zh/api_server.py)
for the simplified "service" translation flow and the MongoDB-backed artefact
retrieval.

Start the API server with:

```bash
uvicorn pdf2zh.api_server:app --host 0.0.0.0 --port 7861
# or: python -m pdf2zh.api_server
```

## MongoDB requirement

Translated PDFs and job metadata are stored in **MongoDB** (PDF binaries in
GridFS, metadata in a collection). The retrieval endpoints below read from
MongoDB and therefore require it to be configured and reachable:

| Variable | Default | Purpose |
|---|---|---|
| `PDF2ZH_API_MONGODB_URI` | *(unset)* | Connection URI. Falls back to `MONGODB_URI`. Retrieval is unavailable when unset. |
| `PDF2ZH_API_MONGODB_DB` | `pdf2zh` | Database name. |
| `PDF2ZH_API_MONGODB_COLLECTION` | `job_artifacts` | Metadata collection (GridFS uses the `<collection>_fs` prefix). |

When MongoDB is unreachable, retrieval endpoints return **503**.

---

## 1. `POST /v1/service/translate`

Submit a translation job with a single `service` knob, intended for an
OpenAI-compatible (`OpenAI-liked`) backend. The translator, its credentials and
the model are resolved **server-side** and never sent by the client.

`service` selects both the kernel mode and the model:

| `service` | kernel mode | model (`OPENAILIKED_MODEL`) |
|---|---|---|
| `fast` (default) | `fast` | `qwen3.6-flash` |
| `precise` | `precise` | `qwen3.6-plus` |

### Request — `multipart/form-data`

| Field | Type | Default | Notes |
|---|---|---|---|
| `file` | file | — | PDF upload. Provide **either** `file` or `link`. |
| `link` | string | `""` | URL to a PDF (used when no `file`). |
| `service` | string | `fast` | `fast` or `precise`. |
| `lang_from` | string | `English` | Source language. |
| `lang_to` | string | `Simplified Chinese` | Target language. |
| `page_range` | string | `All` | `All`, `First`, `First 5 pages`, or custom. |
| `page_input` | string | `""` | Custom pages, e.g. `1,3,5-7` (when `page_range` is custom). |
| `prompt` | string | `""` | Optional custom LLM prompt. |
| `threads` | int | `4` | Worker threads. |
| `skip_subset_fonts` | bool | `false` | |
| `ignore_cache` | bool | `false` | |
| `vfont` | string | `""` | Formula-font regex. |

### Responses

- `202 Accepted` → `{"job_id": "<uuid>"}`
- `400 Bad Request` → unknown `service` (only `fast`/`precise` are valid)

### Example

```bash
curl -X POST http://localhost:7861/v1/service/translate \
  -F "file=@paper.pdf" \
  -F "service=precise" \
  -F "lang_from=English" \
  -F "lang_to=Simplified Chinese"
# => {"job_id": "3f2c...-..."}
```

Poll status, then retrieve the results (see below):

```bash
curl http://localhost:7861/v1/translate/<job_id>          # status
```

> This is a thin wrapper over `POST /v1/translate`; both produce a **mono** and a
> **dual** PDF and share all the retrieval routes.

---

## 2. `GET /v1/translate/{job_id}/both`

Download **both** translated PDFs (mono + dual) in a single response.

### Query parameters

| Param | Type | Default | Behaviour |
|---|---|---|---|
| `zip` | bool | `false` | `false` → `multipart/mixed` with the two PDFs **unzipped**. `true` → a single `application/zip` archive. |

### Responses

- `200 OK`
  - default: `Content-Type: multipart/mixed; boundary=...` — two `application/pdf`
    parts, each with its own `Content-Disposition: attachment; filename="..."`.
  - `?zip=true`: `Content-Type: application/zip`, `Content-Disposition: attachment; filename="<job_id>.zip"`.
- `409 Conflict` — job still running.
- `404 Not Found` — either variant is missing.
- `503 Service Unavailable` — MongoDB unavailable.

### Examples

```bash
# Unzipped (multipart/mixed) — the default
curl http://localhost:7861/v1/translate/<job_id>/both -o both.multipart

# Zipped archive
curl "http://localhost:7861/v1/translate/<job_id>/both?zip=true" -o result.zip
```

---

## 3. `GET /v1/translate/{job_id}/record`

Return the persisted job-artefact **metadata document** from MongoDB.

### Response — `200 OK` (JSON)

The stored document, including:

| Field | Description |
|---|---|
| `_id` / `job_id` | The job id. |
| `status` | Latest status (`running`, `done`, `error`, `artifacts_removed`). |
| `client_ip` | Submitting client IP. |
| `service` | Translator service used. |
| `files` | Output file names. |
| `source`, `mono`, `dual` | Stored file references. |
| `elapsed_seconds` | Wall-clock duration. |
| `llm_usage` | Raw LLM usage snapshot. |
| `llm_requests`, `llm_prompt_tokens`, `llm_completion_tokens`, `llm_total_tokens` | Token counters. |
| `started_at`, `finished_at`, `created_at`, `updated_at` | Epoch timestamps. |
| `events` | Append-only audit trail of lifecycle events. |

### Other responses

- `404 Not Found` — no record for that `job_id`.
- `503 Service Unavailable` — MongoDB unavailable.

### Example

```bash
curl http://localhost:7861/v1/translate/<job_id>/record
```

---

## Related existing endpoints

These pre-existing routes complete the workflow:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/translate` | Full translation submission (explicit translator + `mode_choice`). |
| `GET` | `/v1/translate/{job_id}` | Poll job status / progress. |
| `GET` | `/v1/translate/{job_id}/mono` | Download the mono PDF (from MongoDB). |
| `GET` | `/v1/translate/{job_id}/dual` | Download the dual PDF (from MongoDB). |
| `DELETE` | `/v1/translate/{job_id}` | Cancel a running job. |
| `DELETE` | `/v1/translate/{job_id}/artifacts` | Remove a job's stored PDFs. |

> Route note: `both` and `record` are literal paths declared **before** the
> `{variant}` route, so they are not interpreted as a `mono`/`dual` variant.

## GUI

A **Quick** tab in the FastHTML GUI ([`pdf2zh/gui_fasthtml.py`](pdf2zh/gui_fasthtml.py))
drives `POST /v1/service/translate`: it exposes only the `fast`/`precise`
selector plus the standard translation options, and hides the translator
choice, credentials and model. It requires `PDF2ZH_API_BASE_URL` to point at a
running API server.
