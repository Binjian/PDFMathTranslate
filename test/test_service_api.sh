#!/usr/bin/env bash
set -uo pipefail

# Exercises the new "service" API documented in service_readme.md:
#   POST   /v1/service/translate          submit with a single fast/precise knob
#   GET    /v1/translate/{job_id}          poll status / progress
#   GET    /v1/translate/{job_id}/mono     download the mono PDF
#   GET    /v1/translate/{job_id}/dual     download the dual PDF
#   GET    /v1/translate/{job_id}/both     download both (multipart/mixed, and ?zip=true)
#   GET    /v1/translate/{job_id}/record   fetch the MongoDB metadata document
#   DELETE /v1/translate/{job_id}/artifacts  remove stored PDFs
#
# Unlike test_translate_service.sh, the translator, credentials and model are
# resolved server-side; the client only chooses service=fast|precise and
# (optionally) use_ollama to translate locally with Ollama instead of the
# default OpenAI-liked backend.

# Resolve sample PDFs relative to this script so it runs from any directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

API_BASE_URL="${API_BASE_URL:-http://10.2.2.94:7861}"
# API_BASE_URL="${API_BASE_URL:-http://172.27.74.16:7861}"
PDF_FILE_EN="${PDF_FILE:-${SCRIPT_DIR}/attention_is_all_you_need_1706.03762v7.pdf}"
PDF_FILE_ZH="${PDF_FILE:-${SCRIPT_DIR}/attention_is_all_you_need_1706.03762v7_zh.pdf}"
IGNORE_CACHE="${IGNORE_CACHE:-false}"
OUTPUT_DIR="${OUTPUT_DIR:-service_api_output}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"
MAX_POLLS="${MAX_POLLS:-600}"
CURL=(curl --noproxy "*")

# service knob — backend resolves translator/credentials/model. The model
# depends on the backend:
#   OpenAI-liked (default): fast -> qwen3.6-flash, precise -> qwen3.6-plus
#   Ollama (use_ollama=true): fast -> gemma4:e4b,  precise -> qwen3.6:35b
SERVICE_CASES=(
  "fast"
  "precise"
)

# Translation backend. "openai" uses the default OpenAI-liked qwen models;
# "ollama" submits use_ollama=true. Override with e.g. BACKENDS="openai".
BACKENDS="${BACKENDS:-openai ollama}"
read -r -a BACKEND_CASES <<<"$BACKENDS"

# lang_from|lang_to
LANG_CASES=(
  "English|Simplified Chinese"
  "Simplified Chinese|English"
)

mkdir -p "$OUTPUT_DIR"

json_get() {
  python -c 'import json, sys; print(json.load(sys.stdin).get(sys.argv[1], ""))' "$1"
}

run_case() {
  local backend="$1" service="$2" lang_from="$3" lang_to="$4"

  local pdf_file
  [[ "$lang_from" == "English" ]] && pdf_file="$PDF_FILE_EN" || pdf_file="$PDF_FILE_ZH"

  if [[ ! -f "$pdf_file" ]]; then
    echo "Sample PDF not found: ${pdf_file} (set PDF_FILE to override)" >&2
    return 1
  fi

  # The Ollama backend is opt-in via the use_ollama flag.
  local backend_args=()
  [[ "$backend" == "ollama" ]] && backend_args=(-F "use_ollama=true")

  echo "Submitting service job to ${API_BASE_URL}/v1/service/translate"
  local submit_response
  submit_response="$(
    "${CURL[@]}" --fail --silent --show-error \
      "${API_BASE_URL}/v1/service/translate" \
      -F "file=@${pdf_file}" \
      -F "service=${service}" \
      -F "lang_from=${lang_from}" \
      -F "lang_to=${lang_to}" \
      -F "ignore_cache=${IGNORE_CACHE}" \
      "${backend_args[@]}"
  )" || return 1

  local job_id
  job_id="$(printf '%s' "$submit_response" | json_get job_id)"
  if [[ -z "$job_id" ]]; then
    echo "Submit response did not include job_id:" >&2
    echo "$submit_response" >&2
    return 1
  fi

  echo "Job ID: $job_id"

  local status="" status_response progress message error i
  for ((i = 1; i <= MAX_POLLS; i++)); do
    status_response="$(
      "${CURL[@]}" --fail --silent --show-error \
        "${API_BASE_URL}/v1/translate/${job_id}"
    )" || return 1

    status="$(printf '%s' "$status_response" | json_get status)"
    progress="$(printf '%s' "$status_response" | json_get progress)"
    message="$(printf '%s' "$status_response" | json_get message)"
    error="$(printf '%s' "$status_response" | json_get error)"

    printf '[%03d/%03d] status=%s progress=%s message=%s\n' \
      "$i" "$MAX_POLLS" "$status" "$progress" "$message"

    if [[ "$status" == "done" ]]; then
      break
    fi

    if [[ "$status" == "error" ]]; then
      echo "Translation failed: ${error:-$message}" >&2
      return 1
    fi

    sleep "$POLL_INTERVAL"
  done

  if [[ "$status" != "done" ]]; then
    echo "Timed out waiting for job ${job_id}" >&2
    return 1
  fi

  # ── Single-variant downloads ─────────────────────────────────────────────
  local mono_path="${OUTPUT_DIR}/${job_id}-mono.pdf"
  local dual_path="${OUTPUT_DIR}/${job_id}-dual.pdf"

  echo "Downloading mono PDF to $mono_path"
  "${CURL[@]}" --fail --silent --show-error \
    "${API_BASE_URL}/v1/translate/${job_id}/mono" \
    --output "$mono_path" || return 1

  echo "Downloading dual PDF to $dual_path"
  "${CURL[@]}" --fail --silent --show-error \
    "${API_BASE_URL}/v1/translate/${job_id}/dual" \
    --output "$dual_path" || return 1

  # ── GET /both (multipart/mixed, the default) ─────────────────────────────
  local both_multipart="${OUTPUT_DIR}/${job_id}-both.multipart"
  local both_headers="${OUTPUT_DIR}/${job_id}-both.headers"
  echo "Downloading both (multipart/mixed) to $both_multipart"
  "${CURL[@]}" --fail --silent --show-error -D "$both_headers" \
    "${API_BASE_URL}/v1/translate/${job_id}/both" \
    --output "$both_multipart" || return 1

  if ! grep -qi 'content-type:[[:space:]]*multipart/mixed' "$both_headers"; then
    echo "GET /both did not return multipart/mixed:" >&2
    grep -i 'content-type' "$both_headers" >&2
    return 1
  fi
  local pdf_parts
  pdf_parts="$(grep -a -c '%PDF' "$both_multipart")" || true
  if [[ "${pdf_parts:-0}" -lt 2 ]]; then
    echo "GET /both multipart body did not contain two PDF parts (found ${pdf_parts:-0})" >&2
    return 1
  fi

  # ── GET /both?zip=true (single zip archive) ──────────────────────────────
  local both_zip="${OUTPUT_DIR}/${job_id}-both.zip"
  echo "Downloading both (zip) to $both_zip"
  "${CURL[@]}" --fail --silent --show-error \
    "${API_BASE_URL}/v1/translate/${job_id}/both?zip=true" \
    --output "$both_zip" || return 1

  if [[ "$(head -c2 "$both_zip")" != "PK" ]]; then
    echo "GET /both?zip=true did not return a zip archive (bad magic bytes)" >&2
    return 1
  fi
  if command -v unzip >/dev/null 2>&1; then
    local zip_entries
    zip_entries="$(unzip -Z1 "$both_zip" 2>/dev/null | grep -c -i '\.pdf')" || true
    if [[ "${zip_entries:-0}" -lt 2 ]]; then
      echo "GET /both?zip=true archive did not contain two PDFs (found ${zip_entries:-0})" >&2
      return 1
    fi
  fi

  # ── GET /record (MongoDB metadata document) ──────────────────────────────
  echo "Fetching record for job ${job_id}"
  local record_response
  record_response="$(
    "${CURL[@]}" --fail --silent --show-error \
      "${API_BASE_URL}/v1/translate/${job_id}/record"
  )" || return 1

  echo "$record_response" > "${OUTPUT_DIR}/${job_id}-record.json"
  if ! printf '%s' "$record_response" | python -c '
import json, sys
doc = json.load(sys.stdin)
job_id = sys.argv[1]
assert (doc.get("job_id") or doc.get("_id")) == job_id, "record job_id mismatch"
assert doc.get("status") == "done", "record status=" + repr(doc.get("status"))
files = doc.get("files") or []
assert isinstance(files, list) and files, "record has no files"
for key in ("service", "llm_requests", "llm_prompt_tokens",
            "llm_completion_tokens", "llm_total_tokens"):
    assert key in doc, "record missing field " + key
' "$job_id"; then
    echo "GET /record document failed validation:" >&2
    echo "$record_response" >&2
    return 1
  fi

  echo "Done:"
  ls -lh "$mono_path" "$dual_path" "$both_multipart" "$both_zip"

  # ── Cleanup ──────────────────────────────────────────────────────────────
  echo "Deleting remote artifacts for job ${job_id}"
  local delete_response
  delete_response="$(
    "${CURL[@]}" --fail --silent --show-error \
      -X DELETE \
      "${API_BASE_URL}/v1/translate/${job_id}/artifacts"
  )" || return 1

  local delete_status
  delete_status="$(printf "%s" "$delete_response" | json_get status)"
  if [[ "$delete_status" != "artifacts_removed" ]]; then
    echo "Artifact delete response did not report artifacts_removed:" >&2
    echo "$delete_response" >&2
    return 1
  fi

  if ! printf "%s" "$delete_response" | python -c 'import json, sys; payload = json.load(sys.stdin); removed = payload.get("removed_files") or []; sys.exit(not (any(name.endswith(".pdf") and not name.endswith("mono.pdf") and not name.endswith("dual.pdf") for name in removed) and any(name.endswith("mono.pdf") for name in removed) and any(name.endswith("dual.pdf") for name in removed)))'; then
    echo "Artifact delete response did not include the source, mono, and dual PDFs:" >&2
    echo "$delete_response" >&2
    return 1
  fi

  echo "Remote artifacts deleted"
  return 0
}

total=0
passed=0
failed_cases=()

for backend in "${BACKEND_CASES[@]}"; do
  for service in "${SERVICE_CASES[@]}"; do
    for lang_case in "${LANG_CASES[@]}"; do
      IFS='|' read -r lang_from lang_to <<<"$lang_case"
      total=$((total + 1))
      case_label="backend=${backend} service=${service} ${lang_from}->${lang_to}"
      echo
      echo "=============================================================="
      echo "Case ${total}: ${case_label}"
      echo "=============================================================="
      # Comment the following case if you insist on testing the full supported functionalities.
      if [[ "$lang_from" == "Simplified Chinese" && "$service" == "precise" ]]; then
        echo "SKIP: ${case_label} (Simplified Chinese -> English with precise service is supported but skipped due to very long duration!)"
        continue
      fi
      if run_case "$backend" "$service" "$lang_from" "$lang_to"; then
        passed=$((passed + 1))
        echo "PASS: ${case_label}"
      else
        failed_cases+=("$case_label")
        echo "FAIL: ${case_label}" >&2
      fi
    done
  done
done

echo
echo "=============================================================="
echo "Summary: ${passed}/${total} cases passed"
if ((${#failed_cases[@]} > 0)); then
  echo "Failed cases:" >&2
  for c in "${failed_cases[@]}"; do
    echo "  - $c" >&2
  done
  exit 1
fi
echo "All cases passed"
