#!/usr/bin/env bash
set -euo pipefail

API_BASE_URL="${API_BASE_URL:-http://172.27.74.16:7861}"
PDF_FILE="${PDF_FILE:-maxent-2008.pdf}"
# PDF_FILE="${PDF_FILE:-VAR2404.pdf}"
SERVICE="${SERVICE:-Ollama}"
MODE_CHOICE="${MODE_CHOICE:-fast}"
LANG_FROM="${LANG_FROM:-English}"
LANG_TO="${LANG_TO:-Simplified Chinese}"
OLLAMA_HOST="${OLLAMA_HOST:-http://172.27.74.16:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.6:latest}"
IGNORE_CACHE="${IGNORE_CACHE:-false}"
OUTPUT_DIR="${OUTPUT_DIR:-translate_service_output}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"
MAX_POLLS="${MAX_POLLS:-300}"

mkdir -p "$OUTPUT_DIR"

json_get() {
  python -c 'import json, sys; print(json.load(sys.stdin).get(sys.argv[1], ""))' "$1"
}

echo "Submitting translation job to ${API_BASE_URL}/v1/translate"
submit_response="$(
  curl --fail --silent --show-error \
    "${API_BASE_URL}/v1/translate" \
    -F "file=@${PDF_FILE}" \
    -F "service=${SERVICE}" \
    -F "mode_choice=${MODE_CHOICE}" \
    -F "lang_from=${LANG_FROM}" \
    -F "lang_to=${LANG_TO}" \
    -F "ignore_cache=${IGNORE_CACHE}" \
    -F "env_0=${OLLAMA_HOST}" \
    -F "env_1=${OLLAMA_MODEL}"
)"

job_id="$(printf '%s' "$submit_response" | json_get job_id)"
if [[ -z "$job_id" ]]; then
  echo "Submit response did not include job_id:" >&2
  echo "$submit_response" >&2
  exit 1
fi

echo "Job ID: $job_id"

status=""
for ((i = 1; i <= MAX_POLLS; i++)); do
  status_response="$(
    curl --fail --silent --show-error \
      "${API_BASE_URL}/v1/translate/${job_id}"
  )"

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
    exit 1
  fi

  sleep "$POLL_INTERVAL"
done

if [[ "$status" != "done" ]]; then
  echo "Timed out waiting for job ${job_id}" >&2
  exit 1
fi

mono_path="${OUTPUT_DIR}/${job_id}-mono.pdf"
dual_path="${OUTPUT_DIR}/${job_id}-dual.pdf"

echo "Downloading mono PDF to $mono_path"
curl --fail --silent --show-error \
  "${API_BASE_URL}/v1/translate/${job_id}/mono" \
  --output "$mono_path"

echo "Downloading dual PDF to $dual_path"
curl --fail --silent --show-error \
  "${API_BASE_URL}/v1/translate/${job_id}/dual" \
  --output "$dual_path"

echo "Done:"
ls -lh "$mono_path" "$dual_path"

echo "Deleting remote artifacts for job ${job_id}"
delete_response="$(
  curl --fail --silent --show-error \
    -X DELETE \
    "${API_BASE_URL}/v1/translate/${job_id}/artifacts"
)"

delete_status="$(printf "%s" "$delete_response" | json_get status)"
if [[ "$delete_status" != "artifacts_removed" ]]; then
  echo "Artifact delete response did not report artifacts_removed:" >&2
  echo "$delete_response" >&2
  exit 1
fi

if ! printf "%s" "$delete_response" | python -c 'import json, sys; payload = json.load(sys.stdin); removed = payload.get("removed_files") or []; sys.exit(not (any(name.endswith(".pdf") and not name.endswith("-mono.pdf") and not name.endswith("-dual.pdf") for name in removed) and any(name.endswith("-mono.pdf") for name in removed) and any(name.endswith("-dual.pdf") for name in removed)))'; then
  echo "Artifact delete response did not include the source, mono, and dual PDFs:" >&2
  echo "$delete_response" >&2
  exit 1
fi

echo "Remote artifacts deleted"
