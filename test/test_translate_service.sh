#!/usr/bin/env bash
set -uo pipefail

# Loops over every supported option combination from README.md:
#   services:    Ollama (qwen3.6:latest), OpenAI-liked (qwen3.6-flash, qwen3.6-plus)
#   translation: English->Chinese, Chinese->English
#   modes:       fast, precise

API_BASE_URL="${API_BASE_URL:-http://10.2.2.94:7861}"
# API_BASE_URL="${API_BASE_URL:-http://172.27.74.16:7861}"
PDF_FILE_EN="${PDF_FILE:-attention_is_all_you_need_1706.03762v7.pdf}"
PDF_FILE_ZH="${PDF_FILE:-attention_is_all_you_need_1706.03762v7_zh.pdf}"
# PDF_FILE="${PDF_FILE:-maxent-2008.pdf}"
# PDF_FILE="${PDF_FILE:-VAR2404.pdf}"
OLLAMA_HOST="${OLLAMA_HOST:-http://172.27.74.16:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.6:latest}"
IGNORE_CACHE="${IGNORE_CACHE:-false}"
OUTPUT_DIR="${OUTPUT_DIR:-translate_service_output}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"
MAX_POLLS="${MAX_POLLS:-300}"
CURL=(curl --noproxy "*")

# service|model — env_* mapping depends on the service:
#   Ollama:      env_0=OLLAMA_HOST, env_1=OLLAMA_MODEL
#   OpenAI-liked: env_0/env_1 (base URL, API key) are backend-only and resolved
#                 from the server environment; env_2=OPENAILIKED_MODEL
SERVICE_CASES=(
  "Ollama|${OLLAMA_MODEL}"
  "OpenAI-liked|qwen3.6-flash"
  "OpenAI-liked|qwen3.6-plus"
)

# lang_from|lang_to
LANG_CASES=(
  "English|Simplified Chinese"
  "Simplified Chinese|English"
)

MODE_CASES=(
  "fast"
  "precise"
)

mkdir -p "$OUTPUT_DIR"

json_get() {
  python -c 'import json, sys; print(json.load(sys.stdin).get(sys.argv[1], ""))' "$1"
}

run_case() {
  local service="$1" model="$2" lang_from="$3" lang_to="$4" mode="$5"

  local pdf_file
  [[ "$lang_from" == "English" ]] && pdf_file="$PDF_FILE_EN" || pdf_file="$PDF_FILE_ZH"

  local env_args=()
  if [[ "$service" == "Ollama" ]]; then
    env_args=(-F "env_0=${OLLAMA_HOST}" -F "env_1=${model}")
  else
    env_args=(-F "env_0=" -F "env_1=" -F "env_2=${model}")
  fi

  echo "Submitting translation job to ${API_BASE_URL}/v1/translate"
  local submit_response
  submit_response="$(
    "${CURL[@]}" --fail --silent --show-error \
      "${API_BASE_URL}/v1/translate" \
      -F "file=@${pdf_file}" \
      -F "service=${service}" \
      -F "mode_choice=${mode}" \
      -F "lang_from=${lang_from}" \
      -F "lang_to=${lang_to}" \
      -F "ignore_cache=${IGNORE_CACHE}" \
      "${env_args[@]}"
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

  echo "Done:"
  ls -lh "$mono_path" "$dual_path"

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

for service_case in "${SERVICE_CASES[@]}"; do
  IFS='|' read -r service model <<<"$service_case"
  for lang_case in "${LANG_CASES[@]}"; do
    IFS='|' read -r lang_from lang_to <<<"$lang_case"
    for mode in "${MODE_CASES[@]}"; do
      total=$((total + 1))
      case_label="service=${service} model=${model} ${lang_from}->${lang_to} mode=${mode}"
      echo
      echo "=============================================================="
      echo "Case ${total}: ${case_label}"
      echo "=============================================================="
      if run_case "$service" "$model" "$lang_from" "$lang_to" "$mode"; then
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
