#!/usr/bin/env python3
"""Subprocess worker — runs pdf2zh_next translation in an isolated venv.

Protocol:
  - stdin:  JSON array of CLI args (e.g. ["file.pdf", "--lang-out", "zh", "--openai"])
  - stdout: JSON result (last line, after all progress events)
  - stderr: JSON-lines progress events and log output

This script is executed by PreciseKernel using the venv's Python interpreter.
v2's ConfigManager handles all config parsing from sys.argv + PDF2ZH_* env vars.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time


def _redirect_stdout_to_stderr():
    """Redirect stdout to stderr so library log output doesn't pollute JSON results.

    We save the real stdout fd for writing the final JSON result.
    """
    real_stdout_fd = os.dup(1)  # save fd 1
    os.dup2(2, 1)  # point fd 1 → stderr
    return os.fdopen(real_stdout_fd, "w")


# Redirect before any pdf2zh_next imports (they configure logging on import)
_real_stdout = _redirect_stdout_to_stderr()


def _patched_clean_json_output(self, llm_output: str) -> str:
    """Drop-in replacement for babeldoc's _clean_json_output.

    Extends the stock implementation with two repairs for outputs produced by
    small LLMs (e.g. gemma4:e4b) that otherwise cause json.loads() to raise:
      - Invalid \\escape sequences (e.g. \\p in math text) → escape the backslash.
      - Missing commas between adjacent JSON objects/arrays → insert them.
    """
    llm_output = llm_output.strip()
    if llm_output.startswith("<json>"):
        llm_output = llm_output[6:]
    if llm_output.endswith("</json>"):
        llm_output = llm_output[:-7]
    if llm_output.startswith("```json"):
        llm_output = llm_output[7:]
    if llm_output.startswith("```"):
        llm_output = llm_output[3:]
    if llm_output.endswith("```"):
        llm_output = llm_output[:-3]
    llm_output = llm_output.strip()
    # Fix invalid JSON escape sequences — valid ones are: " \ / b f n r t uXXXX
    llm_output = re.sub(r'\\(?!["\\/bfnrtu]|u[0-9a-fA-F]{4})', r'\\\\', llm_output)
    # Fix missing commas between adjacent JSON objects or arrays
    llm_output = re.sub(r'\}\s*\n(\s*\{)', r'},\n\1', llm_output)
    llm_output = re.sub(r'\]\s*\n(\s*\[)', r'],\n\1', llm_output)
    return llm_output


def _patch_babeldoc_json_cleaning() -> None:
    """Monkey-patch babeldoc's JSON cleaning in both LLM translation modules."""
    try:
        from babeldoc.format.pdf.document_il.midend import (
            automatic_term_extractor,
            il_translator_llm_only,
        )
        il_translator_llm_only.ILTranslatorLLMOnly._clean_json_output = (
            _patched_clean_json_output
        )
        automatic_term_extractor.AutomaticTermExtractor._clean_json_output = (
            _patched_clean_json_output
        )
    except Exception:
        pass  # babeldoc unavailable or API changed; proceed without patch


async def run_translation(cli_args: list[str]) -> dict:
    """Execute translation using v2's own config parsing."""
    _patch_babeldoc_json_cleaning()

    # Patch sys.argv so ConfigManager.initialize_config() picks up our args
    sys.argv = ["pdf2zh_next"] + cli_args

    from pdf2zh_next.config.main import ConfigManager
    from pdf2zh_next.high_level import do_translate_async_stream

    settings = ConfigManager().initialize_config()

    # Extract input files from parsed settings
    input_files = list(settings.basic.input_files)
    settings.basic.input_files = set()

    results = []
    start_time = time.time()

    for file_path in input_files:
        try:
            async for event in do_translate_async_stream(settings, file_path):
                event_type = event.get("type", "")

                if event_type in ("progress_start", "progress_update", "progress_end"):
                    # Forward babeldoc progress events with their actual fields
                    progress_event = {
                        "type": event_type,
                        "stage": event.get("stage", ""),
                        "stage_progress": event.get("stage_progress", 0.0),
                        "stage_current": event.get("stage_current", 0),
                        "stage_total": event.get("stage_total", 0),
                        "overall_progress": event.get("overall_progress", 0.0),
                        "part_index": event.get("part_index", 0),
                        "total_parts": event.get("total_parts", 0),
                    }
                    print(json.dumps(progress_event), file=sys.stderr, flush=True)

                elif event_type == "finish":
                    tr = event.get("translate_result")
                    result = {
                        "mono_pdf": (
                            str(tr.mono_pdf_path) if tr and tr.mono_pdf_path else None
                        ),
                        "dual_pdf": (
                            str(tr.dual_pdf_path) if tr and tr.dual_pdf_path else None
                        ),
                        "time_cost": tr.total_seconds if tr else 0.0,
                    }
                    results.append(result)

                elif event_type == "error":
                    raise RuntimeError(event.get("error", "Unknown error"))

        except Exception as e:
            error_event = {"type": "error", "message": str(e)}
            print(json.dumps(error_event), file=sys.stderr, flush=True)
            raise

    elapsed = time.time() - start_time
    return {"results": results, "time_cost": elapsed}


def main():
    raw = sys.stdin.read()
    try:
        cli_args = json.loads(raw)
    except json.JSONDecodeError as e:
        error = {"type": "error", "message": f"Invalid JSON input: {e}"}
        print(json.dumps(error), file=sys.stderr, flush=True)
        sys.exit(1)

    if not isinstance(cli_args, list):
        error = {"type": "error", "message": "Expected JSON array of CLI args"}
        print(json.dumps(error), file=sys.stderr, flush=True)
        sys.exit(1)

    result = asyncio.run(run_translation(cli_args))
    _real_stdout.write(json.dumps(result) + "\n")
    _real_stdout.flush()


if __name__ == "__main__":
    main()
