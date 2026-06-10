import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pdf2zh import api_server, gui_fasthtml


class _FakeClient:
    created_with = []
    requests = []

    def __init__(self, **kwargs):
        self.created_with.append(kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        return object()


class _FakeRequestsResponse:
    def __init__(self, payload=None, status_code=200, text="", headers=None):
        self._payload = payload or {}
        self.headers = headers or {}
        self.status_code = status_code
        self.text = text
        self.content = b"%PDF"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload

    def iter_content(self, chunk_size):
        yield b"%PDF"


class _FakeKernel:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class _FakeRequestsSession:
    calls = []

    def __init__(self):
        self.trust_env = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs, self.trust_env))
        payload = (
            {"models": [{"name": "qwen3.6:latest"}]}
            if url.endswith("/api/tags")
            else None
        )
        return _FakeRequestsResponse(payload)

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs, self.trust_env))
        return _FakeRequestsResponse({"success": True})


class TestApiBackendClient(unittest.TestCase):
    def setUp(self):
        _FakeClient.created_with.clear()
        _FakeClient.requests.clear()
        _FakeRequestsSession.calls.clear()

    def test_backend_requests_do_not_use_environment_proxies(self):
        with patch.object(gui_fasthtml.httpx, "Client", _FakeClient):
            response = gui_fasthtml._request_api_backend(
                "GET", "http://172.27.74.16:7861/health", timeout=5
            )

        self.assertIsNotNone(response)
        self.assertEqual(_FakeClient.created_with, [{"trust_env": False}])
        self.assertEqual(
            _FakeClient.requests,
            [("GET", "http://172.27.74.16:7861/health", {"timeout": 5})],
        )

    def test_requests_calls_do_not_use_environment_proxies(self):
        with (
            patch.object(gui_fasthtml, "API_BASE_URL", ""),
            patch.object(gui_fasthtml.requests, "Session", _FakeRequestsSession),
        ):
            self.assertEqual(
                gui_fasthtml._ollama_model_options("172.27.74.16:11434"),
                ["qwen3.6:latest"],
            )
            with patch.object(gui_fasthtml, "server_key", "secret", create=True):
                self.assertTrue(gui_fasthtml.verify_recaptcha("token"))
            with TemporaryDirectory() as temp_dir:
                output = gui_fasthtml.download_with_limit(
                    "http://172.27.74.49/document.pdf", Path(temp_dir), None
                )
                self.assertEqual(output.read_bytes(), b"%PDF")

        self.assertEqual([call[3] for call in _FakeRequestsSession.calls], [False] * 3)
        self.assertEqual(
            [call[0] for call in _FakeRequestsSession.calls],
            ["GET", "POST", "GET"],
        )

    def test_download_with_limit_uses_content_disposition_filename(self):
        class _DownloadSession(_FakeRequestsSession):
            def get(self, url, **kwargs):
                self.calls.append(("GET", url, kwargs, self.trust_env))
                return _FakeRequestsResponse(
                    headers={
                        "Content-Disposition": 'attachment; filename="paper.txt"'
                    }
                )

        with (
            patch.object(gui_fasthtml, "API_BASE_URL", ""),
            patch.object(gui_fasthtml.requests, "Session", _DownloadSession),
        ):
            with TemporaryDirectory() as temp_dir:
                output = gui_fasthtml.download_with_limit(
                    "http://172.27.74.49/document.pdf", Path(temp_dir), None
                )

        self.assertEqual(output.name, "paper.pdf")

    def test_api_mode_queries_ollama_models_through_backend(self):
        response = _FakeRequestsResponse({"models": ["qwen3.6:latest"]})
        with (
            patch.object(gui_fasthtml, "API_BASE_URL", "http://172.27.74.49:7861"),
            patch.object(
                gui_fasthtml, "_request_api_backend", return_value=response
            ) as request_api,
            patch.object(
                gui_fasthtml.requests,
                "Session",
                side_effect=AssertionError("Ollama must be reached by the API backend"),
            ),
        ):
            self.assertEqual(
                gui_fasthtml._ollama_model_options("127.0.0.1:11434"),
                ["qwen3.6:latest"],
            )

        request_api.assert_called_once_with(
            "GET",
            "http://172.27.74.49:7861/v1/ollama/models",
            params={"host": "http://127.0.0.1:11434"},
            timeout=2,
        )

    def test_api_ollama_models_are_queried_from_backend_without_proxies(self):
        with patch.object(api_server._requests, "Session", _FakeRequestsSession):
            self.assertEqual(
                api_server.ollama_models("172.27.74.49:11434"),
                {"models": ["qwen3.6:latest"]},
            )

        self.assertEqual(
            _FakeRequestsSession.calls,
            [("GET", "http://172.27.74.49:11434/api/tags", {"timeout": 2}, False)],
        )

    def test_api_translation_preserves_selected_ollama_host(self):
        envs = api_server._resolve_translator_envs(
            "Ollama", ["172.27.74.49:11434", "qwen3.6:latest", "", ""]
        )

        self.assertEqual(envs["OLLAMA_HOST"], "http://172.27.74.49:11434")
        self.assertEqual(envs["OLLAMA_MODEL"], "qwen3.6:latest")
        self.assertEqual(envs["OLLAMA_TIMEOUT"], "300")
        self.assertEqual(envs["OLLAMA_THINK"], "false")

    def test_llm_usage_formatter_reports_zero_requests(self):
        self.assertEqual(
            api_server._format_llm_usage({"requests": 0, "prompt_eval_count": 0}),
            "requests: 0",
        )

    def test_api_ollama_validation_rejects_missing_model(self):
        envs = {
            "OLLAMA_HOST": "http://172.27.74.49:11434",
            "OLLAMA_MODEL": "missing:latest",
        }
        with (
            patch.object(
                api_server, "_ollama_model_names", return_value=["qwen3.6:latest"]
            ),
            self.assertRaises(api_server.HTTPException) as caught,
        ):
            api_server._validate_ollama_envs(envs)

        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("missing:latest", caught.exception.detail)

    def test_gui_ollama_fields_use_client_environment_host(self):
        with (
            patch.dict(os.environ, {"OLLAMA_HOST": "172.27.74.49:11434"}),
            patch.object(gui_fasthtml, "_ollama_model_options", return_value=["gemma2"]),
        ):
            fields = gui_fasthtml._service_env_fields("Ollama").__html__()

        self.assertIn('value="http://172.27.74.49:11434"', fields)
        self.assertIn('hx_swap="outerHTML"', fields)

    def test_gui_openailiked_fields_do_not_render_ollama_widgets(self):
        fields = gui_fasthtml._service_env_fields("OpenAI-liked").__html__()

        self.assertIn("OPENAILIKED_BASE_URL", fields)
        self.assertNotIn("OLLAMA_HOST", fields)
        self.assertNotIn('id="ollama-model-field"', fields)

    def test_gui_openailiked_fields_use_openailiked_env_defaults(self):
        with patch.dict(
            os.environ,
            {
                "OPENAILIKED_BASE_URL": "https://api.example.com/v1",
                "OPENAILIKED_API_KEY": "env-key",
                "OPENAILIKED_MODEL": "env-model",
            },
            clear=False,
        ):
            fields = gui_fasthtml._service_env_fields("OpenAI-liked").__html__()

        self.assertIn('value="https://api.example.com/v1"', fields)
        self.assertIn('value="env-key"', fields)
        self.assertIn('value="env-model"', fields)

    def test_gui_openailiked_fields_use_dashscope_env_defaults(self):
        with patch.dict(
            os.environ,
            {
                "OPENAILIKED_BASE_URL": "",
                "OPENAILIKED_API_KEY": "",
                "OPENAILIKED_MODEL": "",
                "DASHSCOPE_API_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "DASHSCOPE_API_KEY": "dashscope-key",
                "DASHSCOPE_API_MODEL_FLASH": "qwen-plus-latest",
            },
            clear=False,
        ):
            fields = gui_fasthtml._service_env_fields("OpenAI-liked").__html__()

        self.assertIn(
            'value="https://dashscope.aliyuncs.com/compatible-mode/v1"',
            fields,
        )
        self.assertIn('value="dashscope-key"', fields)
        self.assertIn('value="qwen-plus-latest"', fields)

    def test_index_service_switch_replaces_env_fields_node(self):
        with patch.object(gui_fasthtml, "_authorized", return_value=None):
            app = gui_fasthtml.setup_gui()
            index_html = app.routes["/"](None).__html__()

        self.assertIn('hx_target="#env-fields"', index_html)
        self.assertIn('hx_swap="outerHTML"', index_html)

    def test_environment_url_overrides_persisted_api_url(self):
        with (
            patch.dict(
                os.environ,
                {"PDF2ZH_API_BASE_URL": "http://172.27.74.16:7861/"},
            ),
            patch.object(gui_fasthtml.ConfigManager, "get") as config_get,
        ):
            self.assertEqual(
                gui_fasthtml._configured_api_base_url(),
                "http://172.27.74.16:7861",
            )

        config_get.assert_not_called()

    def test_empty_environment_url_disables_persisted_api_url(self):
        with (
            patch.dict(os.environ, {"PDF2ZH_API_BASE_URL": ""}),
            patch.object(
                gui_fasthtml.ConfigManager,
                "get",
                return_value="http://old-api-host:7861",
            ) as config_get,
        ):
            self.assertEqual(gui_fasthtml._configured_api_base_url(), "")

        config_get.assert_not_called()

    def test_api_rejects_unavailable_precise_mode_before_starting_job(self):
        with (
            patch.object(
                api_server.KernelRegistry,
                "get",
                return_value=_FakeKernel(False),
            ),
            self.assertRaises(api_server.HTTPException) as caught,
        ):
            api_server._validate_mode_choice("precise")

        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("pdf2zh-setup-precise", caught.exception.detail)

    def test_gui_client_surfaces_api_detail_on_submit_failure(self):
        session_id = "session-api-error"
        gui_fasthtml.translation_jobs[session_id] = {"status": "running"}
        detail = "Kernel 'precise' is not available on the API server."
        responses = [
            _FakeRequestsResponse({"status": "ok"}),
            _FakeRequestsResponse(
                {"detail": detail},
                status_code=400,
                text="bad request",
            ),
        ]

        def fake_request(*args, **kwargs):
            return responses.pop(0)

        params = {
            "file_type": "Link",
            "file_input": "",
            "link_input": "http://example.test/document.pdf",
            "service": "Google",
            "lang_from": "English",
            "lang_to": "Simplified Chinese",
            "page_range": "All",
            "page_input": "",
            "prompt": "",
            "threads": 4,
            "skip_subset_fonts": False,
            "ignore_cache": False,
            "vfont": "",
            "mode_choice": "precise",
            "env_0": "",
            "env_1": "",
            "env_2": "",
            "env_3": "",
        }

        try:
            with (
                patch.object(gui_fasthtml, "API_BASE_URL", "http://api.test"),
                patch.object(
                    gui_fasthtml,
                    "_request_api_backend",
                    side_effect=fake_request,
                ),
            ):
                gui_fasthtml._run_api_translation_job(session_id, params)

            self.assertEqual(
                gui_fasthtml.translation_jobs[session_id]["status"],
                "error",
            )
            self.assertEqual(
                gui_fasthtml.translation_jobs[session_id]["message"],
                detail,
            )
            self.assertEqual(gui_fasthtml.translation_jobs[session_id]["error"], detail)
        finally:
            gui_fasthtml.translation_jobs.pop(session_id, None)


if __name__ == "__main__":
    unittest.main()
