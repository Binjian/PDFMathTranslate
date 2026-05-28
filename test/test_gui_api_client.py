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
    def __init__(self, payload=None):
        self._payload = payload or {}
        self.headers = {}

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


if __name__ == "__main__":
    unittest.main()
