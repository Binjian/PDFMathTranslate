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
        self.assertIn('hx-swap="outerHTML"', fields)

    def test_gui_openailiked_fields_do_not_render_ollama_widgets(self):
        fields = gui_fasthtml._service_env_fields("OpenAI-liked").__html__()

        self.assertIn("OPENAILIKED_MODEL", fields)
        self.assertNotIn("OLLAMA_HOST", fields)
        self.assertNotIn('id="ollama-model-field"', fields)

    def test_gui_openailiked_fields_hide_backend_only_credentials(self):
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

        self.assertNotIn("OPENAILIKED_BASE_URL", fields)
        self.assertNotIn("OPENAILIKED_API_KEY", fields)
        self.assertNotIn("https://api.example.com/v1", fields)
        self.assertNotIn("env-key", fields)
        self.assertIn('value="env-model"', fields)
        # env_0/env_1 stay as empty hidden inputs so indices remain aligned.
        self.assertIn('name="env_0" value=""', fields)
        self.assertIn('name="env_1" value=""', fields)

    def test_gui_openailiked_fields_never_expose_dashscope_credentials(self):
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

        self.assertNotIn(
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            fields,
        )
        self.assertNotIn("dashscope-key", fields)
        self.assertIn('value="qwen-plus-latest"', fields)

    def test_api_openailiked_envs_resolved_from_server_environment(self):
        submitted = [
            "https://attacker.example.com/v1",
            "client-supplied-key",
            "client-model",
        ]
        with patch.dict(
            os.environ,
            {
                "OPENAILIKED_BASE_URL": "",
                "OPENAILIKED_API_KEY": "",
                "DASHSCOPE_API_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "DASHSCOPE_API_KEY": "dashscope-key",
            },
            clear=False,
        ):
            envs = api_server._resolve_translator_envs("OpenAI-liked", submitted)

        self.assertEqual(
            envs["OPENAILIKED_BASE_URL"],
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.assertEqual(envs["OPENAILIKED_API_KEY"], "dashscope-key")
        self.assertEqual(envs["OPENAILIKED_MODEL"], "client-model")

    def test_index_service_switch_replaces_env_fields_node(self):
        from starlette.testclient import TestClient

        app = gui_fasthtml.create_app()
        client = TestClient(app)
        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('hx-target="#env-fields"', response.text)
        self.assertIn('hx-swap="outerHTML"', response.text)

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


class TestServiceTranslateEndpoint(unittest.TestCase):
    """POST /v1/service/translate maps ``service`` to kernel mode + model."""

    def _client(self):
        from starlette.testclient import TestClient

        return TestClient(api_server.app)

    def _capture_submit(self):
        """Patch the shared submission helper, capturing its delegated kwargs."""
        captured: dict = {}

        async def fake_submit(request, **kwargs):
            captured.update(kwargs)
            return {"job_id": "test-job"}

        return captured, patch.object(api_server, "_submit_translate_job", fake_submit)

    def test_model_map_constant(self):
        self.assertEqual(
            api_server._SERVICE_OPENAILIKED_MODEL,
            {"fast": "qwen3.6-flash", "precise": "qwen3.6-plus"},
        )

    def test_fast_selects_flash_model_and_freezes_service(self):
        captured, patcher = self._capture_submit()
        with patcher:
            response = self._client().post(
                "/v1/service/translate", data={"service": "fast", "link": "x"}
            )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json(), {"job_id": "test-job"})
        self.assertEqual(captured["service"], "OpenAI-liked")
        self.assertEqual(captured["mode_choice"], "fast")
        self.assertEqual(captured["env_overrides"], {"OPENAILIKED_MODEL": "qwen3.6-flash"})

    def test_precise_selects_plus_model(self):
        captured, patcher = self._capture_submit()
        with patcher:
            response = self._client().post(
                "/v1/service/translate", data={"service": "precise"}
            )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(captured["mode_choice"], "precise")
        self.assertEqual(captured["env_overrides"], {"OPENAILIKED_MODEL": "qwen3.6-plus"})

    def test_service_value_is_case_insensitive(self):
        captured, patcher = self._capture_submit()
        with patcher:
            response = self._client().post(
                "/v1/service/translate", data={"service": "  PRECISE  "}
            )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(captured["mode_choice"], "precise")
        self.assertEqual(captured["env_overrides"], {"OPENAILIKED_MODEL": "qwen3.6-plus"})

    def test_default_service_is_fast(self):
        captured, patcher = self._capture_submit()
        with patcher:
            response = self._client().post("/v1/service/translate", data={})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(captured["mode_choice"], "fast")
        self.assertEqual(captured["env_overrides"], {"OPENAILIKED_MODEL": "qwen3.6-flash"})

    def test_unknown_service_rejected_before_submitting(self):
        called = {"submit": False}

        async def fail_submit(request, **kwargs):
            called["submit"] = True
            return {}

        with patch.object(api_server, "_submit_translate_job", fail_submit):
            response = self._client().post(
                "/v1/service/translate", data={"service": "medium"}
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("medium", response.json()["detail"])
        self.assertFalse(called["submit"])

    def test_other_fields_forwarded(self):
        captured, patcher = self._capture_submit()
        with patcher:
            response = self._client().post(
                "/v1/service/translate",
                data={
                    "service": "fast",
                    "lang_from": "German",
                    "lang_to": "English",
                    "page_range": "First",
                    "threads": "8",
                    "vfont": "myfont",
                },
            )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(captured["lang_from"], "German")
        self.assertEqual(captured["lang_to"], "English")
        self.assertEqual(captured["page_range"], "First")
        self.assertEqual(captured["threads"], 8)
        self.assertEqual(captured["vfont"], "myfont")


class TestTranslateEndpointDelegation(unittest.TestCase):
    """POST /v1/translate still forwards to the shared helper after refactor."""

    def _client(self):
        from starlette.testclient import TestClient

        return TestClient(api_server.app)

    def test_service_and_mode_passed_through_without_overrides(self):
        captured: dict = {}

        async def fake_submit(request, **kwargs):
            captured.update(kwargs)
            return {"job_id": "test-job"}

        with patch.object(api_server, "_submit_translate_job", fake_submit):
            response = self._client().post(
                "/v1/translate",
                data={"service": "Google", "mode_choice": "precise", "link": "x"},
            )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(captured["service"], "Google")
        self.assertEqual(captured["mode_choice"], "precise")
        # The plain endpoint never forces an env override.
        self.assertIsNone(captured.get("env_overrides"))


class _FakeStore:
    """In-memory stand-in for gui_fasthtml._artifact_store."""

    def __init__(self, available=True, files=None, jobs=None):
        self._available = available
        self._files = files or {}
        self._jobs = jobs or []
        self.put_calls = []

    def available(self):
        return self._available

    def get_file_by_name(self, name):
        if name in self._files:
            return self._files[name], name
        return None

    def list_jobs(self, limit=500):
        return list(self._jobs)

    def put_file(self, data, filename, **fields):
        self.put_calls.append((filename, fields))
        return "id"


class TestGuiMongoServing(unittest.TestCase):
    """GUI retrieval routes read PDFs and the job log from MongoDB only."""

    def _client(self):
        from starlette.testclient import TestClient

        return TestClient(gui_fasthtml.create_app())

    def test_file_route_serves_blob_from_store(self):
        store = _FakeStore(files={"doc-mono.pdf": b"%PDF-mono"})
        with patch.object(gui_fasthtml, "_artifact_store", store):
            response = self._client().get("/file", params={"name": "doc-mono.pdf"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"%PDF-mono")
        self.assertIn("application/pdf", response.headers["content-type"])
        self.assertIn("inline", response.headers["content-disposition"])

    def test_file_route_404_when_missing(self):
        with patch.object(gui_fasthtml, "_artifact_store", _FakeStore(files={})):
            response = self._client().get("/file", params={"name": "missing.pdf"})
        self.assertEqual(response.status_code, 404)

    def test_file_route_503_when_store_unavailable(self):
        with patch.object(gui_fasthtml, "_artifact_store", _FakeStore(available=False)):
            response = self._client().get("/file", params={"name": "doc.pdf"})
        self.assertEqual(response.status_code, 503)

    def test_download_route_uses_translated_attachment_name(self):
        store = _FakeStore(files={"uuid-paper-mono.pdf": b"%PDF"})
        with patch.object(gui_fasthtml, "_artifact_store", store):
            response = self._client().get(
                "/download", params={"name": "uuid-paper-mono.pdf", "variant": "mono"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn("attachment", response.headers["content-disposition"])
        self.assertIn(
            gui_fasthtml._translated_download_name("uuid-paper-mono.pdf", "mono"),
            response.headers["content-disposition"],
        )

    def test_download_route_rejects_bad_variant(self):
        with patch.object(gui_fasthtml, "_artifact_store", _FakeStore()):
            response = self._client().get(
                "/download", params={"name": "doc.pdf", "variant": "triple"}
            )
        self.assertEqual(response.status_code, 404)

    def test_job_log_renders_rows_from_store(self):
        jobs = [
            {
                "job_id": "job-1",
                "updated_at": 1_700_000_000.0,
                "status": "done",
                "client_ip": "10.0.0.1",
                "service": "OpenAI-liked",
                "files": ["a-mono.pdf", "a-dual.pdf"],
                "elapsed_seconds": 3661,
                "llm_requests": 5,
                "llm_prompt_tokens": 12,
                "llm_completion_tokens": 34,
                "llm_total_tokens": 46,
            }
        ]
        with patch.object(gui_fasthtml, "_artifact_store", _FakeStore(jobs=jobs)):
            response = self._client().get("/job-log")
        self.assertEqual(response.status_code, 200)
        self.assertIn("job-1", response.text)
        self.assertIn("OpenAI-liked", response.text)
        self.assertIn("a-mono.pdf", response.text)
        self.assertIn("1h 01m 01s", response.text)

    def test_job_log_empty_when_store_has_no_jobs(self):
        with patch.object(gui_fasthtml, "_artifact_store", _FakeStore(jobs=[])):
            response = self._client().get("/job-log")
        self.assertEqual(response.status_code, 200)
        self.assertIn("No job log entries yet", response.text)


class TestGuiJobLogTableHelper(unittest.TestCase):
    def test_format_elapsed_hms(self):
        self.assertEqual(gui_fasthtml._format_elapsed_hms(None), "0")
        self.assertEqual(gui_fasthtml._format_elapsed_hms(0), "0")
        self.assertEqual(gui_fasthtml._format_elapsed_hms(5), "5s")
        self.assertEqual(gui_fasthtml._format_elapsed_hms(125), "2m 05s")
        self.assertEqual(gui_fasthtml._format_elapsed_hms(3661), "1h 01m 01s")


class TestQuickServiceTab(unittest.TestCase):
    """The Quick tab drives POST /v1/service/translate and hides parameters."""

    def _client(self):
        from starlette.testclient import TestClient

        return TestClient(gui_fasthtml.create_app())

    def test_quick_tab_link_present_in_nav(self):
        response = self._client().get("/")
        self.assertIn(">Quick<", response.text)

    def test_service_tab_hides_translator_credentials_and_mode(self):
        response = self._client().get("/service")
        self.assertEqual(response.status_code, 200)
        text = response.text
        # Exposes only the fast/precise selector and posts to the new endpoint.
        self.assertIn('action="/service-translate"', text)
        self.assertIn('value="fast"', text)
        self.assertIn('value="precise"', text)
        self.assertIn("tab-active", text)
        # Hidden parameters are not rendered as graphical elements.
        self.assertNotIn('name="env_0"', text)
        self.assertNotIn('name="mode_choice"', text)
        self.assertNotIn("OpenAI-liked", text)
        self.assertNotIn("OPENAILIKED_MODEL", text)

    def test_submit_delegates_with_service_variant(self):
        captured = {}

        def fake_runner(session_id, params):
            captured["params"] = params

        def make_thread(target=None, args=(), daemon=None):
            class _T:
                def start(self_):
                    target(*args)

            return _T()

        with (
            patch.object(gui_fasthtml, "API_BASE_URL", "http://api.test"),
            patch.object(gui_fasthtml.threading, "Thread", side_effect=make_thread),
            patch.object(gui_fasthtml, "_run_api_translation_job", fake_runner),
        ):
            response = self._client().post(
                "/service-translate",
                data={
                    "service": "precise",
                    "file_type": "Link",
                    "link_input": "http://example.test/doc.pdf",
                    "lang_to": "German",
                },
            )
        self.assertEqual(response.status_code, 200)
        params = captured["params"]
        self.assertEqual(params["api_variant"], "service")
        self.assertEqual(params["service"], "precise")
        self.assertEqual(params["mode_choice"], "precise")
        self.assertEqual(params["lang_to"], "German")
        self.assertFalse(any(k.startswith("env_") for k in params))

    def test_submit_without_api_base_url_errors_clearly(self):
        with patch.object(gui_fasthtml, "API_BASE_URL", ""):
            response = self._client().post(
                "/service-translate",
                data={
                    "service": "fast",
                    "file_type": "Link",
                    "link_input": "http://example.test/doc.pdf",
                },
            )
        self.assertEqual(response.status_code, 200)
        # The progress page is returned; the job is already marked errored.
        self.assertIn("translation-progress", response.text)

    def test_api_runner_posts_to_service_endpoint_without_hidden_fields(self):
        session_id = "session-quick"
        gui_fasthtml.translation_jobs[session_id] = {"status": "running"}
        calls = []

        def fake_request(method, url, **kwargs):
            calls.append((method, url, kwargs))
            if url.endswith("/health"):
                return _FakeRequestsResponse({"status": "ok"})
            # Stop after capturing the submit by returning a non-202 status.
            return _FakeRequestsResponse({"detail": "stop"}, status_code=400, text="bad")

        params = {
            "api_variant": "service",
            "file_type": "Link",
            "file_input": "",
            "link_input": "http://example.test/doc.pdf",
            "service": "fast",
            "lang_from": "English",
            "lang_to": "Simplified Chinese",
            "page_range": "All",
            "page_input": "",
            "prompt": "",
            "threads": "4",
            "skip_subset_fonts": False,
            "ignore_cache": False,
            "vfont": "",
            "mode_choice": "fast",
            "session_id": session_id,
        }
        try:
            with (
                patch.object(gui_fasthtml, "API_BASE_URL", "http://api.test"),
                patch.object(gui_fasthtml, "_request_api_backend", side_effect=fake_request),
            ):
                gui_fasthtml._run_api_translation_job(session_id, params)

            submit = next(c for c in calls if c[0] == "POST")
            self.assertTrue(submit[1].endswith("/v1/service/translate"))
            form_data = submit[2]["data"]
            self.assertEqual(form_data["service"], "fast")
            self.assertNotIn("mode_choice", form_data)
            self.assertFalse(any(k.startswith("env_") for k in form_data))
        finally:
            gui_fasthtml.translation_jobs.pop(session_id, None)


class TestDownloadBothEndpoint(unittest.TestCase):
    """GET /v1/translate/{job_id}/both returns mono + dual, unzipped by default."""

    class _Store:
        def __init__(self, available=True, files=None):
            self._available = available
            self._files = files or {}

        def available(self):
            return self._available

        def get_file(self, query):
            return self._files.get((query.get("job_id"), query.get("variant")))

    def _client(self):
        from starlette.testclient import TestClient

        return TestClient(api_server.app)

    def _both(self):
        return {
            ("job-1", "mono"): (b"%PDF-mono", "doc-mono.pdf"),
            ("job-1", "dual"): (b"%PDF-dual", "doc-dual.pdf"),
        }

    def _parse_multipart(self, response):
        import email

        raw = (
            b"Content-Type: "
            + response.headers["content-type"].encode()
            + b"\r\n\r\n"
            + response.content
        )
        return email.message_from_bytes(raw).get_payload()

    def test_default_returns_both_unzipped_as_multipart(self):
        with patch.object(api_server, "_artifact_store", self._Store(files=self._both())):
            response = self._client().get("/v1/translate/job-1/both")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("multipart/mixed"))
        parts = self._parse_multipart(response)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].get_content_type(), "application/pdf")
        self.assertEqual(parts[0].get_filename(), "doc-mono.pdf")
        self.assertEqual(parts[0].get_payload(decode=True), b"%PDF-mono")
        self.assertEqual(parts[1].get_filename(), "doc-dual.pdf")
        self.assertEqual(parts[1].get_payload(decode=True), b"%PDF-dual")

    def test_zip_query_returns_archive(self):
        import io
        import zipfile

        with patch.object(api_server, "_artifact_store", self._Store(files=self._both())):
            response = self._client().get("/v1/translate/job-1/both", params={"zip": "true"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/zip")
        self.assertIn('filename="job-1.zip"', response.headers["content-disposition"])
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        self.assertEqual(sorted(archive.namelist()), ["doc-dual.pdf", "doc-mono.pdf"])
        self.assertEqual(archive.read("doc-mono.pdf"), b"%PDF-mono")
        self.assertEqual(archive.read("doc-dual.pdf"), b"%PDF-dual")

    def test_both_does_not_shadow_variant_route(self):
        with patch.object(api_server, "_artifact_store", self._Store(files=self._both())):
            response = self._client().get("/v1/translate/job-1/mono")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"%PDF-mono")

    def test_404_when_a_variant_missing(self):
        partial = {("job-2", "mono"): (b"x", "m.pdf")}
        with patch.object(api_server, "_artifact_store", self._Store(files=partial)):
            response = self._client().get("/v1/translate/job-2/both")
        self.assertEqual(response.status_code, 404)

    def test_503_when_store_unavailable(self):
        with patch.object(api_server, "_artifact_store", self._Store(available=False)):
            response = self._client().get("/v1/translate/job-1/both")
        self.assertEqual(response.status_code, 503)

    def test_409_while_running(self):
        api_server._jobs["job-run"] = {"status": "running"}
        try:
            with patch.object(api_server, "_artifact_store", self._Store(files=self._both())):
                response = self._client().get("/v1/translate/job-run/both")
        finally:
            api_server._jobs.pop("job-run", None)
        self.assertEqual(response.status_code, 409)


class TestFrontendMetricsEndpoint(unittest.TestCase):
    def _client(self):
        from starlette.testclient import TestClient

        return TestClient(api_server.app)

    def test_returns_404_when_file_does_not_exist(self):
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "frontend_metrics.md"
            with patch.object(api_server, "FRONTEND_METRICS", missing):
                response = self._client().get("/v1/metrics/frontend")
        self.assertEqual(response.status_code, 404)

    def test_returns_404_when_file_is_empty(self):
        with TemporaryDirectory() as tmp:
            empty = Path(tmp) / "frontend_metrics.md"
            empty.write_text("", encoding="utf-8")
            with patch.object(api_server, "FRONTEND_METRICS", empty):
                response = self._client().get("/v1/metrics/frontend")
        self.assertEqual(response.status_code, 404)

    def test_returns_file_content_with_markdown_media_type(self):
        content = (
            "| timestamp | llm_duration | generated_tokens | response |\n"
            "|---|---:|---:|---|\n"
            "| 2026-06-12 10:00:00 | 5m 02s | 4,098 | {\"status\": \"done\"} |\n"
        )
        with TemporaryDirectory() as tmp:
            metrics_file = Path(tmp) / "frontend_metrics.md"
            metrics_file.write_text(content, encoding="utf-8")
            with patch.object(api_server, "FRONTEND_METRICS", metrics_file):
                response = self._client().get("/v1/metrics/frontend")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/markdown", response.headers["content-type"])
        self.assertEqual(response.text, content)


if __name__ == "__main__":
    unittest.main()
