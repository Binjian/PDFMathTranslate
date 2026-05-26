import os
import unittest
from unittest.mock import patch

from pdf2zh import gui_fasthtml


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


class TestApiBackendClient(unittest.TestCase):
    def setUp(self):
        _FakeClient.created_with.clear()
        _FakeClient.requests.clear()

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
