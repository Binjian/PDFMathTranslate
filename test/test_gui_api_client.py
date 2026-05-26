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


if __name__ == "__main__":
    unittest.main()
