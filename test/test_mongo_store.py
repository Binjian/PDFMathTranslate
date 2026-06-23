"""Tests for the optional MongoDB job-artefact store (pdf2zh.mongo_store).

These tests never touch a real MongoDB server. A fake ``pymongo`` module is
injected into ``sys.modules`` so the store's lazy ``from pymongo import
MongoClient`` resolves to a mock, letting us assert connection behaviour,
upsert payloads and the fail-safe (disabled) code paths in isolation.
"""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pdf2zh import mongo_store
from pdf2zh.mongo_store import JobArtifactStore


def _fake_config(values: dict):
    """Return a stand-in for ConfigManager.get backed by ``values``."""

    def _get(key, default=None):
        return values.get(key, default)

    return _get


def _install_fake_pymongo(client: MagicMock):
    """Context manager patching sys.modules['pymongo'] with a fake module."""
    module = types.ModuleType("pymongo")
    module.MongoClient = MagicMock(return_value=client)
    return patch.dict(sys.modules, {"pymongo": module})


def _make_client() -> MagicMock:
    """A MongoClient mock whose ping succeeds and supports db[coll] indexing."""
    client = MagicMock(name="MongoClient")
    client.admin.command.return_value = {"ok": 1}
    return client


def _store_with_config(values: dict) -> JobArtifactStore:
    with patch.object(mongo_store.ConfigManager, "get", _fake_config(values)):
        return JobArtifactStore()


class TestStoreConfiguration(unittest.TestCase):
    """Construction reads connection settings from ConfigManager."""

    def test_disabled_when_no_uri(self):
        store = _store_with_config({})
        self.assertFalse(store.enabled)
        # Defaults still resolved even when disabled.
        self.assertEqual(store._db_name, "pdf2zh")
        self.assertEqual(store._collection_name, "job_artifacts")

    def test_primary_uri_used(self):
        store = _store_with_config(
            {"PDF2ZH_API_MONGODB_URI": "  mongodb://primary:27017  "}
        )
        self.assertEqual(store._uri, "mongodb://primary:27017")

    def test_falls_back_to_generic_mongodb_uri(self):
        store = _store_with_config({"MONGODB_URI": "mongodb://fallback:27017"})
        self.assertEqual(store._uri, "mongodb://fallback:27017")

    def test_primary_uri_takes_precedence(self):
        store = _store_with_config(
            {
                "PDF2ZH_API_MONGODB_URI": "mongodb://primary",
                "MONGODB_URI": "mongodb://fallback",
            }
        )
        self.assertEqual(store._uri, "mongodb://primary")

    def test_custom_db_and_collection(self):
        store = _store_with_config(
            {
                "PDF2ZH_API_MONGODB_DB": "mydb",
                "PDF2ZH_API_MONGODB_COLLECTION": "mycoll",
            }
        )
        self.assertEqual(store._db_name, "mydb")
        self.assertEqual(store._collection_name, "mycoll")


class TestDisabledStoreIsNoOp(unittest.TestCase):
    """With no URI, every method is a safe no-op and never imports pymongo."""

    def setUp(self):
        self.store = _store_with_config({})

    def test_record_does_not_connect(self):
        fake = MagicMock()
        with _install_fake_pymongo(fake):
            self.store.record("job-1", {"status": "done"}, {"status": "done"})
            sys.modules["pymongo"].MongoClient.assert_not_called()
        self.assertFalse(self.store.enabled)

    def test_get_returns_none(self):
        self.assertIsNone(self.store.get("job-1"))

    def test_close_is_safe(self):
        self.store.close()  # must not raise
        self.assertFalse(self.store.enabled)


class TestConnection(unittest.TestCase):
    """Lazy connection establishment and failure handling."""

    def test_connects_and_pings(self):
        client = _make_client()
        store = _store_with_config(
            {
                "PDF2ZH_API_MONGODB_URI": "mongodb://db:27017",
                "PDF2ZH_API_MONGODB_DB": "mydb",
                "PDF2ZH_API_MONGODB_COLLECTION": "mycoll",
            }
        )
        with _install_fake_pymongo(client):
            self.assertTrue(store._ensure_connection())
            ctor = sys.modules["pymongo"].MongoClient
            ctor.assert_called_once()
            # Connection URI passed through and server selection forced via ping.
            self.assertEqual(ctor.call_args.args[0], "mongodb://db:27017")
            client.admin.command.assert_called_once_with("ping")
        self.assertTrue(store.enabled)
        # Collection resolved from configured db / collection names.
        client.__getitem__.assert_called_with("mydb")
        client.__getitem__.return_value.__getitem__.assert_called_with("mycoll")

    def test_connection_failure_disables_store(self):
        client = _make_client()
        client.admin.command.side_effect = RuntimeError("unreachable")
        store = _store_with_config({"PDF2ZH_API_MONGODB_URI": "mongodb://db"})
        with _install_fake_pymongo(client):
            self.assertFalse(store._ensure_connection())
            self.assertFalse(store.enabled)
            # A second attempt must not retry the connection.
            self.assertFalse(store._ensure_connection())
            sys.modules["pymongo"].MongoClient.assert_called_once()

    def test_missing_pymongo_disables_store(self):
        store = _store_with_config({"PDF2ZH_API_MONGODB_URI": "mongodb://db"})
        # Simulate pymongo not being installed.
        with patch.dict(sys.modules, {"pymongo": None}):
            self.assertFalse(store._ensure_connection())
        self.assertFalse(store.enabled)

    def test_connection_established_only_once(self):
        client = _make_client()
        store = _store_with_config({"PDF2ZH_API_MONGODB_URI": "mongodb://db"})
        with _install_fake_pymongo(client):
            self.assertTrue(store._ensure_connection())
            self.assertTrue(store._ensure_connection())
            sys.modules["pymongo"].MongoClient.assert_called_once()


class TestRecord(unittest.TestCase):
    """record() upserts a snapshot and appends lifecycle events."""

    def _connected_store(self, client):
        store = _store_with_config({"PDF2ZH_API_MONGODB_URI": "mongodb://db"})
        with _install_fake_pymongo(client):
            store._ensure_connection()
        return store

    def test_upsert_filter_and_setoninsert(self):
        client = _make_client()
        store = self._connected_store(client)
        collection = client.__getitem__.return_value.__getitem__.return_value

        store.record("job-42", {"status": "done", "service": "Google"})

        collection.update_one.assert_called_once()
        flt, update = collection.update_one.call_args.args[:2]
        self.assertEqual(flt, {"_id": "job-42"})
        self.assertTrue(collection.update_one.call_args.kwargs["upsert"])
        self.assertEqual(update["$set"]["status"], "done")
        self.assertEqual(update["$set"]["service"], "Google")
        self.assertEqual(update["$set"]["job_id"], "job-42")
        self.assertIn("updated_at", update["$set"])
        self.assertIn("created_at", update["$setOnInsert"])

    def test_event_pushed_when_provided(self):
        client = _make_client()
        store = self._connected_store(client)
        collection = client.__getitem__.return_value.__getitem__.return_value

        event = {"timestamp": "2026-06-23", "status": "done"}
        store.record("job-1", {"status": "done"}, event)

        update = collection.update_one.call_args.args[1]
        self.assertEqual(update["$push"], {"events": event})

    def test_no_push_without_event(self):
        client = _make_client()
        store = self._connected_store(client)
        collection = client.__getitem__.return_value.__getitem__.return_value

        store.record("job-1", {"status": "running"})

        update = collection.update_one.call_args.args[1]
        self.assertNotIn("$push", update)

    def test_events_key_stripped_from_snapshot(self):
        client = _make_client()
        store = self._connected_store(client)
        collection = client.__getitem__.return_value.__getitem__.return_value

        store.record("job-1", {"status": "done", "events": ["stale"]})

        update = collection.update_one.call_args.args[1]
        self.assertNotIn("events", update["$set"])

    def test_record_swallows_backend_errors(self):
        client = _make_client()
        store = self._connected_store(client)
        collection = client.__getitem__.return_value.__getitem__.return_value
        collection.update_one.side_effect = RuntimeError("write failed")

        # Must not propagate — translation lifecycle should never break.
        store.record("job-1", {"status": "done"})


class TestGet(unittest.TestCase):
    def test_returns_document(self):
        client = _make_client()
        store = _store_with_config({"PDF2ZH_API_MONGODB_URI": "mongodb://db"})
        with _install_fake_pymongo(client):
            store._ensure_connection()
        collection = client.__getitem__.return_value.__getitem__.return_value
        collection.find_one.return_value = {"_id": "job-1", "status": "done"}

        result = store.get("job-1")
        collection.find_one.assert_called_once_with({"_id": "job-1"})
        self.assertEqual(result, {"_id": "job-1", "status": "done"})

    def test_get_swallows_backend_errors(self):
        client = _make_client()
        store = _store_with_config({"PDF2ZH_API_MONGODB_URI": "mongodb://db"})
        with _install_fake_pymongo(client):
            store._ensure_connection()
        collection = client.__getitem__.return_value.__getitem__.return_value
        collection.find_one.side_effect = RuntimeError("read failed")

        self.assertIsNone(store.get("job-1"))


class TestClose(unittest.TestCase):
    def test_close_resets_state(self):
        client = _make_client()
        store = _store_with_config({"PDF2ZH_API_MONGODB_URI": "mongodb://db"})
        with _install_fake_pymongo(client):
            store._ensure_connection()
        self.assertTrue(store.enabled)

        store.close()
        client.close.assert_called_once()
        self.assertFalse(store.enabled)
        self.assertIsNone(store._client)
        self.assertIsNone(store._collection)


class _FakeGridOut:
    def __init__(self, oid, filename, data, fields):
        self._id = oid
        self.filename = filename
        self._data = data
        self.fields = fields

    def read(self):
        return self._data


class _FakeCursor(list):
    def sort(self, field, direction=1):
        ordered = list(reversed(self)) if direction == -1 else list(self)
        return _FakeCursor(ordered)

    def limit(self, n):
        return _FakeCursor(self[:n])


class _FakeGridFS:
    """Minimal in-memory stand-in for gridfs.GridFS used in tests."""

    def __init__(self):
        self.files: list[_FakeGridOut] = []
        self.deleted: list[int] = []

    def put(self, data, filename=None, **fields):
        oid = len(self.files) + 1
        self.files.append(_FakeGridOut(oid, filename, data, fields))
        return oid

    def find(self, query):
        def _match(go: _FakeGridOut) -> bool:
            for key, value in query.items():
                if key == "filename":
                    if go.filename != value:
                        return False
                elif go.fields.get(key) != value:
                    return False
            return True

        return _FakeCursor([go for go in self.files if _match(go)])

    def delete(self, oid):
        self.deleted.append(oid)
        self.files = [go for go in self.files if go._id != oid]


def _install_fake_gridfs(fs: _FakeGridFS):
    module = types.ModuleType("gridfs")
    module.GridFS = MagicMock(return_value=fs)
    return patch.dict(sys.modules, {"gridfs": module})


def _connected_store():
    store = _store_with_config({"PDF2ZH_API_MONGODB_URI": "mongodb://db"})
    with _install_fake_pymongo(_make_client()):
        store._ensure_connection()
    return store


class TestAvailable(unittest.TestCase):
    def test_available_false_when_disabled(self):
        self.assertFalse(_store_with_config({}).available())

    def test_available_true_when_connected(self):
        store = _store_with_config({"PDF2ZH_API_MONGODB_URI": "mongodb://db"})
        with _install_fake_pymongo(_make_client()):
            self.assertTrue(store.available())


class TestGridFSFiles(unittest.TestCase):
    def test_put_disabled_returns_none(self):
        store = _store_with_config({})
        self.assertIsNone(store.put_file(b"data", "x.pdf"))

    def test_get_disabled_returns_none(self):
        store = _store_with_config({})
        self.assertIsNone(store.get_file_by_name("x.pdf"))

    def test_put_and_get_latest_version_by_name(self):
        store = _connected_store()
        fs = _FakeGridFS()
        with _install_fake_gridfs(fs):
            store.put_file(b"old", "doc.pdf", job_id="j1", variant="mono")
            store.put_file(b"new", "doc.pdf", job_id="j2", variant="mono")
            result = store.get_file_by_name("doc.pdf")
        self.assertEqual(result, (b"new", "doc.pdf"))

    def test_get_file_by_metadata_query(self):
        store = _connected_store()
        fs = _FakeGridFS()
        with _install_fake_gridfs(fs):
            store.put_file(b"mono-bytes", "a-mono.pdf", job_id="j1", variant="mono")
            store.put_file(b"dual-bytes", "a-dual.pdf", job_id="j1", variant="dual")
            result = store.get_file({"job_id": "j1", "variant": "dual"})
        self.assertEqual(result, (b"dual-bytes", "a-dual.pdf"))

    def test_get_missing_returns_none(self):
        store = _connected_store()
        with _install_fake_gridfs(_FakeGridFS()):
            self.assertIsNone(store.get_file({"job_id": "nope", "variant": "mono"}))

    def test_delete_files_returns_names_and_removes(self):
        store = _connected_store()
        fs = _FakeGridFS()
        with _install_fake_gridfs(fs):
            store.put_file(b"m", "a-mono.pdf", job_id="j1", variant="mono")
            store.put_file(b"d", "a-dual.pdf", job_id="j1", variant="dual")
            store.put_file(b"o", "b-mono.pdf", job_id="j2", variant="mono")
            removed = store.delete_files({"job_id": "j1"})
            remaining = store.get_file({"job_id": "j2", "variant": "mono"})
        self.assertEqual(sorted(removed), ["a-dual.pdf", "a-mono.pdf"])
        self.assertEqual(remaining, (b"o", "b-mono.pdf"))

    def test_gridfs_constructed_with_bucket_prefix(self):
        store = _store_with_config(
            {
                "PDF2ZH_API_MONGODB_URI": "mongodb://db",
                "PDF2ZH_API_MONGODB_COLLECTION": "jobs",
            }
        )
        with _install_fake_pymongo(_make_client()):
            store._ensure_connection()
        fs = _FakeGridFS()
        with _install_fake_gridfs(fs):
            store.put_file(b"x", "x.pdf")
            sys.modules["gridfs"].GridFS.assert_called_once()
            self.assertEqual(
                sys.modules["gridfs"].GridFS.call_args.kwargs["collection"], "jobs_fs"
            )


class TestListJobs(unittest.TestCase):
    def test_list_jobs_disabled_returns_empty(self):
        self.assertEqual(_store_with_config({}).list_jobs(), [])

    def test_list_jobs_sorted_query(self):
        client = _make_client()
        store = _store_with_config({"PDF2ZH_API_MONGODB_URI": "mongodb://db"})
        with _install_fake_pymongo(client):
            store._ensure_connection()
        collection = client.__getitem__.return_value.__getitem__.return_value
        docs = [{"_id": "j2"}, {"_id": "j1"}]
        collection.find.return_value.sort.return_value.limit.return_value = docs

        result = store.list_jobs(limit=10)
        collection.find.return_value.sort.assert_called_once_with("updated_at", -1)
        collection.find.return_value.sort.return_value.limit.assert_called_once_with(10)
        self.assertEqual(result, docs)


if __name__ == "__main__":
    unittest.main()
