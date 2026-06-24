"""MongoDB persistence for translation-job artefacts.

MongoDB is the source of truth for finished jobs. For each job the store keeps:

* one **metadata** document per ``job_id`` (status, service, client IP, output
  file names, LLM usage and timings) in the configured collection, and
* the **PDF binaries** (source / mono / dual) in GridFS, since PDFs routinely
  exceed BSON's 16 MB document limit.

The on-disk job folder produced by the translation kernels is treated as
transient scratch: once a job finishes its PDFs are ingested here and all
retrieval (downloads, previews, the job log) reads from MongoDB.

Writes degrade gracefully — a backend hiccup is logged and never breaks a
translation. Retrieval, by contrast, *requires* MongoDB: callers use
``available()`` to decide whether to serve a request or return 503.

Environment variables:
    PDF2ZH_API_MONGODB_URI         MongoDB connection URI (falls back to
                                   MONGODB_URI, then to a local default of
                                   ``mongodb://localhost:27017``).
    PDF2ZH_API_MONGODB_DB          Database name (default: pdf2zh)
    PDF2ZH_API_MONGODB_COLLECTION  Metadata collection name (default:
                                   job_artifacts). GridFS buckets use the
                                   ``<collection>_fs`` prefix.
"""
from __future__ import annotations

import logging
import threading
import time

from pdf2zh.config import ConfigManager

logger = logging.getLogger(__name__)

# Keep server-selection snappy so an unreachable Mongo fails fast instead of
# stalling job lifecycle callbacks.
_SERVER_SELECTION_TIMEOUT_MS = 2000

# Default MongoDB URI when neither PDF2ZH_API_MONGODB_URI nor MONGODB_URI is set.
DEFAULT_MONGODB_URI = "mongodb://localhost:27017"


class JobArtifactStore:
    """Fail-safe wrapper around a MongoDB metadata collection + GridFS bucket.

    Metadata writes are no-ops when the store is disabled; binary retrieval
    callers gate on :meth:`available` and surface 503 when MongoDB is down.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._client = None
        self._db = None
        self._collection = None
        self._fs = None
        self._enabled = False
        self._init_failed = False
        self._uri = (
            ConfigManager.get("PDF2ZH_API_MONGODB_URI")
            or ConfigManager.get("MONGODB_URI")
            or DEFAULT_MONGODB_URI
        ).strip()
        self._db_name = ConfigManager.get("PDF2ZH_API_MONGODB_DB", "pdf2zh")
        self._collection_name = ConfigManager.get(
            "PDF2ZH_API_MONGODB_COLLECTION", "job_artifacts"
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def available(self) -> bool:
        """Public predicate: True when MongoDB can serve a retrieval request."""
        return self._ensure_connection()

    def _ensure_connection(self) -> bool:
        """Lazily connect on first use; returns True when the store is usable."""
        if self._enabled:
            return True
        if self._init_failed or not self._uri:
            return False
        with self._lock:
            if self._enabled:
                return True
            if self._init_failed:
                return False
            try:
                from pymongo import MongoClient

                client = MongoClient(
                    self._uri,
                    serverSelectionTimeoutMS=_SERVER_SELECTION_TIMEOUT_MS,
                    appname="pdf2zh-api",
                )
                # Force server selection so an unreachable host fails here.
                client.admin.command("ping")
                self._client = client
                self._db = client[self._db_name]
                self._collection = self._db[self._collection_name]
                self._enabled = True
                logger.info(
                    "MongoDB artefact store connected (db=%s, collection=%s)",
                    self._db_name,
                    self._collection_name,
                )
                return True
            except Exception as exc:
                self._init_failed = True
                logger.warning(
                    "MongoDB artefact store disabled (%s); retrieval from "
                    "MongoDB will be unavailable.",
                    exc,
                )
                return False

    def _gridfs(self):
        """Return a GridFS handle, or None when the store is unavailable."""
        if not self._ensure_connection():
            return None
        if self._fs is None:
            with self._lock:
                if self._fs is None:
                    from gridfs import GridFS

                    self._fs = GridFS(self._db, collection=f"{self._collection_name}_fs")
        return self._fs

    # ── Metadata ──────────────────────────────────────────────────────────

    def record(self, job_id: str, document: dict, event: dict | None = None) -> None:
        """Upsert the latest job snapshot and append a lifecycle event.

        ``document`` is the current job snapshot (status, service, files, …).
        ``event`` is an optional point-in-time entry pushed onto an ``events``
        array, giving a full audit trail per job.
        """
        if not self._ensure_connection():
            return
        now = time.time()
        snapshot = {k: v for k, v in document.items() if k != "events"}
        snapshot["job_id"] = job_id
        snapshot["updated_at"] = now
        update: dict = {
            "$set": snapshot,
            "$setOnInsert": {"created_at": now},
        }
        if event is not None:
            update["$push"] = {"events": event}
        try:
            self._collection.update_one({"_id": job_id}, update, upsert=True)
        except Exception:
            logger.exception("Unable to persist job %s to MongoDB", job_id)

    def get(self, job_id: str) -> dict | None:
        """Return the stored document for ``job_id`` (None if absent/disabled)."""
        if not self._ensure_connection():
            return None
        try:
            return self._collection.find_one({"_id": job_id})
        except Exception:
            logger.exception("Unable to read job %s from MongoDB", job_id)
            return None

    def list_jobs(self, limit: int = 500) -> list[dict]:
        """Return job metadata documents, most recently updated first."""
        if not self._ensure_connection():
            return []
        try:
            cursor = (
                self._collection.find().sort("updated_at", -1).limit(int(limit))
            )
            return list(cursor)
        except Exception:
            logger.exception("Unable to list jobs from MongoDB")
            return []

    # ── Binary artefacts (GridFS) ─────────────────────────────────────────

    def put_file(self, data: bytes, filename: str, **fields) -> str | None:
        """Store a PDF blob in GridFS, returning its id (None when disabled).

        Extra keyword fields (e.g. ``job_id``, ``variant``, ``session_id``) are
        stored as top-level fields on the GridFS file document so blobs can be
        queried back by job or variant.
        """
        fs = self._gridfs()
        if fs is None:
            return None
        try:
            return str(fs.put(data, filename=filename, **fields))
        except Exception:
            logger.exception("Unable to store file %s in GridFS", filename)
            return None

    def get_file(self, query: dict) -> tuple[bytes, str] | None:
        """Return ``(data, filename)`` for the newest blob matching ``query``."""
        fs = self._gridfs()
        if fs is None:
            return None
        try:
            for grid_out in fs.find(query).sort("uploadDate", -1).limit(1):
                return grid_out.read(), grid_out.filename
            return None
        except Exception:
            logger.exception("Unable to read file %s from GridFS", query)
            return None

    def get_file_by_name(self, filename: str) -> tuple[bytes, str] | None:
        """Return ``(data, filename)`` for the newest blob with ``filename``."""
        return self.get_file({"filename": filename})

    def delete_files(self, query: dict) -> list[str]:
        """Delete all blobs matching ``query``; return the removed file names."""
        fs = self._gridfs()
        if fs is None:
            return []
        removed: list[str] = []
        try:
            for grid_out in fs.find(query):
                removed.append(grid_out.filename)
                fs.delete(grid_out._id)
        except Exception:
            logger.exception("Unable to delete files %s from GridFS", query)
        return removed

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None
        self._db = None
        self._collection = None
        self._fs = None
        self._enabled = False
