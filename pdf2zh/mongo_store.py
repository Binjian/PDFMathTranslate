"""Optional MongoDB persistence for API translation-job artefacts.

The API server records one document per job (keyed by ``job_id``) capturing
status, service, client IP, output file names, LLM usage and timings. This is a
durable, queryable complement to the in-memory ``_jobs`` dict and the Markdown
job log — it does **not** store the PDF binaries (those stay on disk).

The store is *optional*: it is only active when a connection URI is configured.
If the URI is unset, ``pymongo`` is not installed, or the server is unreachable,
the store silently disables itself and the API server keeps working with its
existing file-based logging.

Environment variables:
    PDF2ZH_API_MONGODB_URI         MongoDB connection URI. When unset, the store
                                   is disabled. (Falls back to MONGODB_URI.)
    PDF2ZH_API_MONGODB_DB          Database name (default: pdf2zh)
    PDF2ZH_API_MONGODB_COLLECTION  Collection name (default: job_artifacts)
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


class JobArtifactStore:
    """Thin, fail-safe wrapper around a MongoDB collection of job artefacts.

    Every public method is a no-op when the store is disabled, and all Mongo
    interaction is wrapped so a backend hiccup can never break translation.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._client = None
        self._collection = None
        self._enabled = False
        self._init_failed = False
        self._uri = (
            ConfigManager.get("PDF2ZH_API_MONGODB_URI")
            or ConfigManager.get("MONGODB_URI")
            or ""
        ).strip()
        self._db_name = ConfigManager.get("PDF2ZH_API_MONGODB_DB", "pdf2zh")
        self._collection_name = ConfigManager.get(
            "PDF2ZH_API_MONGODB_COLLECTION", "job_artifacts"
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

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
                self._collection = client[self._db_name][self._collection_name]
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
                    "MongoDB artefact store disabled (%s); falling back to "
                    "file-based job logging.",
                    exc,
                )
                return False

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

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None
        self._collection = None
        self._enabled = False
