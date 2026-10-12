"""Content-addressed file cache.

Namespaces used by the pipeline:

* ``downloads``  — raw documents, keyed by SHA-256 of the bytes
* ``parsed``     — parsed page structure, keyed by content hash + parser version
* ``chunks``     — chunk lists, keyed by content hash + chunker version
* ``screening``  — Jev/heuristic decisions, keyed by chunk hash + model + question version
* ``extraction`` — LLM extraction output, keyed by chunk hash + model + prompt version
* ``search``     — search results, keyed by provider + query (with TTL)

Everything is plain JSON on disk so cached artifacts can be inspected when debugging (spec §5).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


class FileCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, namespace: str, key: str, suffix: str = ".json") -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return self.root / namespace / safe[:2] / f"{safe}{suffix}"

    def _atomic_write(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # -- JSON ---------------------------------------------------------------------------------

    def get_json(self, namespace: str, key: str, ttl_seconds: float | None = None) -> Any | None:
        path = self.path(namespace, key)
        if not path.is_file():
            return None
        if ttl_seconds is not None and time.time() - path.stat().st_mtime > ttl_seconds:
            return None
        try:
            return json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def put_json(self, namespace: str, key: str, value: Any) -> Path:
        path = self.path(namespace, key)
        self._atomic_write(path, json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))
        return path

    # -- bytes --------------------------------------------------------------------------------

    def get_bytes(self, namespace: str, key: str, suffix: str = "") -> bytes | None:
        path = self.path(namespace, key, suffix)
        return path.read_bytes() if path.is_file() else None

    def put_bytes(self, namespace: str, key: str, data: bytes, suffix: str = "") -> Path:
        path = self.path(namespace, key, suffix)
        if not path.is_file():
            self._atomic_write(path, data)
        return path
