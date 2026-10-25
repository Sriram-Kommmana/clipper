"""Keyed stage cache: a cache entry is valid only if its key matches the one computed from inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.core.models import SCHEMA_VERSION
from src.storage.project_store import atomic_write_json, read_json


def make_key(*parts: Any) -> str:
    """Stable short hash of arbitrary JSON-able parts (source fingerprint, config hash, upstream keys...)."""
    blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


class StageCache:
    def __init__(self, directory: Path):
        self.dir = Path(directory)

    def path(self, name: str) -> Path:
        return self.dir / f"{name}.json"

    def load(self, name: str, key: str) -> Any | None:
        """Return cached data iff present, readable, schema-compatible and key-matching; else None."""
        p = self.path(name)
        if not p.exists():
            return None
        try:
            entry = read_json(p)
        except (OSError, ValueError):
            return None  # corrupt cache == miss (never crash a run because of a cache file)
        if not isinstance(entry, dict) or entry.get("key") != key or entry.get("schema") != SCHEMA_VERSION:
            return None
        return entry.get("data")

    def load_any(self, name: str) -> Any | None:
        """Cached data regardless of key (for read-only consumers such as the review UI)."""
        try:
            entry = read_json(self.path(name))
        except (OSError, ValueError):
            return None
        return entry.get("data") if isinstance(entry, dict) else None

    def save(self, name: str, key: str, data: Any) -> None:
        atomic_write_json(self.path(name), {"key": key, "schema": SCHEMA_VERSION, "data": data})

    def invalidate(self, *names: str) -> None:
        for n in names:
            try:
                self.path(n).unlink()
            except FileNotFoundError:
                pass
