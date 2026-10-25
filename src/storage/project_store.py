"""Atomic JSON IO and the per-project store (state + candidates + folders)."""
from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.core.exceptions import ProjectError
from src.core.models import Candidate, ProjectState, StageState, now_iso


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` so readers never see a partial file (tmp file in same dir + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class ProjectStore:
    """Owns one project folder. All state mutations go through here and are lock-protected."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.RLock()
        self._state: ProjectState | None = None

    # ---- paths ---------------------------------------------------------------------------------
    @property
    def state_path(self) -> Path:
        return self.root / "project.json"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    @property
    def renders_dir(self) -> Path:
        return self.root / "renders"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def candidates_path(self) -> Path:
        return self.cache_dir / "candidates.json"

    def ensure_dirs(self) -> None:
        for d in (self.cache_dir, self.work_dir, self.renders_dir, self.exports_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---- state ---------------------------------------------------------------------------------
    def create(self, state: ProjectState) -> None:
        with self._lock:
            self.ensure_dirs()
            self._state = state
            self.save_state()

    def load_state(self) -> ProjectState:
        with self._lock:
            try:
                self._state = ProjectState.model_validate(read_json(self.state_path))
            except FileNotFoundError as exc:
                raise ProjectError(f"No project at {self.root}") from exc
            except (ValueError, ValidationError) as exc:
                raise ProjectError(f"Project file {self.state_path} is unreadable: {exc}") from exc
            self.ensure_dirs()
            return self._state

    @property
    def state(self) -> ProjectState:
        with self._lock:
            if self._state is None:
                return self.load_state()
            return self._state

    def save_state(self) -> None:
        with self._lock:
            assert self._state is not None
            self._state.updated_at = now_iso()
            atomic_write_json(self.state_path, self._state.model_dump(mode="json"))

    def stage(self, name: str) -> StageState:
        with self._lock:
            return self.state.stages.setdefault(name, StageState())

    def update_stage(self, name: str, **changes: Any) -> StageState:
        with self._lock:
            st = self.stage(name)
            for k, v in changes.items():
                setattr(st, k, v)
            self.save_state()
            return st

    def add_warning(self, text: str) -> None:
        with self._lock:
            if text not in self.state.warnings:
                self.state.warnings.append(text)
                self.save_state()

    def add_error(self, text: str) -> None:
        with self._lock:
            self.state.errors.append(f"{now_iso()} {text}")
            self.state.errors = self.state.errors[-50:]
            self.save_state()

    def mutate(self, fn) -> None:
        """Run `fn(state)` under the lock and persist."""
        with self._lock:
            fn(self.state)
            self.save_state()

    # ---- candidates ----------------------------------------------------------------------------
    def load_candidates(self) -> list[Candidate]:
        with self._lock:
            if not self.candidates_path.exists():
                return []
            try:
                raw = read_json(self.candidates_path)
                return [Candidate.model_validate(c) for c in raw.get("candidates", [])]
            except (ValueError, ValidationError) as exc:
                raise ProjectError(f"candidates.json is corrupt: {exc}") from exc

    def save_candidates(self, candidates: list[Candidate], extra: dict[str, Any] | None = None) -> None:
        with self._lock:
            payload: dict[str, Any] = {"updated_at": now_iso(), "candidates": [c.model_dump(mode="json") for c in candidates]}
            if extra:
                payload.update(extra)
            atomic_write_json(self.candidates_path, payload)

    def get_candidate(self, cid: str) -> Candidate:
        for c in self.load_candidates():
            if c.id == cid:
                return c
        raise ProjectError(f"Unknown candidate id {cid!r}")

    def update_candidate(self, cid: str, **changes: Any) -> Candidate:
        """Atomically modify one candidate (used by the review UI)."""
        with self._lock:
            cands = self.load_candidates()
            for i, c in enumerate(cands):
                if c.id == cid:
                    data = c.model_dump()
                    data.update(changes)
                    data["updated_at"] = now_iso()
                    cands[i] = Candidate.model_validate(data)
                    self.save_candidates(cands)
                    return cands[i]
        raise ProjectError(f"Unknown candidate id {cid!r}")
