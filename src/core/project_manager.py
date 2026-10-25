"""Creating, opening and listing projects."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from src.core.exceptions import ProjectError
from src.core.models import ProjectState, VideoInfo
from src.storage.project_store import ProjectStore


def slugify(text: str, limit: int = 40) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return (text or "video")[:limit].strip("-") or "video"


def project_id_for(info: VideoInfo) -> str:
    return f"{slugify(Path(info.filename).stem)}-{info.fingerprint[:8]}"


def create_or_open_project(info: VideoInfo, projects_dir: Path) -> ProjectStore:
    """Idempotent: the same file content always maps to the same project folder (so work resumes)."""
    if not info.fingerprint:
        raise ProjectError("VideoInfo has no fingerprint; call inspect_video(..., fingerprint=True).")
    pid = project_id_for(info)
    store = ProjectStore(Path(projects_dir) / pid)
    if store.state_path.exists():
        state = store.load_state()
        # The file may have been moved/renamed: refresh path metadata but keep all work.
        state.source = info
        store.save_state()
        return store
    store.create(ProjectState(project_id=pid, source=info))
    return store


def open_project(path: Path) -> ProjectStore:
    store = ProjectStore(Path(path))
    store.load_state()
    return store


def list_projects(projects_dir: Path) -> list[tuple[str, Path]]:
    """[(label, folder)] for every readable project, newest first."""
    out: list[tuple[float, str, Path]] = []
    root = Path(projects_dir)
    if not root.is_dir():
        return []
    for d in root.iterdir():
        sp = d / "project.json"
        if sp.is_file():
            try:
                out.append((sp.stat().st_mtime, d.name, d))
            except OSError:
                continue
    out.sort(reverse=True)
    return [(name, d) for _, name, d in out]
