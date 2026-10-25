"""Input path validation and content fingerprinting."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from src.core.exceptions import InvalidMediaError

SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".mkv")
_CHUNK = 1 << 20  # 1 MiB


def validate_input_path(raw: str | os.PathLike[str], extensions: tuple[str, ...] = SUPPORTED_EXTENSIONS) -> Path:
    """Return an absolute, resolved path to a readable regular file with a supported extension.

    The absolute form matters for safety: a relative name beginning with "-" could otherwise be parsed
    by FFmpeg as an option.
    """
    text = os.fspath(raw)
    if not text or not text.strip():
        raise InvalidMediaError("No video file was given.")
    if "\x00" in text:
        raise InvalidMediaError("The path contains an invalid character.")
    p = Path(text.strip().strip('"')).expanduser()
    try:
        p = p.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise InvalidMediaError(f"Cannot resolve path: {text}") from exc
    if not p.exists():
        raise InvalidMediaError(f"File not found: {p}", recovery="Check the path (drag the file into the box to paste it).")
    if not p.is_file():
        raise InvalidMediaError(f"Not a file: {p}")
    if p.suffix.lower() not in extensions:
        raise InvalidMediaError(
            f"Unsupported file type '{p.suffix}'. Supported: {', '.join(extensions)}",
            recovery="Convert the file to MP4 first, e.g. with FFmpeg: ffmpeg -i input.avi output.mp4",
        )
    try:
        size = p.stat().st_size
        with open(p, "rb") as f:
            f.read(1)
    except OSError as exc:
        raise InvalidMediaError(f"Cannot read {p}: {exc}", recovery="Check permissions and that the file is not locked by another program.") from exc
    if size == 0:
        raise InvalidMediaError(f"The file is empty: {p}")
    return p


def compute_fingerprint(path: Path) -> str:
    """Fast content fingerprint: size + SHA-1 of head, middle and tail megabytes.

    Independent of filename and mtime, so moving/copying a file keeps its caches valid, while a
    different or re-encoded file gets a new fingerprint.
    """
    size = path.stat().st_size
    h = hashlib.sha1()
    h.update(str(size).encode())
    offsets = sorted({0, max(0, size // 2 - _CHUNK // 2), max(0, size - _CHUNK)})
    with open(path, "rb") as f:
        for off in offsets:
            f.seek(off)
            h.update(f.read(_CHUNK))
    return h.hexdigest()[:20]
