"""Structured logging: console + rotating project file + in-memory ring buffer for the UI."""
from __future__ import annotations

import collections
import logging
import logging.handlers
from pathlib import Path

_RING: collections.deque[str] = collections.deque(maxlen=400)
_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


class _RingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            _RING.append(self.format(record))
        except Exception:
            pass


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level))
    if not any(isinstance(h, _RingHandler) for h in root.handlers):
        ring = _RingHandler()
        ring.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        root.addHandler(ring)
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_FMT))
        root.addHandler(console)
    for noisy in ("urllib3", "httpx", "httpcore", "matplotlib", "numba", "PIL", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def attach_project_log(logs_dir: Path) -> logging.Handler:
    """Add a rotating file handler for one project; returns it so callers can detach later."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / "clipper.log"
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.handlers.RotatingFileHandler) and Path(h.baseFilename) == path.resolve():
            return h
    fh = logging.handlers.RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FMT))
    logging.getLogger().addHandler(fh)
    return fh


def detach_handler(handler: logging.Handler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()


def recent_log_lines(n: int = 200) -> list[str]:
    return list(_RING)[-n:]
