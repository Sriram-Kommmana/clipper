"""Frame sampling (OpenCV, one frame in memory at a time) and basic visual-suitability checks."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
from pydantic import BaseModel, Field

from src.analysis.scene_detection import Scene

log = logging.getLogger(__name__)

DARK_MEAN = 18.0        # mean gray level below which a frame counts as (nearly) black
BLUR_VAR = 20.0         # Laplacian variance below which a frame counts as very blurry/flat


class VisualCheck(BaseModel):
    score: float = Field(100, ge=0, le=100)
    samples: int = 0
    mean_brightness: float = 0.0
    sharpness: float = 0.0
    dark_ratio: float = 0.0
    warnings: list[str] = Field(default_factory=list)


def sample_times(start: float, end: float, n: int) -> list[float]:
    """n timestamps evenly inside (start, end), avoiding the exact edges."""
    n = max(1, n)
    return [start + (end - start) * (i + 0.5) / n for i in range(n)]


def read_frames(video: Path, times: list[float], max_width: int = 640) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (t, BGR frame) for each timestamp; skips unreadable positions. Frames are not retained."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            log.warning("OpenCV could not open %s", video)
            return
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            if w > max_width:
                frame = cv2.resize(frame, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
            yield t, frame
    finally:
        cap.release()


def frame_stats(frame: np.ndarray) -> tuple[float, float]:
    """(mean gray level 0-255, Laplacian variance) of a frame."""
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(gray.mean()), float(cv2.Laplacian(gray, cv2.CV_64F).var())


def score_stats(brightness: list[float], sharpness: list[float]) -> VisualCheck:
    """Pure scoring from per-frame stats (unit-tested)."""
    if not brightness:
        return VisualCheck(score=50, warnings=["Could not read any frames to check visuals."])
    b, s = np.asarray(brightness), np.asarray(sharpness)
    dark_ratio = float((b < DARK_MEAN).mean())
    blurry_ratio = float((s < BLUR_VAR).mean())
    score = 100.0 - 70.0 * dark_ratio - 30.0 * blurry_ratio
    vc = VisualCheck(score=max(0.0, score), samples=len(b), mean_brightness=float(b.mean()),
                     sharpness=float(s.mean()), dark_ratio=dark_ratio)
    if dark_ratio >= 0.5:
        vc.warnings.append("Mostly black/dark frames.")
    if blurry_ratio >= 0.6:
        vc.warnings.append("Frames are very flat/blurry (little visual detail).")
    return vc


def visual_suitability(video: Path, start: float, end: float, n: int = 4) -> VisualCheck:
    bright, sharp = [], []
    for _, frame in read_frames(video, sample_times(start, end, n), max_width=320):
        b, s = frame_stats(frame)
        bright.append(b)
        sharp.append(s)
    return score_stats(bright, sharp)


def extract_thumbnails(video: Path, scenes: list[Scene], out_dir: Path, max_n: int) -> list[tuple[float, str]]:
    """Representative (mid-scene) JPEG per scene; evenly subsampled to at most max_n."""
    import cv2

    if max_n <= 0 or not scenes:
        return []
    chosen = scenes if len(scenes) <= max_n else [scenes[int(i * len(scenes) / max_n)] for i in range(max_n)]
    out_dir.mkdir(parents=True, exist_ok=True)
    result: list[tuple[float, str]] = []
    mids = [(s.start + s.end) / 2 for s in chosen]
    for i, (t, frame) in enumerate(read_frames(video, mids, max_width=320)):
        name = f"scene_{i:04d}_{t:.1f}.jpg"
        if cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 80]):
            result.append((round(t, 2), name))
    return result
