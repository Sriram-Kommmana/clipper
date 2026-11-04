"""Optional subject tracking for smart crop, behind an interchangeable detector interface.

Built-in detectors use only OpenCV (Apache-2.0) - no extra downloads, no AGPL code. Anything with a
`detect(frame_bgr) -> list[Box]` method (e.g. an ONNX person detector you supply) can be plugged in.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from config import SmartCropConfig

SmoothingParams = SmartCropConfig
from src.analysis.frame_extraction import read_frames

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Box:
    cx: float   # normalised centre x, 0-1
    cy: float
    w: float
    h: float
    conf: float = 1.0

    @property
    def area(self) -> float:
        return self.w * self.h


class SubjectDetector(Protocol):
    name: str

    def detect(self, frame_bgr: np.ndarray) -> list[Box]: ...


class FaceDetector:
    """OpenCV Haar frontal-face cascade (ships inside the opencv wheel)."""

    name = "opencv-haar-face"

    def __init__(self) -> None:
        import cv2

        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(path))
        if self._cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade at {path}")

    def detect(self, frame_bgr: np.ndarray) -> list[Box]:
        import cv2

        h, w = frame_bgr.shape[:2]
        gray = cv2.equalizeHist(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
        found = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(max(24, h // 14), max(24, h // 14)))
        return [Box((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h) for x, y, bw, bh in found]


class PersonDetector:
    """OpenCV HOG + linear SVM pedestrian detector (slower, coarser than a neural detector)."""

    name = "opencv-hog-person"

    def __init__(self) -> None:
        import cv2

        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame_bgr: np.ndarray) -> list[Box]:
        import cv2

        h, w = frame_bgr.shape[:2]
        scale = 480 / w if w > 480 else 1.0
        small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale))) if scale != 1.0 else frame_bgr
        rects, weights = self._hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
        sh, sw = small.shape[:2]
        return [Box((x + bw / 2) / sw, (y + bh / 2) / sh, bw / sw, bh / sh, float(wt)) for (x, y, bw, bh), wt in zip(rects, np.ravel(weights))]


def make_detector(name: str) -> SubjectDetector | None:
    if name == "none":
        return None
    try:
        return FaceDetector() if name == "face" else PersonDetector()
    except Exception as exc:
        log.warning("Subject detector '%s' unavailable (%s) - using centre crop", name, exc)
        return None


@dataclass
class CropPlan:
    keypoints: list[tuple[float, float]]      # (clip-relative t, crop left edge in source px)
    detection_ratio: float = 0.0
    fallback: bool = False
    notes: list[str] = field(default_factory=list)


def pick_subject(boxes: list[Box], prev_cx: float | None) -> Box | None:
    """Largest box wins; among comparably large ones prefer the one nearest the previous subject (no hopping)."""
    if not boxes:
        return None
    biggest = max(b.area * b.conf for b in boxes)
    contenders = [b for b in boxes if b.area * b.conf >= 0.6 * biggest]
    if prev_cx is None:
        return max(contenders, key=lambda b: b.area * b.conf)
    return min(contenders, key=lambda b: abs(b.cx - prev_cx))


CUT_RAMP_S = 0.04   # a cut is followed by a jump this fast (about one frame); never a pan through the edit


def _smooth_shot(samples: list[tuple[float, float | None]], src_w: int, crop_w: int, cfg: SmoothingParams) -> list[tuple[float, float]]:
    """Dead-banded, gliding crop path for ONE shot. Needs at least one detection."""
    cx = np.array([np.nan if c is None else c * src_w for _, c in samples], dtype=float)
    idx = np.arange(len(cx))
    good = ~np.isnan(cx)
    cx = np.interp(idx, idx[good], cx[good])           # forward/back-fill gaps so the path is defined everywhere
    if len(cx) >= 3:                                   # median-filter single-sample outliers
        padded = np.pad(cx, 1, mode="edge")
        cx = np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0)
    dead = cfg.deadband_pct * src_w
    cur = cx[0]
    pts: list[tuple[float, float]] = []
    for (t, _), target in zip(samples, cx):
        if abs(target - cur) > dead:
            edge = target - np.sign(target - cur) * dead      # glide toward the dead-band edge, not the centre
            cur += (edge - cur) * max(cfg.smoothing, 0.05)
        left = float(np.clip(cur - crop_w / 2, 0, src_w - crop_w))
        pts.append((float(t), left))
    return pts


def _drop_collinear(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Drop points that lie (within 2 px) on the line between their neighbours."""
    if len(pts) <= 2:
        return pts
    thin = [pts[0]]
    for i in range(1, len(pts) - 1):
        (t0, x0), (t1, x1), (t2, x2) = thin[-1], pts[i], pts[i + 1]
        interp = x0 + (x2 - x0) * (t1 - t0) / max(t2 - t0, 1e-6)
        if abs(interp - x1) > 2.0:
            thin.append(pts[i])
    thin.append(pts[-1])
    return thin


def _cap(pts: list[tuple[float, float]], limit: int) -> list[tuple[float, float]]:
    if len(pts) <= limit:
        return pts
    step = (len(pts) - 1) / (limit - 1)
    return [pts[round(i * step)] for i in range(limit)]


def split_shots(samples: list[tuple[float, float | None]], cuts: list[float] | None, duration: float | None = None
                ) -> list[tuple[float, float, list[tuple[float, float | None]]]]:
    """[(shot_start, shot_end, samples inside it)] for clip-relative `cuts`; no cuts = one shot."""
    end = duration if duration is not None else (samples[-1][0] if samples else 0.0)
    edges = [0.0] + sorted(c for c in (cuts or []) if 0.0 < c < end) + [end]
    shots = []
    for a, b in zip(edges, edges[1:]):
        inside = [(t, c) for t, c in samples if a <= t < b or (b == end and t == end)]
        shots.append((a, b, inside))
    return shots


def plan_crop_path(samples: list[tuple[float, float | None]], src_w: int, crop_w: int, cfg: SmartCropConfig,
                   cuts: list[float] | None = None, duration: float | None = None) -> CropPlan:
    """Turn raw subject-centre samples [(t, cx_norm or None)] into a smooth, dead-banded crop path.

    * too few detections -> centre crop (fallback)
    * the crop holds still until the subject leaves a dead-band, then glides toward it (smoothing factor)
    * `cuts` (clip-relative scene cuts): every shot is planned on its own and the crop JUMPS at a cut instead of panning
      through the edit, which is the most obvious sign of automatic reframing. A shot where the subject was not found
      gets a centre crop rather than inheriting the previous camera's position.
    * keypoints are thinned to cfg.max_keypoints (per shot, within a global cap)
    """
    centre_x = (src_w - crop_w) / 2
    hits = [(t, c) for t, c in samples if c is not None]
    ratio = len(hits) / max(1, len(samples))
    if ratio < cfg.min_detection_ratio or not hits:
        return CropPlan([(0.0, centre_x)], ratio, True, [f"Subject found in only {ratio:.0%} of samples - used a centre crop."])
    shots = split_shots(samples, cuts, duration)
    notes: list[str] = []
    out: list[tuple[float, float]] = []
    budget = max(3, cfg.max_keypoints // max(1, len(shots)))
    for i, (a, b, inside) in enumerate(shots):
        found = sum(1 for _, c in inside if c is not None)
        if inside and found / len(inside) >= cfg.min_detection_ratio:
            pts = _cap(_drop_collinear(_smooth_shot(inside, src_w, crop_w, cfg)), budget)
        else:
            pts = [(a, centre_x)]
            if len(shots) > 1:
                notes.append(f"Shot {i + 1} ({a:.1f}-{b:.1f}s): subject not found - centre crop.")
        pts = [(a, pts[0][1])] + [p for p in pts if p[0] > a]       # a shot's path starts at its first frame
        if i < len(shots) - 1 and b - a > 2 * CUT_RAMP_S:
            hold, final = b - CUT_RAMP_S, pts[-1][1]                  # stay put right up to the cut, then jump
            pts = [p for p in pts if p[0] < hold] + [(hold, final)]
        out.extend(pts)
    clean: list[tuple[float, float]] = []                             # strictly increasing times (a shot can be shorter than the ramp)
    for t, x in out:
        if not clean or t > clean[-1][0]:
            clean.append((t, x))
    if len(shots) > 1:
        notes.append(f"Crop path reset at {len(shots) - 1} scene cut(s).")
    return CropPlan(clean, ratio, False, notes)


def sample_subject_track(
    video: Path, start: float, end: float, detector: SubjectDetector, cfg: SmartCropConfig,
    cancel: threading.Event | None = None, cuts: list[float] | None = None,
) -> list[tuple[float, float | None]]:
    """Detect the main subject every `sample_interval_s` (one frame in memory at a time).

    `cuts` are clip-relative scene cuts: one extra sample is taken just after each (so every shot is seen from its
    first moments) and the "stay with the previous subject" preference is forgotten at a cut, because after an edit
    the previous subject's position means nothing."""
    n = max(2, int((end - start) / cfg.sample_interval_s))
    times = [start + (end - start) * i / (n - 1) for i in range(n)]
    extra = [start + c + 0.15 for c in (cuts or []) if 0 < c and start + c + 0.15 < end]
    times = sorted(set(round(t, 3) for t in times + extra))
    cut_abs = sorted(start + c for c in (cuts or []))
    got: dict[float, float | None] = {t: None for t in times}
    prev: float | None = None
    last_t = start
    for t, frame in read_frames(video, times, max_width=640):
        if cancel is not None and cancel.is_set():
            break
        if any(last_t < c <= t for c in cut_abs):
            prev = None
        last_t = t
        box = pick_subject(detector.detect(frame), prev)
        if box:
            got[t] = prev = box.cx
    return [(t - start, got[t]) for t in times]
