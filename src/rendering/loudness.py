"""Two-pass loudness normalisation for exported clips.

Clips cut from different parts of a long recording differ a lot in level, and social platforms re-normalise to about
-14 LUFS anyway. FFmpeg's `loudnorm` in its default single-pass mode rides the gain dynamically, which is audible as
"pumping" on speech (quiet asides swell, the tail of a loud line ducks). Given the clip's *measured* loudness it applies
one transparent linear gain instead (with true-peak limiting only where needed). So: measure first, then normalise with
the measurements. Any failure just means the clip keeps its original level; it never stops a render.
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
from dataclasses import dataclass

from src.core.exceptions import FFmpegError, JobCancelled
from src.core.models import VideoInfo
from src.rendering.ffmpeg_runner import FFmpegRunner

log = logging.getLogger(__name__)

TARGET_TP = -1.5
TARGET_LRA = 11.0
MIN_SPAN_S = 1.0             # shorter than this is not measurable
MAX_GAIN_DB = 24.0           # a source this quiet is broken (muted mic, wrong track); amplifying it only adds noise
TOLERANCE_DB = 0.5           # already this close to the target: leave the audio untouched


@dataclass(frozen=True)
class Loudness:
    input_i: float
    input_tp: float
    input_lra: float
    input_thresh: float
    target_offset: float


def _num(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def parse_measurement(stderr: str) -> Loudness | None:
    """The JSON block `loudnorm=print_format=json` writes at the end of ffmpeg's stderr, or None if absent/unusable
    (digital silence reports -inf, which cannot be normalised)."""
    blocks = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr, flags=re.S)
    if not blocks:
        return None
    try:
        raw = json.loads(blocks[-1])
    except ValueError:
        return None
    vals = [_num(raw.get(k)) for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")]
    if any(v is None for v in vals):
        return None
    return Loudness(*vals)  # type: ignore[arg-type]


def measure_args(info: VideoInfo, start: float, end: float, target_lufs: float) -> list[str]:
    return ["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", info.path, "-vn", "-map", "0:a:0",
            "-af", f"loudnorm=I={target_lufs:g}:TP={TARGET_TP:g}:LRA={TARGET_LRA:g}:print_format=json", "-f", "null", "-"]


def measure(runner: FFmpegRunner, info: VideoInfo, start: float, end: float, target_lufs: float,
            cancel: threading.Event | None = None) -> Loudness | None:
    if not info.has_audio or end - start < MIN_SPAN_S:
        return None
    try:
        res = runner.run(measure_args(info, start, end, target_lufs), duration=end - start, cancel=cancel, keep_stderr=True)
    except JobCancelled:
        raise
    except FFmpegError as exc:
        log.warning("Loudness measurement failed (%s); keeping the original level", exc)
        return None
    return parse_measurement(res.stderr)


def normalise_filter(m: Loudness | None, target_lufs: float) -> tuple[str | None, str]:
    """(audio filter or None, human note). None = leave the audio alone."""
    if m is None:
        return None, "Loudness could not be measured; original level kept."
    gain = target_lufs - m.input_i
    if m.input_i < -70:
        return None, "Audio is almost silent; original level kept."
    if abs(gain) <= TOLERANCE_DB:
        return None, f"Already at {m.input_i:.1f} LUFS; original level kept."
    if gain > MAX_GAIN_DB:
        return None, f"Audio is very quiet ({m.input_i:.1f} LUFS); not amplified by {gain:.0f} dB."
    f = (f"loudnorm=I={target_lufs:g}:TP={TARGET_TP:g}:LRA={TARGET_LRA:g}:measured_I={m.input_i:.2f}:measured_LRA={m.input_lra:.2f}"
         f":measured_TP={m.input_tp:.2f}:measured_thresh={m.input_thresh:.2f}:offset={m.target_offset:.2f}:linear=true")
    return f, f"Loudness {m.input_i:.1f} -> {target_lufs:g} LUFS."
