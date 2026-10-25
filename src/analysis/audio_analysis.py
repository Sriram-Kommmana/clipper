"""Audio analysis: RMS energy, silences, peaks, sustained high-energy spans, sudden changes, quality.

Audio is streamed in blocks (never loaded whole) so a multi-hour file stays within RAM limits.
Energy is normalised *per recording with a minimum dynamic range*, so flat/quiet audio does not
manufacture fake "peaks". These events are only hints for candidate generation and scoring.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, Field
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from config import AudioConfig
from src.core.exceptions import JobCancelled

MIN_SPAN_DB = 12.0  # smallest dynamic range treated as meaningful


class AudioEvent(BaseModel):
    start: float
    end: float
    kind: Literal["peak", "sustained", "change_up", "change_down"]
    strength: float = Field(ge=0, le=1)


class Interval(BaseModel):
    start: float
    end: float


class AudioAnalysis(BaseModel):
    window_s: float
    duration: float = 0.0
    has_audio: bool = True
    energy: list[float] = Field(default_factory=list)        # normalised 0-1 per window
    rms_db: list[float] = Field(default_factory=list)
    silences: list[Interval] = Field(default_factory=list)
    events: list[AudioEvent] = Field(default_factory=list)
    quality: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    def _slice(self, arr: list[float], start: float, end: float) -> np.ndarray:
        if not arr:
            return np.zeros(0)
        i0 = max(0, int(start / self.window_s))
        i1 = max(i0 + 1, int(np.ceil(end / self.window_s)))
        return np.asarray(arr[i0:i1], dtype=float)

    def mean_energy(self, start: float, end: float) -> float:
        s = self._slice(self.energy, start, end)
        return float(s.mean()) if s.size else 0.0

    def max_energy(self, start: float, end: float) -> float:
        s = self._slice(self.energy, start, end)
        return float(s.max()) if s.size else 0.0

    def silence_ratio(self, start: float, end: float) -> float:
        if end <= start:
            return 0.0
        covered = sum(max(0.0, min(end, s.end) - max(start, s.start)) for s in self.silences)
        return min(1.0, covered / (end - start))

    def events_in(self, start: float, end: float, kinds: tuple[str, ...] | None = None) -> list[AudioEvent]:
        return [e for e in self.events if e.end > start and e.start < end and (kinds is None or e.kind in kinds)]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(first_index, last_index_exclusive)] of True runs."""
    if mask.size == 0:
        return []
    padded = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(padded)
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def compute_rms_blocks(
    wav: Path, window_s: float, *, cancel: threading.Event | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Per-window RMS (linear) via librosa, streaming from disk. Returns (rms, duration_s, raw_stats)."""
    import librosa
    import soundfile as sf

    with sf.SoundFile(str(wav)) as f:
        sr, total = f.samplerate, f.frames
        win = max(1, int(round(window_s * sr)))
        block = win * 600  # ~5 minutes at 0.5 s windows
        rms_parts: list[np.ndarray] = []
        clipped = 0
        peak = 0.0
        dc_acc = 0.0
        read = 0
        for chunk in f.blocks(blocksize=block, dtype="float32", always_2d=True):
            if cancel is not None and cancel.is_set():
                raise JobCancelled("Cancelled during audio analysis")
            y = chunk.mean(axis=1)
            n_full = (len(y) // win) * win
            if n_full:
                rms_parts.append(librosa.feature.rms(y=y[:n_full], frame_length=win, hop_length=win, center=False)[0])
            clipped += int(np.count_nonzero(np.abs(y) >= 0.99))
            peak = max(peak, float(np.abs(y).max(initial=0.0)))
            dc_acc += float(y.sum())
            read += len(y)
            if on_progress and total:
                on_progress(min(1.0, read / total))
    rms = np.concatenate(rms_parts) if rms_parts else np.zeros(0)
    stats = {
        "clipping_ratio": clipped / max(1, read),
        "peak_dbfs": float(20 * np.log10(max(peak, 1e-9))),
        "dc_offset": abs(dc_acc / max(1, read)),
    }
    return rms, total / sr if sr else 0.0, stats


def analyze_rms(rms: np.ndarray, duration: float, cfg: AudioConfig, raw: dict[str, float] | None = None) -> AudioAnalysis:
    """Pure function from per-window RMS to the full analysis (unit-tested on synthetic arrays)."""
    w = cfg.window_s
    out = AudioAnalysis(window_s=w, duration=duration)
    if rms.size == 0:
        out.has_audio = False
        out.warnings.append("Audio stream is empty.")
        return out
    db = 20 * np.log10(np.maximum(rms, 1e-7))
    p5, p95, p995 = (float(np.percentile(db, q)) for q in (5, 95, 99.5))
    floor = max(p5, cfg.silence_floor_db)
    span = max(p995 - floor, MIN_SPAN_DB)
    energy = np.clip((db - floor) / span, 0.0, 1.0)
    smooth = uniform_filter1d(energy, size=3, mode="nearest")

    # silences: absolute floor, or a little above the recording's own noise floor (capped)
    sil_thr = min(max(cfg.silence_floor_db, p5 + 4.0), -30.0)
    min_sil = max(1, int(round(cfg.min_silence_s / w)))
    out.silences = [Interval(start=a * w, end=b * w) for a, b in _runs(db < sil_thr) if b - a >= min_sil]

    events: list[AudioEvent] = []
    # isolated peaks
    dist = max(1, int(2.0 / w))
    # (raw energy, not smoothed: a half-second shout or crash is a legitimate peak)
    idx, _ = find_peaks(energy, height=0.6, prominence=cfg.peak_prominence, distance=dist)
    for i in idx:
        events.append(AudioEvent(start=i * w, end=(i + 1) * w, kind="peak", strength=float(energy[i])))
    # sustained high energy (bridge 1-window gaps)
    high = smooth >= cfg.sustained_threshold
    closed = np.convolve(high.astype(int), np.ones(3, dtype=int), "same") >= 2 if high.any() else high
    min_len = max(1, int(round(cfg.sustained_min_s / w)))
    for a, b in _runs(closed | high):
        if b - a >= min_len:
            events.append(AudioEvent(start=a * w, end=b * w, kind="sustained", strength=float(smooth[a:b].mean())))
    # sudden changes
    if smooth.size > 3:
        d = np.diff(smooth)
        thr = max(cfg.change_sigma * float(d.std()), 0.2)
        for i in np.where(d > thr)[0]:
            events.append(AudioEvent(start=(i + 1) * w, end=(i + 2) * w, kind="change_up", strength=float(min(1.0, d[i]))))
        for i in np.where(d < -thr)[0]:
            events.append(AudioEvent(start=(i + 1) * w, end=(i + 2) * w, kind="change_down", strength=float(min(1.0, -d[i]))))
    events.sort(key=lambda e: (e.start, e.kind))

    out.energy = [round(float(x), 4) for x in energy]
    out.rms_db = [round(float(x), 2) for x in db]
    out.events = events
    sil_total = sum(s.end - s.start for s in out.silences)
    out.quality = {
        "mean_db": round(float(db.mean()), 2),
        "noise_floor_db": round(float(np.percentile(db[db > -90], 5)) if (db > -90).any() else p5, 2),
        "dynamic_range_db": round(p95 - p5, 2),
        "silence_ratio": round(sil_total / max(duration, 1e-6), 4),
        **{k: round(v, 5) for k, v in (raw or {}).items()},
    }
    q = out.quality
    if q["silence_ratio"] > 0.6:
        out.warnings.append(f"Audio is {q['silence_ratio']:.0%} silence - few clips expected.")
    if q.get("clipping_ratio", 0) > 0.005:
        out.warnings.append("Audio clipping detected (distorted loud sections).")
    if q["mean_db"] < -45:
        out.warnings.append("Audio is very quiet; transcription accuracy may suffer.")
    if q["dynamic_range_db"] < 6:
        out.warnings.append("Audio has almost no dynamic range (flat/noise-like); energy peaks are unreliable.")
    return out


def analyze_audio_file(wav: Path, cfg: AudioConfig, *, cancel=None, on_progress=None) -> AudioAnalysis:
    rms, dur, raw = compute_rms_blocks(wav, cfg.window_s, cancel=cancel, on_progress=on_progress)
    return analyze_rms(rms, dur, cfg, raw)
