"""Timestamp validation and boundary refinement (Stage G).

Refinement uses the transcript's own word timing, so clips begin at the start of a sentence (or at
least a word) and end after the last word - never mid-word, and never with a fixed offset.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from config import BoundariesConfig, SelectionConfig
from src.analysis.audio_analysis import AudioAnalysis
from src.core.exceptions import CandidateValidationError
from src.core.models import Candidate
from src.transcription.transcript_models import Transcript, nearest

MANUAL_MIN_S = 2.0
MANUAL_MAX_S = 180.0
MIN_SPEECH_WORDS = 8


@dataclass(frozen=True)
class Limits:
    min_s: float
    max_s: float
    overflow: float

    @property
    def hard_max(self) -> float:
        return self.max_s * self.overflow

    @classmethod
    def from_config(cls, sel: SelectionConfig) -> "Limits":
        return cls(sel.min_duration_s, sel.max_duration_s, sel.overflow_tolerance)


def check_bounds(start: float, end: float, source_duration: float) -> None:
    """Hard validity of a time range. Raises CandidateValidationError with a precise message."""
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v) for v in (start, end)):
        raise CandidateValidationError("Start and end must be finite numbers (seconds).")
    if start < 0:
        raise CandidateValidationError(f"Start ({start:.2f}s) is negative.")
    if end <= start:
        raise CandidateValidationError(f"End ({end:.2f}s) must be after start ({start:.2f}s).")
    if start >= source_duration:
        raise CandidateValidationError(f"Start ({start:.2f}s) is beyond the end of the video ({source_duration:.2f}s).")
    if end > source_duration + 0.05:
        raise CandidateValidationError(f"End ({end:.2f}s) is beyond the end of the video ({source_duration:.2f}s).")


def validate_manual_bounds(start: float, end: float, source_duration: float, limits: Limits) -> list[str]:
    """For user edits: raise on invalid ranges, return soft warnings for out-of-preference durations."""
    check_bounds(start, end, source_duration)
    d = end - start
    if d < MANUAL_MIN_S:
        raise CandidateValidationError(f"A clip must be at least {MANUAL_MIN_S:.0f}s long (got {d:.1f}s).")
    if d > MANUAL_MAX_S:
        raise CandidateValidationError(f"A clip may be at most {MANUAL_MAX_S:.0f}s long (got {d:.1f}s).")
    warns = []
    if d < limits.min_s:
        warns.append(f"Shorter than the configured minimum ({limits.min_s:.0f}s).")
    if d > limits.max_s:
        warns.append(f"Longer than the configured maximum ({limits.max_s:.0f}s).")
    return warns


def refine_boundaries(c: Candidate, transcript: Transcript, bcfg: BoundariesConfig, limits: Limits, source_duration: float) -> Candidate:
    """Snap to sentence/word edges, add small padding, enforce duration limits. Returns a new Candidate."""
    words = transcript.words()
    starts, ends = transcript.sentence_starts(), transcript.sentence_ends()
    start, end = c.start, min(c.end, source_duration)
    warnings = list(c.warnings)
    snap = bcfg.sentence_snap_s

    s = nearest(starts, start, lo=start - snap, hi=start + snap * 0.5)
    if s is not None:
        start = s
    else:  # no sentence edge nearby: at least never start inside a word
        for w in words:
            if w.start <= start < w.end:
                start = w.start
                break
            if w.start > start:
                break
    e = nearest(ends, end, lo=end - snap * 0.5, hi=end + snap)
    if e is not None:
        end = e
    else:  # never end inside a word
        for w in words:
            if w.start < end <= w.end:
                end = w.end
                break
            if w.start >= end:
                break

    # trim/extend toward the duration limits using sentence edges (setup & payoff stay intact)
    if end - start > limits.hard_max:
        t = nearest(ends, start + limits.max_s, lo=start + limits.min_s, hi=start + limits.hard_max)
        if t is not None:
            warnings.append(f"Trimmed to a sentence end to fit the {limits.max_s:.0f}s limit; check the payoff is intact.")
            end = t
    if end - start < limits.min_s:
        t = nearest(ends, start + limits.min_s, lo=start + limits.min_s, hi=min(start + limits.max_s, source_duration))
        if t is not None:
            warnings.append("Extended to a sentence end to reach the minimum duration.")
            end = t

    # padding that never reaches into neighbouring speech
    if words:
        prev_end = max((w.end for w in words if w.end <= start + 0.01), default=0.0)
        next_start = min((w.start for w in words if w.start >= end - 0.01), default=source_duration)
        start = max(start - bcfg.pad_before_s, (prev_end + start) / 2 if prev_end < start else start, 0.0)
        end = min(end + bcfg.pad_after_s, (end + next_start) / 2 if next_start > end else end, source_duration)
    else:
        start, end = max(0.0, start - bcfg.pad_before_s), min(source_duration, end + bcfg.pad_after_s)

    d = end - start
    if limits.max_s < d <= limits.hard_max:
        warnings.append(f"Longer than {limits.max_s:.0f}s - kept whole because the moment is complete.")
    out = c.model_copy(update={
        "start": round(start, 3), "end": round(end, 3), "warnings": _dedupe(warnings),
        "excerpt": transcript.text_in_range(start, end),
    })
    return out


def hard_failures(c: Candidate, transcript: Transcript, audio: AudioAnalysis | None, source_duration: float, limits: Limits) -> list[str]:
    """Reasons the candidate must not be rendered (empty list = OK)."""
    try:
        check_bounds(c.start, c.end, source_duration)
    except CandidateValidationError as exc:
        return [str(exc)]
    reasons: list[str] = []
    d = c.duration
    if d < limits.min_s - 0.5:
        reasons.append(f"too short ({d:.0f}s < {limits.min_s:.0f}s)")
    if d > limits.hard_max + 0.5:
        reasons.append(f"too long ({d:.0f}s > {limits.hard_max:.0f}s)")
    words = transcript.word_count_in_range(c.start, c.end)
    has_energy = bool(audio and audio.has_audio and audio.mean_energy(c.start, c.end) > 0.05)
    if words < MIN_SPEECH_WORDS and not (c.source == "audio" and has_energy):
        reasons.append(f"almost no speech ({words} words) and no strong audio")
    if audio and audio.has_audio and audio.silence_ratio(c.start, c.end) > 0.8:
        reasons.append("mostly silence")
    return reasons


def _dedupe(items: list[str]) -> list[str]:
    seen: list[str] = []
    for i in items:
        if i not in seen:
            seen.append(i)
    return seen
