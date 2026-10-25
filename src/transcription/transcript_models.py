"""Transcript data model with helpers used by candidate generation, validation and subtitles."""
from __future__ import annotations

import bisect
import re

from pydantic import BaseModel, Field

_SENTENCE_END = re.compile(r"[.!?…。！？]+[\"')\]]*$")


class Word(BaseModel):
    text: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    prob: float = Field(1.0, ge=0, le=1)


class TranscriptSegment(BaseModel):
    id: int
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    words: list[Word] = Field(default_factory=list)

    @property
    def ends_sentence(self) -> bool:
        return bool(_SENTENCE_END.search(self.text.strip()))


class Transcript(BaseModel):
    language: str = "unknown"
    language_probability: float = 0.0
    duration: float = 0.0
    model: str = ""
    device: str = ""
    compute_type: str = ""
    has_word_timestamps: bool = False
    segments: list[TranscriptSegment] = Field(default_factory=list)

    # ---- queries -------------------------------------------------------------------------------
    def words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]

    def segments_in_range(self, start: float, end: float) -> list[TranscriptSegment]:
        return [s for s in self.segments if s.end > start and s.start < end]

    def text_in_range(self, start: float, end: float) -> str:
        """Text of the words (or, without word timing, whole segments) overlapping [start, end]."""
        parts: list[str] = []
        for s in self.segments_in_range(start, end):
            if s.words:
                parts.extend(w.text for w in s.words if w.end > start and w.start < end)
            else:
                parts.append(s.text.strip())
        return _join_words(parts)

    def word_count_in_range(self, start: float, end: float) -> int:
        n = 0
        for s in self.segments_in_range(start, end):
            if s.words:
                n += sum(1 for w in s.words if w.end > start and w.start < end)
            else:
                n += len(s.text.split())
        return n

    def sentence_starts(self) -> list[float]:
        """Start times of sentence beginnings (segment starts after a sentence end, plus mid-segment ones)."""
        starts: list[float] = []
        prev_end_sentence = True
        for s in self.segments:
            if s.words:
                for w in s.words:
                    if prev_end_sentence:
                        starts.append(w.start)
                    prev_end_sentence = bool(_SENTENCE_END.search(w.text.strip()))
            else:
                if prev_end_sentence:
                    starts.append(s.start)
                prev_end_sentence = s.ends_sentence
        return starts

    def sentence_ends(self) -> list[float]:
        ends: list[float] = []
        for s in self.segments:
            if s.words:
                ends.extend(w.end for w in s.words if _SENTENCE_END.search(w.text.strip()))
            elif s.ends_sentence:
                ends.append(s.end)
        return ends

    def speech_coverage(self, start: float, end: float) -> float:
        """Fraction of [start, end] covered by transcript segments (0-1)."""
        if end <= start:
            return 0.0
        covered = 0.0
        for s in self.segments_in_range(start, end):
            covered += max(0.0, min(end, s.end) - max(start, s.start))
        return min(1.0, covered / (end - start))

    # ---- export --------------------------------------------------------------------------------
    def to_srt(self) -> str:
        blocks = []
        for i, s in enumerate(self.segments, 1):
            blocks.append(f"{i}\n{srt_time(s.start)} --> {srt_time(max(s.end, s.start + 0.05))}\n{s.text.strip()}\n")
        return "\n".join(blocks)


def _join_words(words: list[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(w.strip() for w in words)).strip()


def srt_time(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def nearest(sorted_values: list[float], target: float, *, lo: float, hi: float) -> float | None:
    """Value in `sorted_values` closest to `target` within [lo, hi], or None."""
    if not sorted_values:
        return None
    i = bisect.bisect_left(sorted_values, lo)
    best: float | None = None
    while i < len(sorted_values) and sorted_values[i] <= hi:
        v = sorted_values[i]
        if best is None or abs(v - target) < abs(best - target):
            best = v
        i += 1
    return best
