"""Builders for synthetic transcripts / candidates used across tests."""
from __future__ import annotations

from src.core.models import Candidate, ScoreBreakdown
from src.transcription.transcript_models import Transcript, TranscriptSegment, Word


def make_transcript(lines: list[tuple[float, str]], word_dur: float = 0.35, gap: float = 0.1, duration: float | None = None) -> Transcript:
    """lines: [(start_time, text)]. Words are laid out sequentially from each start."""
    segs = []
    for i, (start, text) in enumerate(lines):
        t, words = start, []
        for w in text.split():
            words.append(Word(text=w, start=round(t, 3), end=round(t + word_dur, 3)))
            t += word_dur + gap
        segs.append(TranscriptSegment(id=i, start=start, end=round(t - gap, 3), text=text, words=words))
    dur = duration if duration is not None else segs[-1].end + 5
    return Transcript(language="en", duration=dur, segments=segs, has_word_timestamps=True)


def long_transcript(minutes: int = 30, seg_every: float = 6.0) -> Transcript:
    lines, t, i = [], 0.0, 0
    while t < minutes * 60:
        lines.append((t, f"This is sentence number {i} in the long talk about topic {i // 10}."))
        t += seg_every
        i += 1
    return make_transcript(lines)


def cand(id: str, start: float, end: float, *, score: float | None = None, category: str = "story", **kw) -> Candidate:
    scores = ScoreBreakdown(total=score) if score is not None else None
    return Candidate(id=id, start=start, end=end, category=category, title=f"title {id}", scores=scores, **kw)
