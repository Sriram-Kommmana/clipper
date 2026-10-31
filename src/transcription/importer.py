"""Use an existing timed transcript (.srt / .vtt) instead of running Whisper.

Clips are cut by time, so a transcript without timestamps cannot be used. Word timing comes from the
file when it has it (YouTube-style WebVTT); otherwise it is estimated from each word's length, which is
close enough for boundaries and captions but not as exact as Whisper's.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.core.exceptions import TranscriptionError
from src.transcription.transcript_models import Transcript, TranscriptSegment, Word

SUPPORTED = (".srt", ".vtt")
STORED_NAME = "imported_transcript"

_STAMP = r"(?:\d+:)?\d{1,2}:\d{2}[.,]\d{1,3}"
_CUE = re.compile(rf"^\s*({_STAMP})\s*-->\s*({_STAMP})")
_INLINE = re.compile(r"<(\d+:\d{2}:\d{2}\.\d{3})>")
_TAG = re.compile(r"<[^>]*>|\{\\[^}]*\}")


def parse_stamp(text: str) -> float:
    parts = text.replace(",", ".").split(":")
    return sum(float(p) * 60 ** i for i, p in enumerate(reversed(parts)))


@dataclass
class _Cue:
    start: float
    end: float
    lines: list[str]


def _decode(raw: bytes) -> str:
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise TranscriptionError("The transcript file is not UTF-8 text.", recovery="Save it as UTF-8 (Notepad: Save as → Encoding: UTF-8) and try again.") from exc


def _read_cues(text: str) -> list[_Cue]:
    cues: list[_Cue] = []
    current: _Cue | None = None
    for line in text.splitlines():
        m = _CUE.match(line)
        if m:
            current = _Cue(parse_stamp(m.group(1)), parse_stamp(m.group(2)), [])
            cues.append(current)
        elif not line:              # only a truly empty line ends a cue: YouTube's cues begin with a line holding a single space
            current = None
        elif current is not None and line.strip():
            current.lines.append(line.strip())
    return sorted(cues, key=lambda c: c.start)


def _estimated_words(text: str, start: float, end: float) -> list[Word]:
    tokens = text.split()
    total = sum(len(t) for t in tokens) or 1
    words, t = [], start
    for tok in tokens:
        dur = (end - start) * len(tok) / total
        words.append(Word(text=tok, start=round(t, 3), end=round(t + dur, 3)))
        t += dur
    return words


def _timed_words(line: str, start: float, end: float) -> list[Word]:
    """Words of a line with inline <time> tags: 'hello<00:00:00.480><c> there</c>'."""
    pieces = _INLINE.split(line)          # [text, time, text, time, text...]
    times = [start] + [parse_stamp(t) for t in pieces[1::2]]
    texts = [_TAG.sub("", p).strip() for p in pieces[0::2]]
    words = []
    for i, (t, w) in enumerate(zip(times, texts)):
        if w:
            words.append(Word(text=w, start=t, end=max(t, times[i + 1] if i + 1 < len(times) else end)))
    return words


def _guess_language(text: str, fallback: str | None) -> str:
    if fallback:
        return fallback
    letters = [c for c in text if c.isalpha()]
    devanagari = sum("ऀ" <= c <= "ॿ" for c in letters)
    return "hi" if letters and devanagari / len(letters) > 0.3 else "unknown"


def parse_transcript(text: str, duration: float, language: str | None = None) -> tuple[Transcript, list[str]]:
    """Returns (transcript, warnings). Raises TranscriptionError if nothing usable lines up with the video."""
    cues = _read_cues(text)
    if not cues:
        raise TranscriptionError(
            "The transcript has no timestamps.",
            recovery="Clips are cut by time, so plain text cannot be used. Provide an .srt or .vtt file (text with start and end times).",
        )
    segments: list[TranscriptSegment] = []
    previous: set[str] = set()
    last_end = 0.0
    beyond = 0
    for cue in cues:
        if cue.start >= duration:
            beyond += 1
            continue
        # YouTube's rolling captions repeat the previous cue's lines, so only new lines count
        cleaned = [(raw, " ".join(_TAG.sub("", raw).split())) for raw in cue.lines]
        fresh = [(raw, plain) for raw, plain in cleaned if plain and plain not in previous]
        previous = {plain for _, plain in cleaned}
        start, end = max(cue.start, last_end), min(cue.end, duration)
        if not fresh or end <= start:
            continue
        words: list[Word] = []
        for raw, plain in fresh:
            words += _timed_words(raw, start, end) if _INLINE.search(raw) else _estimated_words(plain, start, end)
        segments.append(TranscriptSegment(id=len(segments), start=round(start, 3), end=round(end, 3),
                                          text=" ".join(plain for _, plain in fresh), words=words))
        last_end = end
    if not segments:
        raise TranscriptionError(
            "None of the transcript's timestamps fall inside this video.",
            recovery="Check that the transcript belongs to this video and starts at the same point.",
        )
    warnings = ["Imported transcript has no word-level timing, so word timing is estimated and captions may drift slightly."] \
        if not any(_INLINE.search(line) for c in cues for line in c.lines) else []
    if beyond:
        warnings.append(f"{beyond} transcript lines start after the end of the video and were ignored; check it is the right transcript.")
    if segments[-1].end < duration * 0.5:
        warnings.append(f"The transcript stops at {segments[-1].end:.0f}s but the video lasts {duration:.0f}s; clips can only come from the transcribed part.")
    body = " ".join(s.text for s in segments)
    has_timing = any(_INLINE.search(line) for c in cues for line in c.lines)
    return Transcript(language=_guess_language(body, language), language_probability=1.0, duration=duration, model="imported",
                      has_word_timestamps=has_timing, segments=segments), warnings


def stored_import(project_root: Path) -> Path | None:
    """The transcript previously imported into this project, if any."""
    return next((p for p in (project_root / f"{STORED_NAME}{ext}" for ext in SUPPORTED) if p.is_file()), None)


def store_import(source: str | Path, project_root: Path) -> Path:
    """Copy a transcript into the project, so later runs keep using it without the original file."""
    src = Path(str(source).strip().strip('"')).expanduser()
    if not src.is_file():
        raise TranscriptionError(f"Transcript file not found: {src}", recovery="Check the path of the transcript file.")
    if src.suffix.lower() not in SUPPORTED:
        raise TranscriptionError(f"Unsupported transcript type '{src.suffix}'.",
                                 recovery="Clips are cut by time, so the transcript needs timestamps: use an .srt or .vtt file.")
    target = project_root / f"{STORED_NAME}{src.suffix.lower()}"
    if src.resolve() != target.resolve():
        old = stored_import(project_root)
        tmp = target.with_suffix(target.suffix + ".part")
        shutil.copyfile(src, tmp)
        tmp.replace(target)
        if old and old != target:
            old.unlink(missing_ok=True)
    return target


def load_import(path: Path, duration: float, language: str | None = None) -> tuple[Transcript, list[str]]:
    return parse_transcript(_decode(path.read_bytes()), duration, language)
