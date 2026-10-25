"""Caption generation: word-timed phrases, ASS (karaoke highlight / plain) and SRT output, plus editing.

All SubtitleLine times are in *source-video* seconds, so captions stay aligned when clip boundaries change.
ASS/SRT output is shifted to clip time (0 = first frame of the clip).
"""
from __future__ import annotations

import re

from config import SubtitlesConfig
from src.core.models import SubtitleLine, SubtitleWord
from src.transcription.transcript_models import Transcript, srt_time

_SENT_END = re.compile(r"[.!?…。！？]+[\"')\]]*$")
_COMMA = re.compile(r"[,;:，、]$")
_EDIT_LINE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*\|\s*(.*?)\s*$")
MIN_EVENT_S = 0.04


# ---------------------------------------------------------------- building lines
def words_for_range(transcript: Transcript, start: float, end: float) -> list[SubtitleWord]:
    out: list[SubtitleWord] = []
    for seg in transcript.segments_in_range(start, end):
        if seg.words:
            for w in seg.words:
                mid = (w.start + w.end) / 2
                if start <= mid <= end and w.text.strip():
                    out.append(SubtitleWord(text=w.text.strip(), start=w.start, end=w.end))
        else:  # no word timing from the recogniser: spread the words evenly over the segment
            toks = seg.text.split()
            if not toks:
                continue
            step = (seg.end - seg.start) / len(toks)
            for i, tok in enumerate(toks):
                a, b = seg.start + i * step, seg.start + (i + 1) * step
                if start <= (a + b) / 2 <= end:
                    out.append(SubtitleWord(text=tok, start=a, end=b))
    return out


def group_words(words: list[SubtitleWord], cfg: SubtitlesConfig) -> list[SubtitleLine]:
    """Split into caption lines: hard breaks at pauses / sentence ends / commas, then balanced chunks
    (a 9-word sentence becomes 3+3+3 rather than 4+4+1)."""
    runs: list[list[SubtitleWord]] = []
    cur: list[SubtitleWord] = []
    for w in words:
        if cur and w.start - cur[-1].end >= cfg.pause_s:
            runs.append(cur)
            cur = []
        cur.append(w)
        if _SENT_END.search(w.text) or (_COMMA.search(w.text) and len(cur) >= 3):
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    lines: list[SubtitleLine] = []
    for run in runs:
        for chunk in _balanced_chunks(run, cfg):
            lines.append(SubtitleLine(start=chunk[0].start, end=chunk[-1].end, text=" ".join(w.text for w in chunk), words=list(chunk)))
    return lines


def _balanced_chunks(run: list[SubtitleWord], cfg: SubtitlesConfig) -> list[list[SubtitleWord]]:
    n = len(run)
    k = max(1, -(-n // cfg.max_words))
    while k < n:                                   # more lines until every chunk fits max_chars
        sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
        chunks, i = [], 0
        for sz in sizes:
            chunks.append(run[i:i + sz])
            i += sz
        if all(len(" ".join(w.text for w in c)) <= cfg.max_chars or len(c) == 1 for c in chunks):
            return chunks
        k += 1
    return [[w] for w in run] if k >= n and n > 1 else [run]


def lines_from_transcript(transcript: Transcript, start: float, end: float, cfg: SubtitlesConfig) -> list[SubtitleLine]:
    return group_words(words_for_range(transcript, start, end), cfg)


def clip_lines(lines: list[SubtitleLine], start: float, end: float) -> list[SubtitleLine]:
    """Lines (and words) that fall inside [start, end]; used so an override survives boundary edits."""
    out = []
    for ln in lines:
        if ln.end <= start or ln.start >= end:
            continue
        words = [w for w in ln.words if w.end > start and w.start < end]
        if ln.words and not words:
            continue
        out.append(ln.model_copy(update={"start": max(ln.start, start), "end": min(ln.end, end), "words": words}))
    return out


# ---------------------------------------------------------------- editing round trip
def format_for_editor(lines: list[SubtitleLine], clip_start: float) -> str:
    return "\n".join(f"{max(0.0, l.start - clip_start):.2f}-{max(0.0, l.end - clip_start):.2f} | {l.text}" for l in lines)


def parse_editor_text(text: str, clip_start: float, clip_end: float) -> list[tuple[float, float, str]]:
    """Parse 'start-end | text' lines (clip-relative seconds). Raises ValueError naming the bad line."""
    dur = clip_end - clip_start
    specs: list[tuple[float, float, str]] = []
    for n, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        m = _EDIT_LINE.match(raw)
        if not m:
            raise ValueError(f"Line {n}: expected 'start-end | text' (seconds), got: {raw[:60]!r}")
        a, b, t = float(m.group(1)), float(m.group(2)), m.group(3)
        if b <= a:
            raise ValueError(f"Line {n}: end ({b}) must be after start ({a}).")
        if a > dur + 0.5:
            raise ValueError(f"Line {n}: starts after the end of the clip ({dur:.1f}s).")
        if t:
            specs.append((clip_start + a, min(clip_start + b, clip_end), t))
    specs.sort()
    return specs


def reconcile(old: list[SubtitleLine], specs: list[tuple[float, float, str]]) -> list[SubtitleLine]:
    """Turn edited (start, end, text) specs into lines, keeping per-word timing wherever possible."""
    out: list[SubtitleLine] = []
    for a, b, text in specs:
        words_txt = text.split()
        best = max(old, key=lambda o: max(0.0, min(o.end, b) - max(o.start, a)), default=None)
        overlap = max(0.0, min(best.end, b) - max(best.start, a)) if best else 0.0
        if best and best.words and overlap > 0 and len(best.words) == len(words_txt) and abs(best.start - a) < 0.05 and abs(best.end - b) < 0.05:
            ws = [SubtitleWord(text=t, start=w.start, end=w.end) for t, w in zip(words_txt, best.words)]
        else:
            step = (b - a) / max(1, len(words_txt))
            ws = [SubtitleWord(text=t, start=a + i * step, end=a + (i + 1) * step) for i, t in enumerate(words_txt)]
        out.append(SubtitleLine(start=a, end=b, text=" ".join(words_txt), words=ws))
    return out


# ---------------------------------------------------------------- outputs
def build_srt(lines: list[SubtitleLine], clip_start: float, clip_end: float) -> str:
    blocks = []
    for i, l in enumerate(clip_lines(lines, clip_start, clip_end), 1):
        a, b = max(0.0, l.start - clip_start), l.end - clip_start
        blocks.append(f"{i}\n{srt_time(a)} --> {srt_time(max(b, a + 0.1))}\n{l.text}\n")
    return "\n".join(blocks)


def ass_color(hex_rgb: str, alpha: int = 0) -> str:
    r, g, b = hex_rgb[1:3], hex_rgb[3:5], hex_rgb[5:7]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def ass_time(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def escape_ass(text: str, upper: bool = False) -> str:
    t = text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\r", " ").replace("\n", " ")
    return t.upper() if upper else t


def build_ass(lines: list[SubtitleLine], clip_start: float, clip_end: float, width: int, height: int, cfg: SubtitlesConfig) -> str:
    """ASS document in clip time. `karaoke` = one event per word state with the active word highlighted."""
    scale = min(width, height) / 1080.0
    size = max(10, round(cfg.font_size * scale))
    outline, shadow = cfg.outline_px * scale, cfg.shadow_px * scale
    align = {"bottom": 2, "middle": 5, "top": 8}[cfg.position]
    margin_v = round(cfg.margin_v_pct * height)
    margin_h = round(cfg.margin_h_pct * width)
    primary, hi, edge = ass_color(cfg.primary_color), ass_color(cfg.highlight_color), ass_color(cfg.outline_color)
    head = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: 0\nScaledBorderAndShadow: yes\nYCbCr Matrix: None\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, "
        "StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{cfg.font},{size},{primary},{hi},{edge},&H80000000,{-1 if cfg.bold else 0},0,0,0,100,100,0,0,1,"
        f"{outline:.1f},{shadow:.1f},{align},{margin_h},{margin_h},{margin_v},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events: list[str] = []
    visible = clip_lines(lines, clip_start, clip_end)
    for idx, ln in enumerate(visible):
        limit = visible[idx + 1].start if idx + 1 < len(visible) else clip_end
        limit = min(limit, clip_end)
        line_end = min(max(ln.end, ln.start + 0.3), limit) if limit > ln.start else ln.end
        if cfg.mode == "plain" or len(ln.words) < 1:
            events.append(_dialogue(ln.start - clip_start, line_end - clip_start, escape_ass(ln.text, cfg.uppercase)))
            continue
        words = ln.words
        for i, w in enumerate(words):
            ev_start = w.start if i > 0 else ln.start
            nxt = words[i + 1].start if i + 1 < len(words) else line_end
            ev_end = min(max(nxt, ev_start + MIN_EVENT_S), line_end if i + 1 == len(words) else nxt)
            if ev_end - ev_start < MIN_EVENT_S:
                continue
            parts = []
            for j, x in enumerate(words):
                t = escape_ass(x.text, cfg.uppercase)
                if j == i:
                    pop = "\\t(0,70,\\fscx108\\fscy108)\\t(70,140,\\fscx100\\fscy100)" if cfg.pop_animation else ""
                    parts.append(f"{{\\1c{hi}{pop}}}{t}{{\\r}}")
                else:
                    parts.append(t)
            events.append(_dialogue(ev_start - clip_start, ev_end - clip_start, " ".join(parts)))
    return head + "\n".join(events) + ("\n" if events else "")


def _dialogue(a: float, b: float, text: str) -> str:
    return f"Dialogue: 0,{ass_time(a)},{ass_time(b)},Default,,0,0,0,,{text}"
