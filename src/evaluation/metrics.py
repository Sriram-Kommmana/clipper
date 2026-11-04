"""Measure candidate quality against moments a person marked by hand.

Nothing in this project can say whether its clips are *good* without a reference; this module compares what the
pipeline found with a small hand-labelled set so that a change (a prompt, a weight, `llm.audio_hints`, a different
model) can be judged by numbers instead of by feel. It is deliberately tiny and dependency-free.

Labels are plain JSON, in source-video seconds:
    {"moments": [{"start": 312.0, "end": 361.5, "note": "the story about the broken guitar"}, ...]}
A moment is "found" when some clip overlaps it with temporal IoU >= `iou` (default 0.4): clips that capture most of a
moment, with a bit of slack on either side, count; clips that merely touch it do not.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_IOU = 0.4


class LabelError(ValueError):
    """The labels file is not usable (message says why)."""


@dataclass(frozen=True)
class Span:
    start: float
    end: float
    score: float = 0.0
    id: str = ""
    note: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


def iou(a: Span, b: Span) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = a.duration + b.duration - inter
    return inter / union if union > 0 else 0.0


def load_labels(path: Path | str) -> list[Span]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise LabelError(f"Cannot read labels file {path}: {exc}") from exc
    except ValueError as exc:
        raise LabelError(f"Labels file {path} is not valid JSON: {exc}") from exc
    items = raw.get("moments") if isinstance(raw, dict) else raw
    if not isinstance(items, list) or not items:
        raise LabelError('Labels file needs a non-empty "moments" list: [{"start": 12.0, "end": 58.0}, ...]')
    out: list[Span] = []
    for n, m in enumerate(items, 1):
        try:
            s, e = float(m["start"]), float(m["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LabelError(f"Moment {n}: needs numeric 'start' and 'end' (seconds)") from exc
        if not (0 <= s < e):
            raise LabelError(f"Moment {n}: end ({e}) must be after start ({s}), and start must be >= 0")
        out.append(Span(s, e, note=str(m.get("note", "")) if isinstance(m, dict) else ""))
    return out


@dataclass
class Report:
    labelled: int
    clips: int
    selected: int
    iou: float
    recall_candidates: float          # labelled moments covered by ANY candidate: the ceiling for everything downstream
    recall_selected: float            # ... covered by a clip the pipeline actually picked
    precision_selected: float         # picked clips that match a labelled moment
    mean_reciprocal_rank: float       # 1/rank of the best-scored clip matching each moment (rank among all clips by score)
    start_error_s: float | None       # median |clip start - moment start| over matched moments
    end_error_s: float | None
    score_gap: float | None           # mean score of matching clips minus mean score of the others (are good clips ranked higher?)
    missed: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _best(label: Span, clips: Sequence[Span]) -> tuple[Span | None, float]:
    best, best_iou = None, 0.0
    for c in clips:
        v = iou(label, c)
        if v > best_iou:
            best, best_iou = c, v
    return best, best_iou


def evaluate(labels: Sequence[Span], candidates: Iterable[Span], selected: Iterable[Span], *, iou_min: float = DEFAULT_IOU) -> Report:
    cands, sel = list(candidates), list(selected)
    ranked = sorted(cands, key=lambda c: -c.score)

    def covered(clips: Sequence[Span]) -> list[tuple[Span, Span | None]]:
        out = []
        for lab in labels:
            c, v = _best(lab, clips)
            out.append((lab, c if v >= iou_min else None))
        return out

    cov_c, cov_s = covered(cands), covered(sel)
    matched = [(lab, c) for lab, c in cov_c if c is not None]
    rr = []
    for lab in labels:
        pos = next((i for i, c in enumerate(ranked, 1) if iou(lab, c) >= iou_min), None)
        rr.append(1.0 / pos if pos else 0.0)
    hit_ids = {id(c) for c in cands if any(iou(lab, c) >= iou_min for lab in labels)}
    hit = [c.score for c in cands if id(c) in hit_ids]
    miss = [c.score for c in cands if id(c) not in hit_ids]
    return Report(
        labelled=len(labels), clips=len(cands), selected=len(sel), iou=iou_min,
        recall_candidates=sum(c is not None for _, c in cov_c) / len(labels),
        recall_selected=sum(c is not None for _, c in cov_s) / len(labels),
        precision_selected=(sum(any(iou(lab, c) >= iou_min for lab in labels) for c in sel) / len(sel)) if sel else 0.0,
        mean_reciprocal_rank=sum(rr) / len(rr),
        start_error_s=statistics.median(abs(c.start - lab.start) for lab, c in matched) if matched else None,
        end_error_s=statistics.median(abs(c.end - lab.end) for lab, c in matched) if matched else None,
        score_gap=(statistics.fmean(hit) - statistics.fmean(miss)) if hit and miss else None,
        missed=[{"start": lab.start, "end": lab.end, "note": lab.note} for lab, c in cov_c if c is None],
    )


def format_report(r: Report, name: str = "") -> str:
    def pct(x: float) -> str:
        return f"{100 * x:5.1f}%"

    def sec(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.1f}s"

    lines = [
        f"Evaluation{(' - ' + name) if name else ''}  ({r.labelled} labelled moments, {r.clips} candidates, {r.selected} selected, IoU >= {r.iou})",
        f"  moments found by any candidate   {pct(r.recall_candidates)}   <- the ceiling: a moment never proposed cannot be recovered later",
        f"  moments found among selected     {pct(r.recall_selected)}",
        f"  selected clips that are real     {pct(r.precision_selected)}",
        f"  mean reciprocal rank             {r.mean_reciprocal_rank:.2f}   (1.00 = a matching clip is always ranked first)",
        f"  median start / end error         {sec(r.start_error_s)} / {sec(r.end_error_s)}",
        f"  score gap (matching - others)    {'n/a' if r.score_gap is None else f'{r.score_gap:+.1f} points'}   (should be clearly positive)",
    ]
    for m in r.missed:
        lines.append(f"  MISSED {m['start']:.0f}-{m['end']:.0f}s {m['note']}".rstrip())
    return "\n".join(lines)


def compare(a: Report, b: Report) -> str:
    """Side-by-side deltas (b minus a), for A/B runs of the same video with different settings."""
    rows = [("moments found (any candidate)", a.recall_candidates, b.recall_candidates, True),
            ("moments found (selected)", a.recall_selected, b.recall_selected, True),
            ("selected clips that are real", a.precision_selected, b.precision_selected, True),
            ("mean reciprocal rank", a.mean_reciprocal_rank, b.mean_reciprocal_rank, False)]
    lines = ["                                   A        B     B - A"]
    for label, x, y, is_pct in rows:
        f = (lambda v: f"{100 * v:6.1f}%") if is_pct else (lambda v: f"{v:7.2f}")
        d = (y - x) * (100 if is_pct else 1)
        lines.append(f"  {label:<30} {f(x)} {f(y)}  {d:+6.1f}{'pp' if is_pct else ''}")
    return "\n".join(lines)
