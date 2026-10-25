"""Merge duplicate / overlapping candidates (incl. the same moment found by two overlapping sections)."""
from __future__ import annotations

from src.core.models import Candidate


def overlap_seconds(a: Candidate, b: Candidate) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def overlap_ratio(a: Candidate, b: Candidate) -> float:
    """Intersection divided by the *shorter* clip (so a clip fully inside another counts as 1.0)."""
    return overlap_seconds(a, b) / max(min(a.duration, b.duration), 1e-6)


def _strength(c: Candidate) -> tuple[bool, float]:
    # LLM understanding always outranks an audio-only trigger; then own confidence and completeness
    return (c.source != "audio", c.confidence * 0.5 + c.context_completeness * 0.3 + c.payoff_estimate * 0.2)


def merge_pair(a: Candidate, b: Candidate, max_union_s: float) -> Candidate:
    """Combine two overlapping candidates. Union the span if it stays within `max_union_s`, else keep the stronger one."""
    primary, other = (a, b) if _strength(a) >= _strength(b) else (b, a)
    start, end = min(a.start, b.start), max(a.end, b.end)
    warnings = list(dict.fromkeys(primary.warnings + other.warnings))
    if end - start <= max_union_s:
        s, e = start, end
    else:
        s, e = primary.start, primary.end
    return primary.model_copy(update={
        "start": s, "end": e,
        "warnings": warnings,
        "context_completeness": max(a.context_completeness, b.context_completeness),
        "payoff_estimate": max(a.payoff_estimate, b.payoff_estimate),
        "merged_from": list(dict.fromkeys(primary.merged_from + other.merged_from + [other.id])),
        "source": "llm" if "llm" in (a.source, b.source) else primary.source,
    })


def merge_candidates(cands: list[Candidate], max_union_s: float, threshold: float = 0.5) -> tuple[list[Candidate], int]:
    """Repeatedly merge any pair with overlap_ratio >= threshold. Returns (merged list sorted by start, merges done)."""
    pool = sorted(cands, key=lambda c: (c.start, c.end))
    merges = 0
    changed = True
    while changed:
        changed = False
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                if pool[j].start >= pool[i].end:
                    break
                if overlap_ratio(pool[i], pool[j]) >= threshold:
                    pool[i] = merge_pair(pool[i], pool[j], max_union_s)
                    del pool[j]
                    merges += 1
                    changed = True
                    break
            if changed:
                break
        pool.sort(key=lambda c: (c.start, c.end))
    return pool, merges
