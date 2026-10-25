"""Candidate selection (Stage F): filters + greedy pick with separation, diversity and category preference."""
from __future__ import annotations

from collections import Counter

from config import SelectionConfig
from src.core.models import Candidate


def gap_between(a: Candidate, b: Candidate) -> float:
    """Seconds between two clips (negative if they overlap)."""
    return max(a.start, b.start) - min(a.end, b.end)


def select_candidates(cands: list[Candidate], sel: SelectionConfig) -> list[Candidate]:
    """Return all candidates with `selected` / `filter_reason` set. Order: selected first by score, then the rest.

    User decisions win: approved clips are always kept (and counted); rejected clips are never picked.
    """
    reset = [c.model_copy(update={"selected": False, "filter_reason": c.filter_reason if c.filter_reason and c.filter_reason.startswith("invalid:") else None}) for c in cands]
    by_id = {c.id: c for c in reset}
    chosen: list[Candidate] = []
    counts: Counter[str] = Counter()

    def can_add(c: Candidate) -> str | None:
        for s in chosen:
            if gap_between(c, s) < sel.min_separation_s:
                return f"too close to / overlapping selected clip {s.id}"
        return None

    def add(c: Candidate) -> None:
        chosen.append(c)
        counts[c.category] += 1
        by_id[c.id] = c.model_copy(update={"selected": True, "filter_reason": None})

    for c in sorted((c for c in reset if c.status == "approved"), key=lambda c: -c.total_score):
        if c.filter_reason is None and can_add(c) is None:
            add(c)

    pool = []
    for c in reset:
        if c.id in {x.id for x in chosen}:
            continue
        reason = c.filter_reason
        if reason is None and c.status == "rejected":
            reason = "rejected by user"
        elif reason is None and c.scores is None:
            reason = "not scored"
        elif reason is None and c.total_score < sel.min_score:
            reason = f"score {c.total_score:.0f} below minimum {sel.min_score:.0f}"
        if reason:
            by_id[c.id] = c.model_copy(update={"filter_reason": reason})
        else:
            pool.append(c)

    while pool and len(chosen) < sel.max_clips:
        def adjusted(c: Candidate) -> float:
            return c.total_score * sel.category_preferences.get(c.category, 1.0) - sel.diversity_penalty * counts[c.category]

        pool.sort(key=adjusted, reverse=True)
        c = pool.pop(0)
        why = can_add(c)
        if why:
            by_id[c.id] = c.model_copy(update={"filter_reason": why})
            continue
        add(c)
    for c in pool:  # left over because max_clips was reached
        by_id[c.id] = c.model_copy(update={"filter_reason": f"beyond the maximum of {sel.max_clips} clips"})

    ordered = sorted(by_id.values(), key=lambda c: (not c.selected, -c.total_score))
    return ordered
