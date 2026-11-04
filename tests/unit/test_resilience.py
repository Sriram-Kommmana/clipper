"""Resilience: finished LLM answers survive failures, and failed parts of the transcript are reported, not hidden."""
import json
import re

import pytest

from config import load_config
from src.candidates.generator import llm_candidates, make_sections
from src.candidates.scorer import score_candidates
from src.core.models import SCHEMA_VERSION
from src.storage.cache import KeyedStore
from tests.fakes import ScriptedLLM
from tests.helpers import cand, long_transcript, make_transcript

CFG = load_config(local_path=None)


def moment(first_line, last_line):
    return dict(first_line=first_line, last_line=last_line, category="story", title="T", explanation="E", hook="", caption="C",
                context_completeness=8, payoff=7, confidence=7)


# ---------------------------------------------------------------- KeyedStore
def test_keyed_store_roundtrip_and_persistence(tmp_path):
    p = tmp_path / "a" / "answers.json"
    s = KeyedStore(p)
    assert s.get("k") is None and len(s) == 0
    s.put("k", {"x": 1})
    assert KeyedStore(p).get("k") == {"x": 1}, "a new instance reads what the old one wrote"


def test_keyed_store_treats_corrupt_or_foreign_files_as_empty(tmp_path):
    p = tmp_path / "answers.json"
    p.write_text("{not json", encoding="utf-8")
    s = KeyedStore(p)
    assert s.get("k") is None
    s.put("k", 1)                              # and it recovers by overwriting
    assert KeyedStore(p).get("k") == 1
    p.write_text(json.dumps({"schema": SCHEMA_VERSION + 99, "entries": {"k": 2}}), encoding="utf-8")
    assert KeyedStore(p).get("k") is None, "an entry from another schema version must not be trusted"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert KeyedStore(p).get("k") is None


def test_keyed_store_is_bounded_and_keeps_newest(tmp_path, monkeypatch):
    monkeypatch.setattr(KeyedStore, "MAX_ENTRIES", 3)
    s = KeyedStore(tmp_path / "a.json")
    for i in range(6):
        s.put(f"k{i}", i)
    assert len(s) == 3 and s.get("k0") is None and s.get("k5") == 5


# ---------------------------------------------------------------- section answers are kept
def _responder(fail_sections=(), calls=None):
    """Answers the brief and every section; sections whose window starts in `fail_sections` return garbage."""
    t = long_transcript(30)
    secs = make_sections(t, CFG.llm)

    def respond(system, prompt):
        if "File name:" in prompt:
            return json.dumps({"summary": "A talk."})
        for i, s in enumerate(secs):
            if f"({s.start:.0f}s - {s.end:.0f}s)" in prompt:
                if calls is not None:
                    calls.append(i)
                return "garbage" if i in fail_sections else json.dumps({"moments": [moment(10, 15)]})
        raise AssertionError("unexpected prompt")

    return t, secs, respond


def test_second_run_reuses_every_finished_section(tmp_path):
    calls: list[int] = []
    t, secs, respond = _responder(calls=calls)
    store = KeyedStore(tmp_path / "answers.json")
    first_llm = ScriptedLLM(respond)
    c1, s1 = llm_candidates(t, CFG, first_llm, "fp", store=store)
    assert s1["cached_sections"] == 0 and len(first_llm.calls) == len(secs) + 1

    again = ScriptedLLM(lambda s, p: (_ for _ in ()).throw(AssertionError("the model must not be asked again")))
    c2, s2 = llm_candidates(t, CFG, again, "fp", store=KeyedStore(tmp_path / "answers.json"))
    assert s2["cached_sections"] == len(secs) and not again.calls
    assert [(c.start, c.end) for c in c2] == [(c.start, c.end) for c in c1]


def test_only_failed_sections_are_retried(tmp_path):
    calls: list[int] = []
    t, secs, respond = _responder(fail_sections={1}, calls=calls)
    store = KeyedStore(tmp_path / "answers.json")
    _, s1 = llm_candidates(t, CFG, ScriptedLLM(respond), "fp", store=store)
    assert s1["failed_sections"] == [1]

    calls.clear()
    t2, _, healthy = _responder(calls=calls)           # the model recovers
    _, s2 = llm_candidates(t2, CFG, ScriptedLLM(healthy), "fp", store=KeyedStore(tmp_path / "answers.json"))
    assert s2["failed_sections"] == [] and calls == [1], "exactly the failed section is asked again"
    assert s2["cached_sections"] == len(secs) - 1


def test_changing_the_model_does_not_reuse_answers(tmp_path):
    t, secs, respond = _responder()
    store = KeyedStore(tmp_path / "answers.json")
    llm_candidates(t, CFG, ScriptedLLM(respond, model="a:1"), "fp", store=store)
    _, st = llm_candidates(t, CFG, ScriptedLLM(respond, model="b:2"), "fp", store=store)
    assert st["cached_sections"] == 0


# ---------------------------------------------------------------- scoring
def _score_respond(system, prompt):
    ids = re.findall(r"### Candidate (\S+) ", prompt)
    return json.dumps({"scores": [{"id": i, "hook": 8, "clarity": 7, "completeness": 8, "payoff": 8, "escalation": 7, "reason": "ok"} for i in ids]})


def test_score_batches_are_kept(tmp_path):
    t = make_transcript([(0, "A full sentence about something. And then more words follow right after it."),
                         (30, "Another sentence entirely. Followed by yet more talking about it.")], duration=80)
    cands = [cand(f"c{i}", i * 10.0, i * 10.0 + 30) for i in range(5)]
    cfg = CFG.with_overrides({"llm": {"scoring_batch_size": 2}})
    store = KeyedStore(tmp_path / "answers.json")
    llm = ScriptedLLM(_score_respond)
    out = score_candidates(cands, t, None, cfg, llm, store=store)
    assert len(llm.calls) == 3 and all(c.scores.llm_scored for c in out)

    again = ScriptedLLM(lambda s, p: (_ for _ in ()).throw(AssertionError("scored twice")))
    out2 = score_candidates(cands, t, None, cfg, again, store=KeyedStore(tmp_path / "answers.json"))
    assert not again.calls and [c.scores.total for c in out2] == [c.scores.total for c in out]


def test_scoring_prompt_shows_only_the_clip_text():
    """Context lets the model resolve references a viewer who starts at the first word cannot."""
    t = make_transcript([(0, "He said it again."), (4, "That is the whole story of the week.")], duration=20)
    c = cand("c1", 4.0, 8.0, excerpt="That is the whole story of the week.")
    llm = ScriptedLLM(_score_respond)
    score_candidates([c], t, None, CFG, llm, context="A DISTINCTIVE BRIEF")
    prompt = llm.calls[0][1]
    assert "That is the whole story" in prompt
    assert "He said it again" not in prompt and "DISTINCTIVE BRIEF" not in prompt
    assert "context before" not in prompt.lower() and "context after" not in prompt.lower()


@pytest.mark.parametrize("n_fail", [0, 1])
def test_coverage_stats_name_the_failed_sections(tmp_path, n_fail):
    t, secs, respond = _responder(fail_sections=set(range(n_fail)))
    _, st = llm_candidates(t, CFG, ScriptedLLM(respond), "fp", store=KeyedStore(tmp_path / "a.json"))
    assert st["failed_sections"] == list(range(n_fail)) and st["sections"] == len(secs)


# ---------------------------------------------------------------- review warnings (heuristics)
from src.candidates.heuristics import opening_warnings, promo_warnings, review_warnings  # noqa: E402
from src.candidates.generator import AUDIO_TAG_PEAK, AUDIO_TAG_SUSTAINED, audio_tags, make_blocks  # noqa: E402
from src.analysis.audio_analysis import AudioAnalysis, AudioEvent  # noqa: E402


@pytest.mark.parametrize("text,flagged", [
    ("And that is why the whole plan failed in the end.", True),
    ("As I mentioned earlier, the numbers were off.", True),
    ("He told me it was over and I just stood there.", True),
    ("और फिर उसने दरवाज़ा खोल दिया।", True),
    ("जैसा कि मैंने पहले कहा था, यह आसान नहीं है।", True),
    ("Aur phir usne darwaza khol diya.", True),
    ("The biggest mistake I ever made cost me everything.", False),
    ("मैंने कभी नहीं सोचा था कि ऐसा होगा।", False),
    ("", False),
])
def test_opening_warnings(text, flagged):
    assert bool(opening_warnings(text)) is flagged


@pytest.mark.parametrize("text,flagged", [
    ("This episode is sponsored by Acme, use code SAVE20 at checkout.", True),
    ("Please subscribe and check out the link, it is a free trial.", True),
    ("आज का एपिसोड स्पॉन्सर्ड है, प्रोमो कोड इस्तेमाल करें।", True),
    ("I told them to subscribe once, that was it, and the story moved on.", False),
    ("The merchant said the code was wrong, so we sat down and talked.", False),
])
def test_promo_warnings(text, flagged):
    assert bool(promo_warnings(text)) is flagged


def test_review_warnings_never_raise_on_odd_text():
    for t in ("", "   ", "!!!", "\n\t", "😀😀😀", "x" * 5000):
        assert isinstance(review_warnings(t), list)


# ---------------------------------------------------------------- audio tags on transcript lines
def _audio_with(events):
    return AudioAnalysis(window_s=0.5, duration=120.0, events=events)


def test_audio_tags_mark_only_lines_with_loud_events():
    t = make_transcript([(i * 20.0, f"This is line {i} of the talk and " + "it goes on for quite a while " * 3 + "now.") for i in range(10)])
    blocks = make_blocks(t.segments)
    assert len(blocks) >= 8
    a = _audio_with([AudioEvent(start=blocks[2].start + 1, end=blocks[2].start + 5, kind="sustained", strength=0.9),
                     AudioEvent(start=blocks[5].start + 1, end=blocks[5].start + 1.5, kind="peak", strength=0.95),
                     AudioEvent(start=blocks[7].start + 1, end=blocks[7].start + 1.5, kind="change_up", strength=0.9)])
    tags = audio_tags(blocks, a)
    assert tags[2] == AUDIO_TAG_SUSTAINED and tags[5] == AUDIO_TAG_PEAK
    assert tags[7] == "" and sum(1 for x in tags if x) == 2, "only loud events count; level changes do not"
    assert audio_tags(blocks, None) == [""] * len(blocks)
    assert audio_tags(blocks, AudioAnalysis(window_s=0.5, has_audio=False)) == [""] * len(blocks)


def test_audio_hints_reach_the_prompt_only_when_enabled_and_present(tmp_path):
    t = long_transcript(3)
    sec = make_sections(t, CFG.llm)[0]
    a = _audio_with([AudioEvent(start=sec.blocks[1].start + 1, end=sec.blocks[1].start + 6, kind="sustained", strength=0.9)])
    seen: list[str] = []

    def respond(system, prompt):
        seen.append(prompt)
        return json.dumps({"summary": "x"}) if "File name:" in prompt else json.dumps({"moments": []})

    off = CFG.with_overrides({"llm": {"audio_hints": False}})
    llm_candidates(t, off, ScriptedLLM(respond), "fp", audio=a)
    assert not any("[audio:" in p for p in seen)
    seen.clear()
    on = CFG.with_overrides({"llm": {"audio_hints": True}})
    _, st = llm_candidates(t, on, ScriptedLLM(respond), "fp", audio=a)
    moment_prompts = [p for p in seen if "Transcript section" in p]
    assert any(AUDIO_TAG_SUSTAINED in p and "measured from the recording" in p for p in moment_prompts) and st["hinted_sections"] >= 1
    seen.clear()
    llm_candidates(t, on, ScriptedLLM(respond), "fp", audio=_audio_with([]))
    assert not any("[audio:" in p or "measured from the recording" in p for p in seen), "no loud lines -> no tags and no note"
