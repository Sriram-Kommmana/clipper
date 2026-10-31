import json

import numpy as np
import pytest

from config import load_config
from src.analysis.audio_analysis import analyze_rms
from src.analysis.frame_extraction import VisualCheck
from src.candidates.generator import (BLOCK_MAX_S, audio_candidates, candidate_id, first_sentence, generate_candidates, heuristic_candidates,
                                      hook_is_quote, llm_candidates, make_blocks, make_sections, moment_to_candidate, video_brief,
                                      widen_to_sentences)
from src.candidates.merger import merge_candidates, overlap_ratio
from src.candidates.scorer import boundary_quality, compute_breakdown, format_fit, score_candidates, speech_density_score
from src.candidates.selector import select_candidates
from src.candidates.validator import Limits, check_bounds, hard_failures, refine_boundaries, validate_manual_bounds
from src.core.exceptions import CandidateValidationError, LLMOutputError
from src.core.schemas import RawMoment
from tests.fakes import ScriptedLLM
from tests.helpers import cand, long_transcript, make_transcript

CFG = load_config(local_path=None)
LIM = Limits.from_config(CFG.selection)


def moment(first_line, last_line, **kw):
    d = dict(first_line=first_line, last_line=last_line, category="story", title="T", explanation="E", hook="", caption="C",
             context_completeness=8, payoff=7, confidence=7)
    d.update(kw)
    return d


# ---------------------------------------------------------------- sections
def test_sections_cover_transcript_with_overlap_and_limits():
    t = long_transcript(30)
    secs = make_sections(t, CFG.llm)
    assert len(secs) >= 3
    assert secs[0].start == 0 and secs[-1].end == pytest.approx(t.segments[-1].end)
    for a, b in zip(secs, secs[1:]):
        assert b.start < a.end, "consecutive sections must overlap"
        assert b.start > a.start, "and make progress"
    assert all(s.end - s.start <= CFG.llm.section_seconds + 10 for s in secs)
    assert all(sum(len(x.text.split()) for x in s.segments) <= CFG.llm.section_max_words for s in secs)
    seen = {x.id for s in secs for x in s.segments}
    assert seen == {x.id for x in t.segments}, "every segment appears in at least one section"
    assert secs[0].render().startswith("0 [0:00] This is sentence number 0")


def test_sections_word_cap_and_tiny_transcript():
    t = make_transcript([(0, "one two three four five six seven eight nine ten."), (5, "eleven twelve.")])
    assert len(make_sections(t, CFG.llm)) == 1
    small = CFG.llm.model_copy(update={"section_max_words": 100, "section_seconds": 60, "section_overlap_s": 10})
    assert len(make_sections(long_transcript(10), small)) > 5
    empty = make_transcript([(0, "x")]).model_copy(update={"segments": []})
    assert make_sections(empty, CFG.llm) == []


# ---------------------------------------------------------------- moment verification
def test_hook_verification():
    ex = "So I walked in, and the lights went out! Nobody moved."
    assert hook_is_quote("the lights went out!", ex)
    assert hook_is_quote("The LIGHTS went out", ex)
    assert not hook_is_quote("a dragon appeared", ex)
    assert not hook_is_quote("", ex)
    assert first_sentence("Hello there. More text.") == "Hello there."
    assert first_sentence("मैं वहाँ गया। फिर लौटा।") == "मैं वहाँ गया।"


def test_moment_to_candidate_uses_transcript_not_model_text():
    t = make_transcript([(0, "So I walked into the room."), (4, "The lights went out!"), (8, "Everyone screamed loudly.")] , duration=60)
    sec = make_sections(t, CFG.llm)[0]
    m = RawMoment(**moment(0, 2, hook="a fabricated hook about dragons", title="x"))
    c, why = moment_to_candidate(m, sec, t, load_config(overrides={"selection": {"min_duration_s": 6, "target_duration_s": 8}}, local_path=None), "fp")
    assert why is None and "lights went out" in c.excerpt
    assert c.hook == "So I walked into the room." and any("verbatim" in w for w in c.warnings)
    assert c.confidence == 70 and c.id == candidate_id("fp", 0, c.end, "story") and c.end == sec.end
    ok = RawMoment(**moment(0, 2, hook="The lights went out!"))
    c2, _ = moment_to_candidate(ok, sec, t, load_config(overrides={"selection": {"min_duration_s": 6, "target_duration_s": 8}}, local_path=None), "fp")
    assert c2.hook == "The lights went out!" and not c2.warnings


def test_moment_drops():
    t = long_transcript(5)
    sec = make_sections(t, CFG.llm)[0]
    assert moment_to_candidate(RawMoment(**moment(0, 46)), sec, t, CFG, "fp")[0] is None       # way too long
    assert moment_to_candidate(RawMoment(**moment(500, 520)), sec, t, CFG, "fp")[0] is None    # outside section
    assert moment_to_candidate(RawMoment(**moment(9, 4)), sec, t, CFG, "fp")[0] is None        # reversed range
    c, _ = moment_to_candidate(RawMoment(**moment(-5, 4)), sec, t, CFG, "fp")
    assert c.start == 0, "lines are clamped into the section"
    assert c.end == sec.blocks[4].end, "the clip ends with its last line"


def test_short_moment_is_extended_forward_not_dropped():
    t = long_transcript(5)
    sec = make_sections(t, CFG.llm)[0]
    c, why = moment_to_candidate(RawMoment(**moment(8, 8)), sec, t, CFG, "fp")
    assert why is None and c.start == sec.blocks[8].start, "the chosen line stays the opening"
    assert c.duration >= CFG.selection.min_duration_s and any("extended" in w for w in c.warnings)
    tail, _ = moment_to_candidate(RawMoment(**moment(len(sec.blocks) - 1, len(sec.blocks) - 1)), sec, t, CFG, "fp")
    assert tail.duration >= CFG.selection.min_duration_s and tail.end == sec.blocks[-1].end, "at the end it grows backwards"


def test_blocks_group_segments_at_sentence_edges():
    segs = make_transcript([(0, "One two three"), (2, "four five six."), (4, "seven eight nine."), (8, "ten eleven twelve।"), (30, "alone")]).segments
    blocks = make_blocks(segs)
    assert blocks[0].text == "One two three four five six. seven eight nine. ten eleven twelve।", "a Hindi danda also ends a sentence"
    assert blocks[-1].text == "alone" and blocks[0].end == segs[3].end
    overlong = make_transcript([(i * 3.0, f"words without an end {i}") for i in range(10)]).segments
    assert all(b.end - b.start <= BLOCK_MAX_S + 3 for b in make_blocks(overlong)), "no sentence end still gives bounded lines"


def test_video_brief_uses_title_and_survives_failure():
    t = long_transcript(5)
    llm = ScriptedLLM([json.dumps({"summary": "A talk about topics."})])
    assert video_brief(t, llm, "My Great Talk") == "A talk about topics."
    assert "My Great Talk" in llm.calls[0][1] and "sentence number" in llm.calls[0][1]
    assert video_brief(t, ScriptedLLM(["junk"])) == ""


def test_prompts_receive_the_video_brief():
    t = long_transcript(5)

    def respond(system, prompt):
        return json.dumps({"summary": "BRIEF TEXT"}) if "File name:" in prompt else json.dumps({"moments": [moment(2, 5)]})

    llm = ScriptedLLM(respond)
    _, stats = llm_candidates(t, CFG, llm, "fp", title="Talk")
    assert stats["video_brief"] == "BRIEF TEXT"
    assert any("About this video: BRIEF TEXT" in prompt for _, prompt in llm.calls)


# ---------------------------------------------------------------- LLM generation
def test_llm_generation_merges_duplicates_across_sections():
    t = long_transcript(30)
    secs = make_sections(t, CFG.llm)
    boundary = secs[1].start + 20           # lies in the overlap zone of sections 0 and 1
    assert boundary < secs[0].end

    def respond(system, prompt):
        # every section that can see [boundary, boundary+35] reports it (slightly different bounds)
        for i, s in enumerate(secs):
            if f"({s.start:.0f}s - {s.end:.0f}s)" in prompt:
                if s.start <= boundary and boundary + 35 <= s.end:
                    first = next(n for n, b in enumerate(s.blocks) if b.start >= boundary)
                    return json.dumps({"moments": [moment(first + (i % 2), first + 3 - (i % 2))]})
        return json.dumps({"summary": "A talk."}) if "File name:" in prompt else json.dumps({"moments": []})

    llm = ScriptedLLM(respond)
    cands, stats = generate_candidates(t, None, CFG, llm, "fp", t.duration)
    assert stats["sections"] == len(secs) and stats["raw_moments"] >= 2
    assert len(cands) == 1 and stats["merged"] >= 1 and cands[0].merged_from
    assert len(llm.calls) == len(secs) + 1, "one call per section, plus the video brief"


def test_llm_generation_survives_bad_section_but_not_all():
    t = long_transcript(30)
    n = len(make_sections(t, CFG.llm))
    seq = [json.dumps({"summary": "A talk."})] + ["garbage"] * 3 + [json.dumps({"moments": [moment(10, 15)]})] * 50
    cands, stats = llm_candidates(t, CFG, ScriptedLLM(seq), "fp")
    assert stats["failed_sections"] == [0] and stats["sections"] == n
    with pytest.raises(LLMOutputError):
        llm_candidates(t, CFG, ScriptedLLM(["garbage"]), "fp")


def test_heuristic_fallback_is_flagged():
    t = long_transcript(5)
    cs = heuristic_candidates(t, None, CFG, "fp", t.duration)
    assert cs and all("Heuristic" in c.warnings[0] for c in cs)
    assert all(CFG.selection.min_duration_s <= c.duration <= CFG.selection.max_duration_s for c in cs)
    assert not any(a.end > b.start for a in sorted(cs, key=lambda c: c.start) for b in sorted(cs, key=lambda c: c.start) if a is not b and a.start < b.start)
    noll, st = generate_candidates(t, None, load_config(overrides={"llm": {"use_llm": False}}, local_path=None), None, "fp", t.duration)
    assert st["mode"] == "heuristic" and noll


# ---------------------------------------------------------------- audio supplements
def audio_with_burst():
    rms = np.full(240, 0.01)       # 120 s
    rms[100:116] = 0.5             # 50-58 s sustained loud
    return analyze_rms(rms, 120.0, CFG.audio)


def test_audio_candidates_supplement_only_uncovered_events():
    t = long_transcript(2)
    a = audio_with_burst()
    out = audio_candidates(t, a, CFG, [], "fp", 120.0)
    assert len(out) == 1 and out[0].source == "audio" and out[0].category == "reaction"
    c = out[0]
    assert c.start <= 50 - CFG.boundaries.reaction_lead_in_s + 0.6 and c.end >= 58
    assert c.duration >= CFG.selection.min_duration_s * 0.9 and c.confidence <= 30
    covered = cand("x", 40, 70)
    assert audio_candidates(t, a, CFG, [covered], "fp", 120.0) == []
    assert audio_candidates(t, None, CFG, [], "fp", 120.0) == []


def test_audio_candidates_never_cut_a_sentence():
    words = " ".join(f"word{i}" for i in range(39))        # a 17.5 s sentence, longer than boundary snapping reaches
    t = make_transcript([(0, "Opening line."), (5, f"{words}."), (40, f"{words}."), (59, f"{words}.")], duration=120)
    out = audio_candidates(t, audio_with_burst(), CFG, [], "fp", 120.0)    # burst at 50-58 s: raw window is inside both long sentences
    assert len(out) == 1
    assert out[0].excerpt.startswith("word0") and out[0].excerpt.endswith("word38.")
    assert (out[0].start, out[0].end) == (t.segments[2].start, t.segments[3].end)


def test_widen_to_sentences_keeps_edges_between_sentences():
    t = make_transcript([(0, "One two three."), (10, "Four five six seven.")], duration=30)
    assert widen_to_sentences(t, 0.5, 11.0) == (0, t.segments[1].end), "edges inside sentences move out"
    assert widen_to_sentences(t, 5.0, 7.0) == (5.0, 7.0), "the silence between sentences is left alone"
    assert widen_to_sentences(t, 20.0, 25.0) == (20.0, 25.0)


def test_audio_candidate_too_long_to_cut_cleanly_is_skipped():
    words = " ".join(f"word{i}" for i in range(240))       # 108 s without a sentence end
    t = make_transcript([(0, f"{words}.")], duration=120)
    assert audio_candidates(t, audio_with_burst(), CFG, [], "fp", 120.0) == []


# ---------------------------------------------------------------- merger
def test_overlap_and_merge_rules():
    a, b = cand("a", 0, 30), cand("b", 20, 50)
    assert overlap_ratio(a, b) == pytest.approx(10 / 30)
    assert overlap_ratio(cand("o", 0, 60), cand("i", 10, 20)) == 1.0
    merged, n = merge_candidates([a, b], 72)
    assert n == 0 and len(merged) == 2                       # 33% < 50%: distinct
    merged, n = merge_candidates([cand("a", 0, 30), cand("b", 10, 40)], 72)
    assert n == 1 and len(merged) == 1 and (merged[0].start, merged[0].end) == (0, 40)
    # union would exceed the maximum -> keep the stronger one unchanged
    strong = cand("s", 0, 50, confidence=90)
    weak = cand("w", 10, 80, confidence=10)
    m, _ = merge_candidates([strong, weak], 72)
    assert len(m) == 1 and m[0].id == "s" and (m[0].start, m[0].end) == (0, 50) and "w" in m[0].merged_from
    # chain: a~b, b~c collapses to one
    m, n = merge_candidates([cand("a", 0, 30), cand("b", 15, 45), cand("c", 30, 60)], 72)
    assert len(m) == 1 and (m[0].start, m[0].end) == (0, 60)
    # LLM beats audio-only as primary
    m, _ = merge_candidates([cand("l", 0, 30, source="llm", confidence=20), cand("au", 0, 30, source="audio", confidence=90)], 72)
    assert m[0].id == "l"
    assert merge_candidates([], 72) == ([], 0)


# ---------------------------------------------------------------- validator
@pytest.mark.parametrize("s,e,ok", [(0, 10, True), (-1, 10, False), (10, 10, False), (10, 5, False), (0, 200, False),
                                    (101, 105, False), (float("nan"), 5, False), (0, float("inf"), False), (5, 100.04, True)])
def test_check_bounds(s, e, ok):
    if ok:
        check_bounds(s, e, 100.0)
    else:
        with pytest.raises(CandidateValidationError):
            check_bounds(s, e, 100.0)


def test_manual_bounds_limits_and_warnings():
    assert validate_manual_bounds(0, 40, 100, LIM) == []
    assert any("Longer" in w for w in validate_manual_bounds(0, 95, 100, LIM))
    assert any("Shorter" in w for w in validate_manual_bounds(0, 8, 100, LIM))
    with pytest.raises(CandidateValidationError, match="at least"):
        validate_manual_bounds(0, 1, 100, LIM)
    with pytest.raises(CandidateValidationError, match="at most"):
        validate_manual_bounds(0, 190, 300, LIM)


def story_transcript():
    return make_transcript([
        (0.0, "Welcome back everyone."), (2.5, "Today I want to tell you a story."),
        (6.0, "Last year I got lost in the mountains"), (10.0, "and I found a strange cabin."),
        (14.0, "Inside, a man was waiting for me."), (18.0, "He smiled and said you are late!"),
        (22.0, "I never forgot that moment."), (26.0, "Anyway, that is the story."), (30.0, "Thanks for watching, see you next week.")
    ], duration=60)


def test_refine_snaps_to_sentence_edges_and_pads_safely():
    t = story_transcript()
    c = cand("c1", 6.9, 24.0, category="story")        # starts mid-sentence, ends mid-sentence
    r = refine_boundaries(c, t, CFG.boundaries, Limits(10, 60, 1.2), 60.0)
    first_word_of_sentence = next(w for w in t.words() if w.text == "Last")
    assert r.start <= first_word_of_sentence.start and r.start >= first_word_of_sentence.start - 0.3
    prev_end = max(w.end for w in t.words() if w.end <= first_word_of_sentence.start)
    assert r.start > prev_end - 1e-6, "padding must not reach into the previous word"
    assert r.excerpt.startswith("Last year") and r.excerpt.rstrip().endswith("moment.")
    assert not any(w.start < r.start < w.end for w in t.words()), "never starts inside a word"
    assert not any(w.start < r.end < w.end for w in t.words()), "never ends inside a word"


def test_refine_without_sentence_edges_never_cuts_words():
    t = make_transcript([(0.0, "no punctuation at all here just words flowing on and on forever and ever")], duration=30)
    w = t.words()[3]
    c = cand("c", w.start + 0.1, t.words()[9].end - 0.05)   # inside words at both ends
    r = refine_boundaries(c, t, CFG.boundaries, Limits(2, 60, 1.2), 30.0)
    assert not any(x.start < r.start < x.end for x in t.words())
    assert not any(x.start < r.end < x.end for x in t.words())


def test_refine_trims_overlong_to_sentence_end_and_flags():
    lines = [(i * 5.0, f"Sentence number {i} is right here.") for i in range(30)]
    t = make_transcript(lines, duration=200)
    c = cand("c", 0.0, 120.0)
    r = refine_boundaries(c, t, CFG.boundaries, Limits(15, 60, 1.2), 200.0)
    assert r.duration <= 61 and any("Trimmed" in w for w in r.warnings)
    c2 = cand("c2", 0.0, 66.0)
    r2 = refine_boundaries(c2, t, CFG.boundaries, Limits(15, 60, 1.2), 200.0)
    assert any("Longer" in w or "Trimmed" in w for w in r2.warnings)


def test_hard_failures():
    t = story_transcript()
    ok = cand("a", 6.0, 24.0)
    assert hard_failures(ok, t, None, 60.0, Limits(10, 60, 1.2)) == []
    assert any("too short" in r for r in hard_failures(cand("b", 6, 12), t, None, 60.0, Limits(15, 60, 1.2)))
    assert any("too long" in r for r in hard_failures(cand("c", 0, 80), t, None, 100.0, Limits(15, 60, 1.2)))
    assert any("beyond" in r for r in hard_failures(cand("d", 50, 70), t, None, 60.0, LIM))
    nospeech = cand("e", 40, 58)
    assert any("almost no speech" in r for r in hard_failures(nospeech, t, None, 60.0, Limits(15, 60, 1.2)))
    loud = analyze_rms(np.where(np.arange(120) > 70, 0.5, 0.01), 60.0, CFG.audio)
    audio_c = cand("f", 40, 58, source="audio")
    assert hard_failures(audio_c, t, loud, 60.0, Limits(15, 60, 1.2)) == [], "audio-sourced clips may lack speech if energetic"


# ---------------------------------------------------------------- scorer
def test_format_fit_curve():
    s = CFG.selection
    f = lambda d: format_fit(d, s.target_duration_s, s.min_duration_s, s.max_duration_s)
    assert f(45) == pytest.approx(100)
    assert 55 <= f(15) <= 70 and 30 <= f(90) <= 40
    assert f(10) <= 40 and f(100) <= 40 and f(100) >= 0
    assert f(35) > f(25) > f(18)


def test_speech_density():
    assert speech_density_score(0, 30) == 0
    assert speech_density_score(75, 30) == 100            # 2.5 wps
    assert speech_density_score(30, 30) == 50             # 1 wps
    assert speech_density_score(240, 30) < 80             # 8 wps


def test_boundary_quality():
    t = story_transcript()
    good = cand("g", 5.9, 24.2)
    assert boundary_quality(good, t, tol=1.0) == 100
    bad = cand("b", 8.0, 20.0)
    assert boundary_quality(bad, t, tol=0.5) == 0


def test_breakdown_weights_and_fallback():
    t = story_transcript()
    c = cand("c", 6.0, 24.0, context_completeness=80, payoff_estimate=60)
    vals = {"hook": 90.0, "clarity": 80.0, "completeness": 70.0, "payoff": 60.0, "escalation": 50.0, "_reason": "good setup"}
    bd = compute_breakdown(c, vals, t, None, None, CFG)
    assert bd.llm_scored and "good setup" in bd.notes
    w = CFG.scoring.weights
    expect = sum(w[k] * getattr(bd, k) for k in w) / sum(w.values())
    assert bd.total == pytest.approx(expect, abs=0.11)
    assert 0 <= bd.total <= 100 and bd.hook == 90
    fb = compute_breakdown(c, None, t, None, None, CFG)
    assert not fb.llm_scored and any("unavailable" in n for n in fb.notes)
    better = compute_breakdown(c, {**vals, "hook": 100.0, "payoff": 100.0}, t, None, None, CFG)
    assert better.total > bd.total
    bad_vis = compute_breakdown(c, vals, t, None, VisualCheck(score=0, warnings=["Mostly black/dark frames."]), CFG)
    assert bad_vis.total < bd.total and any("black" in n for n in bad_vis.notes)


def test_llm_scoring_batches_and_ranking():
    t = long_transcript(10)
    cands = [cand(f"c{i}", 60 + i * 60, 100 + i * 60, excerpt=f"text {i}", context_completeness=50, payoff_estimate=50) for i in range(6)]

    def respond(system, prompt):
        ids = [f"c{i}" for i in range(6) if f"Candidate c{i} " in prompt]
        rows = [{"id": i, "hook": 9 if i == "c3" else 4, "clarity": 6, "completeness": 6, "payoff": 6, "escalation": 5, "reason": "r"} for i in ids]
        rows.append({"id": "ghost", "hook": 10, "clarity": 10, "completeness": 10, "payoff": 10, "escalation": 10, "reason": ""})
        return json.dumps({"scores": rows})

    llm = ScriptedLLM(respond)
    out = score_candidates(cands, t, None, CFG, llm)
    assert len(llm.calls) == 2                                           # 6 candidates / batch of 4
    assert all(c.scores and c.scores.llm_scored for c in out)
    assert max(out, key=lambda c: c.total_score).id == "c3"
    assert "ghost" not in {c.id for c in out}
    # failing LLM -> still scored, flagged as not llm-scored
    out2 = score_candidates(cands, t, None, CFG, ScriptedLLM(["junk"]))
    assert all(c.scores and not c.scores.llm_scored for c in out2)
    # visual function: only top candidates are checked; exceptions are swallowed
    seen = []

    def vis(c):
        seen.append(c.id)
        if c.id == "c0":
            raise RuntimeError("boom")
        return VisualCheck(score=100)

    out3 = score_candidates(cands, t, None, CFG, ScriptedLLM(respond), visual_fn=vis)
    assert len(out3) == 6 and seen


# ---------------------------------------------------------------- selector
SEL = CFG.selection


def test_selector_filters_and_limits():
    cands = [cand("a", 0, 30, score=90), cand("b", 100, 130, score=80, category="funny"), cand("c", 200, 230, score=30),
             cand("d", 300, 330, score=70, category="quote"), cand("e", 400, 430), ]
    out = {c.id: c for c in select_candidates(cands, SEL)}
    assert out["a"].selected and out["b"].selected and out["d"].selected
    assert "below minimum" in out["c"].filter_reason and out["e"].filter_reason == "not scored"
    small = SEL.model_copy(update={"max_clips": 2})
    out = {c.id: c for c in select_candidates(cands, small)}
    assert sorted(i for i, c in out.items() if c.selected) == ["a", "b"]
    assert "maximum of 2" in out["d"].filter_reason


def test_selector_separation_and_overlap():
    cands = [cand("a", 0, 30, score=90), cand("b", 35, 65, score=85), cand("c", 20, 50, score=80), cand("d", 100, 130, score=60)]
    strict = SEL.model_copy(update={"min_separation_s": 15})
    out = {c.id: c for c in select_candidates(cands, strict)}
    assert out["a"].selected and out["d"].selected
    assert not out["b"].selected and "too close" in out["b"].filter_reason
    assert not out["c"].selected
    loose = SEL.model_copy(update={"min_separation_s": 0})
    out = {c.id: c for c in select_candidates(cands, loose)}
    assert out["b"].selected and not out["c"].selected, "overlap is never allowed"


def test_selector_diversity_and_category_preference():
    cands = [cand(f"s{i}", i * 100, i * 100 + 30, score=80 - i, category="story") for i in range(3)] + [cand("f", 400, 430, score=70, category="funny")]
    div = SEL.model_copy(update={"max_clips": 2, "diversity_penalty": 15})
    out = {c.id: c for c in select_candidates(cands, div)}
    assert out["s0"].selected and out["f"].selected and not out["s1"].selected
    nodiv = SEL.model_copy(update={"max_clips": 2, "diversity_penalty": 0})
    out = {c.id: c for c in select_candidates(cands, nodiv)}
    assert out["s0"].selected and out["s1"].selected
    pref = SEL.model_copy(update={"max_clips": 1, "diversity_penalty": 0, "category_preferences": {"funny": 1.3}})
    assert next(c for c in select_candidates(cands, pref) if c.selected).id == "f"


def test_selector_respects_user_decisions_and_prior_invalid():
    cands = [cand("a", 0, 30, score=90, status="rejected"), cand("b", 100, 130, score=20, status="approved"),
             cand("x", 200, 230, score=95, filter_reason="invalid: too short")]
    out = {c.id: c for c in select_candidates(cands, SEL)}
    assert not out["a"].selected and out["a"].filter_reason == "rejected by user"
    assert out["b"].selected, "approved clips stay even below min_score"
    assert not out["x"].selected and out["x"].filter_reason.startswith("invalid")
    again = {c.id: c for c in select_candidates(list(out.values()), SEL)}
    assert again["b"].selected and not again["a"].selected, "re-running selection is stable"
