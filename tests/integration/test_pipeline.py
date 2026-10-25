"""End-to-end pipeline on a generated MP4 with a fake Whisper and a scripted LLM (everything else is real)."""
import json
import re
import threading

import pytest

from config import load_config
from src.core.exceptions import CandidateValidationError, LLMUnavailableError
from src.core.exporter import build_summary, export_clips, write_summary
from src.core.job_manager import JobManager, STAGES, load_audio, load_transcript, merge_user_decisions
from src.core.project_manager import create_or_open_project
from src.rendering.cutting import ClipRenderer
from src.review import actions
from tests.conftest import needs_ffmpeg
from tests.fakes import ScriptedLLM
from tests.helpers import cand, make_transcript

pytestmark = [needs_ffmpeg]

SPEECH = [(2.0, "So here is the story of my strange day."), (5.0, "I opened the door and nobody was there at all."),
          (9.0, "Then the lights went out and everyone started to scream!"), (13.0, "It turned out the cat had pulled the plug."),
          (17.0, "Honestly I laughed for an hour about that."), (21.0, "That is all, thanks for listening.")]


class FakeWhisper:
    device, compute_type = "cpu", "int8"
    fallback_note = None

    def __init__(self):
        self.calls = 0

    def transcribe(self, wav, *, language, duration, on_progress, cancel):
        self.calls += 1
        assert wav.exists()
        return make_transcript(SPEECH, duration=24.0, word_dur=0.3, gap=0.05)

    def close(self):
        pass


def llm_responder(fail=False):
    def respond(system, prompt):
        if fail:
            raise LLMUnavailableError("Ollama is down")
        if "Transcript section" in prompt:
            return json.dumps({"moments": [{"start": 2.0, "end": 22.0, "category": "story", "title": "The lights went out",
                                            "explanation": "Setup and funny payoff.", "hook": "Then the lights went out and everyone started to scream!",
                                            "caption": "A strange day.", "context_completeness": 8, "payoff": 8, "confidence": 8}]})
        ids = re.findall(r"### Candidate (\S+) ", prompt)
        return json.dumps({"scores": [{"id": i, "hook": 8, "clarity": 7, "completeness": 8, "payoff": 8, "escalation": 7, "reason": "clear arc"} for i in ids]})
    return respond


@pytest.fixture()
def env(tmp_path, sample_info, runner):
    cfg = load_config(overrides={"render": {"width": 270, "height": 480, "preview_width": 270, "preview_height": 480}}, local_path=None)
    store = create_or_open_project(sample_info, tmp_path / "projects")
    return cfg, store, runner


def make_job(env, llm=None, whisper=None, cfg=None):
    c, store, runner = env
    return JobManager(store, cfg or c, runner, llm=llm or ScriptedLLM(llm_responder()), whisper=whisper or FakeWhisper())


def test_full_run_produces_scored_selected_candidates(env):
    job = make_job(env)
    progress = []
    rep = job.run(on_progress=lambda s, f, o, m: progress.append((s, o)))
    assert rep.success, rep.error
    assert rep.completed == STAGES and not rep.cached
    store = env[1]
    assert all(store.state.stages[s].status == "done" for s in STAGES)
    cands = store.load_candidates()
    sel = [c for c in cands if c.selected]
    assert len(sel) == 1
    c = sel[0]
    assert c.scores.llm_scored and 50 < c.total_score <= 100 and c.status == "pending"
    assert c.excerpt.startswith("So here is the story") and c.hook.startswith("Then the lights")
    assert c.category == "story" and c.start <= 2.0 and 21 <= c.end <= 24.0
    assert (store.cache_dir / "transcript.srt").exists() and (store.cache_dir / "transcript.json").exists()
    assert list(store.work_dir.iterdir()) == [], "temporary audio removed after a successful run"
    assert sorted(set(p[0] for p in progress)) == sorted(STAGES) and progress[-1][1] == pytest.approx(1.0)
    assert [o for _, o in progress] == sorted(o for _, o in progress), "overall progress is monotonic"
    assert load_audio(store).has_audio and load_transcript(store).language == "en"
    assert any(e.kind == "sustained" for e in load_audio(store).events), "loud burst at 10-14 s detected"


def test_resume_and_selective_invalidation(env):
    whisper, llm = FakeWhisper(), ScriptedLLM(llm_responder())
    job = make_job(env, llm, whisper)
    assert job.run().success
    calls = len(llm.calls)
    rep = make_job(env, llm, whisper).run()
    assert rep.success and rep.cached == STAGES and not rep.completed
    assert whisper.calls == 1 and len(llm.calls) == calls, "nothing recomputed on resume"
    # selection settings only -> only selection reruns
    cfg2 = load_config(overrides={**{"render": env[0].render.model_dump()}, "selection": {"max_clips": 3}}, local_path=None)
    rep = make_job(env, llm, whisper, cfg2).run()
    assert rep.completed == ["selection"] and rep.cached == STAGES[:-1]
    # scoring weights -> scoring + selection
    cfg3 = load_config(overrides={"scoring": {"weights": {"hook": 0.5}}}, local_path=None)
    rep = make_job(env, llm, whisper, cfg3).run()
    assert rep.completed == ["scoring", "selection"]
    # whisper model -> transcript and everything downstream; audio/scene analysis depend only on the video
    cfg4 = load_config(overrides={"transcription": {"model": "base"}}, local_path=None)
    rep = make_job(env, llm, whisper, cfg4).run()
    assert rep.completed == ["transcribe", "candidates", "scoring", "selection"] and rep.cached == ["audio", "scenes"] and whisper.calls == 2
    # llm model -> candidates onward, transcript/audio/scenes reused
    cfg5 = load_config(overrides={"transcription": {"model": "base"}, "llm": {"model": "qwen3:1.7b"}}, local_path=None)
    rep = make_job(env, llm, whisper, cfg5).run()
    assert rep.cached == ["transcribe", "audio", "scenes"] and rep.completed == ["candidates", "scoring", "selection"]


def test_failure_keeps_earlier_work_and_resumes(env):
    whisper = FakeWhisper()
    rep = make_job(env, ScriptedLLM(llm_responder(fail=True)), whisper).run()
    store = env[1]
    assert not rep.success and rep.failed_stage == "candidates" and "Ollama is down" in rep.error
    assert rep.recovery and "ollama" in rep.recovery.lower()
    st = store.state.stages
    assert [st[s].status for s in ("transcribe", "audio", "scenes")] == ["done"] * 3
    assert st["candidates"].status == "failed" and st["candidates"].error and store.state.errors
    rep2 = make_job(env, ScriptedLLM(llm_responder()), whisper).run()
    assert rep2.success and rep2.cached == ["transcribe", "audio", "scenes"] and whisper.calls == 1


def test_crash_mid_stage_is_reported_interrupted_then_rerun(env):
    job = make_job(env)
    assert job.run().success
    store = env[1]
    store.update_stage("scoring", status="running")          # simulate a killed process
    rep = make_job(env).run()
    assert rep.success and "scoring" in rep.completed and "transcribe" in rep.cached


def test_cancel_before_and_during(env):
    ev = threading.Event()
    ev.set()
    rep = make_job(env).run(cancel=ev)
    assert rep.cancelled and not rep.success and rep.recovery
    ev2 = threading.Event()
    whisper = FakeWhisper()
    orig = whisper.transcribe

    def cancelling(*a, **k):
        r = orig(*a, **k)
        ev2.set()
        return r

    whisper.transcribe = cancelling
    rep = make_job(env, whisper=whisper).run(cancel=ev2)
    assert rep.cancelled
    assert env[1].state.stages["transcribe"].status == "done", "work finished before the cancel is kept"


def test_user_decisions_survive_rerun(env):
    job = make_job(env)
    job.run()
    store = env[1]
    c = next(c for c in store.load_candidates() if c.selected)
    actions.set_status(store, [c.id], "approved")
    actions.update_title(store, c.id, "My custom title")
    cfg2 = load_config(overrides={"selection": {"min_score": 99}}, local_path=None)
    make_job(env, cfg=cfg2).run()
    again = store.get_candidate(c.id)
    assert again.status == "approved" and again.selected, "approved clips stay selected"
    cfg3 = load_config(overrides={"llm": {"model": "other:1b"}}, local_path=None)
    make_job(env, cfg=cfg3).run()
    kept = store.get_candidate(c.id)
    assert kept.status == "approved" and kept.title == "My custom title", "edited titles survive a re-run with a different model"


def test_merge_user_decisions_keeps_manual_and_edited():
    fresh = [cand("a", 0, 30, score=80)]
    live = [cand("a", 1, 29, score=70, status="approved", boundaries_edited=True), cand("m", 100, 130, source="manual"),
            cand("gone", 200, 230, status="pending"), cand("rej", 300, 330, status="rejected")]
    out = {c.id: c for c in merge_user_decisions(fresh, live)}
    assert out["a"].status == "approved" and (out["a"].start, out["a"].end) == (1, 29)
    assert "m" in out and "rej" in out and "gone" not in out


def test_heuristic_mode_without_llm_is_labelled(env):
    cfg = load_config(overrides={"llm": {"use_llm": False}}, local_path=None)
    llm = ScriptedLLM(["should not be called"])
    rep = make_job(env, llm, cfg=cfg).run()
    assert rep.success and llm.calls == []
    cands = env[1].load_candidates()
    assert cands and all(any("Heuristic" in w for w in c.warnings) or c.source == "audio" for c in cands)
    assert all(c.scores and not c.scores.llm_scored for c in cands)


def test_real_ollama_client_down_gives_actionable_failure(env):
    cfg = load_config(overrides={"llm": {"host": "http://127.0.0.1:9"}}, local_path=None)
    c, store, runner = env
    rep = JobManager(store, cfg, runner, whisper=FakeWhisper()).run()
    assert not rep.success and rep.failed_stage == "candidates"
    assert "ollama pull" in rep.recovery and "use_llm" in rep.recovery


def test_review_actions_and_no_render_on_approval(env):
    job = make_job(env)
    job.run()
    store, cfg = env[1], env[0]
    transcript, audio = load_transcript(store), load_audio(store)
    c = next(c for c in store.load_candidates() if c.selected)
    assert actions.set_status(store, [c.id], "approved") == 1
    assert not list(store.renders_dir.glob("*")), "approval alone must never render"
    assert actions.bulk_approve(store, 0) == [], "already approved ones are skipped"
    # bounds: invalid ones are rejected with a clear message, valid ones rescored
    for bad in [(-1, 10), (10, 10), (5, 999)]:
        with pytest.raises(CandidateValidationError):
            actions.update_bounds(store, c.id, *bad, transcript, audio, cfg)
    before = store.get_candidate(c.id)
    new, warns = actions.update_bounds(store, c.id, 3.0, 20.5, transcript, audio, cfg)
    assert new.boundaries_edited and (new.start, new.end) == (3.0, 20.5) and new.excerpt != before.excerpt
    assert new.scores.hook == before.scores.hook and any("Boundaries edited" in n for n in new.scores.notes)
    _, warns = actions.update_bounds(store, c.id, 3.0, 8.0, transcript, audio, cfg)
    assert any("Shorter" in w for w in warns)
    actions.update_bounds(store, c.id, 3.0, 20.5, transcript, audio, cfg)
    # subtitles
    text = actions.subtitle_editor_text(store.get_candidate(c.id), transcript, cfg)
    assert text.splitlines()[0].count("|") == 1
    edited = text.replace("strange", "weird")
    out = actions.update_subtitles(store, c.id, edited, transcript, cfg)
    assert out.subtitle_lines and any("weird" in l.text for l in out.subtitle_lines)
    with pytest.raises(ValueError, match="Line 1"):
        actions.update_subtitles(store, c.id, "garbage", transcript, cfg)
    # widening bounds after a subtitle edit keeps the edit and adds auto captions for the new part
    actions.update_bounds(store, c.id, 2.0, 22.5, transcript, audio, cfg)
    lines = store.get_candidate(c.id).subtitle_lines
    assert any("weird" in l.text for l in lines) and lines[-1].end > 20.5
    assert actions.reset_subtitles(store, c.id).subtitle_lines is None
    assert actions.update_title(store, c.id, "  New   title ").title == "New title"
    with pytest.raises(CandidateValidationError):
        actions.update_title(store, c.id, "   ")
    m = actions.add_manual_candidate(store, 4.0, 19.0, "Hand picked", transcript, audio, cfg, store.state.source.fingerprint)
    assert m.source == "manual" and store.get_candidate(m.id).selected


@pytest.mark.slow
def test_export_writes_all_artifacts_and_summary(env):
    job = make_job(env)
    job.run()
    cfg, store, runner = env
    transcript = load_transcript(store)
    c = next(c for c in store.load_candidates() if c.selected)
    actions.set_status(store, [c.id], "approved")
    renderer = ClipRenderer(store.state.source, store, cfg, runner)
    approved = [x for x in store.load_candidates() if x.status == "approved"]
    prog = []
    res = export_clips(store, cfg, renderer, transcript, approved, on_progress=lambda f, m: prog.append(f))
    assert len(res) == 1 and res[0]["ok"], res
    folder = next(store.exports_dir.iterdir())
    names = sorted(p.suffix for p in folder.iterdir())
    assert names == [".json", ".mp4", ".srt", ".txt"]
    meta = json.loads(next(folder.glob("*.json")).read_text())
    assert meta["title"] == "The lights went out" and meta["scores"]["total"] > 0 and "heuristic" in meta["note"]
    assert next(folder.glob("*.mp4")).stat().st_size > 5000 and prog[-1] == pytest.approx(1.0)
    again = export_clips(store, cfg, renderer, transcript, approved)
    assert again[0]["cached"], "re-export reuses the cached render"
    sm = write_summary(store)
    assert sm["candidates"]["approved"] == 1 and sm["timing_s"]["rendering"] > 0 and sm["timing_s"]["transcription"] >= 0
    assert (store.root / "summary.md").read_text().startswith("# Project summary")
    assert build_summary(store)["source"]["duration_s"] == 24.0
