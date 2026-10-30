
import pytest

from src.core.exceptions import JobCancelled, NoAudioError, TranscriptionError
from src.core.project_manager import create_or_open_project
from src.transcription.transcriber import FasterWhisperBackend, transcribe_video
from src.transcription.transcript_models import Transcript, TranscriptSegment, Word, nearest, srt_time
from config import load_config
from tests.conftest import needs_ffmpeg


def make_transcript() -> Transcript:
    def seg(i, start, words):
        ws, t = [], start
        for w in words.split():
            ws.append(Word(text=w, start=t, end=t + 0.4))
            t += 0.5
        return TranscriptSegment(id=i, start=start, end=t, text=words, words=ws)

    return Transcript(language="en", duration=30, segments=[
        seg(0, 0.0, "So here is the thing."),
        seg(1, 3.0, "I walked into the room and nobody was there"),
        seg(2, 8.0, "Then the lights went out!"),
        seg(3, 12.0, "Honestly that was wild."),
    ], has_word_timestamps=True)


class FakeBackend:
    device, compute_type = "cpu", "int8"
    fallback_note = "pretend fallback"

    def __init__(self, transcript=None, raises=None):
        self.t, self.raises, self.closed = transcript, raises, False

    def transcribe(self, wav, *, language, duration, on_progress, cancel):
        if self.raises:
            raise self.raises
        if on_progress:
            on_progress(0.5)
        return self.t

    def close(self):
        self.closed = True


def test_text_queries():
    t = make_transcript()
    assert t.text_in_range(3.0, 8.0).startswith("I walked into")
    assert "lights" not in t.text_in_range(3.0, 7.0)
    assert t.word_count_in_range(0, 3) == 5
    starts = t.sentence_starts()
    assert 0.0 in starts and 3.0 in starts and 12.0 in starts
    assert 8.0 not in starts, 'previous segment has no terminal punctuation, so this is not a sentence start'
    ends = t.sentence_ends()
    assert any(abs(e - 2.4) < 0.01 for e in ends), ends
    assert t.speech_coverage(0, 3) == pytest.approx(1.0 * (2.5 / 3), abs=0.01) or t.speech_coverage(0, 3) > 0.7
    assert t.speech_coverage(5, 5) == 0


def test_srt_format():
    assert srt_time(3661.5) == "01:01:01,500"
    assert srt_time(-1) == "00:00:00,000"
    out = make_transcript().to_srt()
    assert out.startswith("1\n00:00:00,000 --> ") and "So here is the thing." in out


def test_nearest():
    assert nearest([1.0, 5.0, 9.0], 6.0, lo=0, hi=10) == 5.0
    assert nearest([1.0, 5.0, 9.0], 6.0, lo=7, hi=10) == 9.0
    assert nearest([1.0], 6.0, lo=3, hi=4) is None


@needs_ffmpeg
def test_transcribe_stage_caches_and_writes_srt(tmp_path, sample_info, runner):
    cfg = load_config(local_path=None)
    store = create_or_open_project(sample_info, tmp_path)
    be = FakeBackend(make_transcript())
    t, cached = transcribe_video(sample_info, store, cfg, runner, backend=be)
    assert not cached and len(t.segments) == 4
    assert (store.cache_dir / "transcript.srt").read_text(encoding="utf-8").startswith("1\n")
    assert (store.work_dir / "audio_16k.wav").exists()
    assert store.state.warnings == ["pretend fallback"]
    # second call: served from cache, backend must not be touched
    be2 = FakeBackend(raises=RuntimeError("should not be called"))
    t2, cached2 = transcribe_video(sample_info, store, cfg, runner, backend=be2)
    assert cached2 and t2 == t
    # changing the model invalidates
    cfg2 = load_config(overrides={"transcription": {"model": "base"}}, local_path=None)
    _, cached3 = transcribe_video(sample_info, store, cfg2, runner, backend=FakeBackend(make_transcript()))
    assert not cached3


@needs_ffmpeg
def test_no_speech_and_no_audio_errors(tmp_path, sample_info, silent_video, runner):
    from src.ingestion.video_probe import inspect_video

    cfg = load_config(local_path=None)
    store = create_or_open_project(sample_info, tmp_path)
    with pytest.raises(TranscriptionError, match="No speech"):
        transcribe_video(sample_info, store, cfg, runner, backend=FakeBackend(Transcript(language="en", segments=[])))
    silent = inspect_video(silent_video, runner)
    store2 = create_or_open_project(silent, tmp_path)
    with pytest.raises(NoAudioError):
        transcribe_video(silent, store2, cfg, runner, backend=FakeBackend(make_transcript()))


@needs_ffmpeg
def test_cancel_propagates(tmp_path, sample_info, runner):
    cfg = load_config(local_path=None)
    store = create_or_open_project(sample_info, tmp_path)
    with pytest.raises(JobCancelled):
        transcribe_video(sample_info, store, cfg, runner, backend=FakeBackend(raises=JobCancelled("x")))


def test_backend_plan_without_cuda(monkeypatch):
    # On CPU-only machines auto must plan CPU int8 and say nothing is wrong.
    monkeypatch.setattr("ctranslate2.get_cuda_device_count", lambda: 0)
    be = FasterWhisperBackend(load_config(local_path=None), 2)
    plan = be._plan()
    assert plan[-1][0] == "cpu" and plan[-1][1] == "int8"
    forced = FasterWhisperBackend(load_config(overrides={"transcription": {"device": "cuda"}}, local_path=None), 2)
    plan = forced._plan()
    assert all(d == "cpu" for d, _ in plan) and forced.fallback_note, "must tell the user CUDA was unavailable"


def test_backend_missing_model_gives_actionable_error():
    be = FasterWhisperBackend(load_config(overrides={"transcription": {"model": "tiny"}}, local_path=None), 2)
    with pytest.raises(TranscriptionError) as ei:
        be._load()
    assert "download_models" in ei.value.recovery or "not installed" in str(ei.value)
