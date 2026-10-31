import pytest

from config import load_config
from src.core.exceptions import TranscriptionError
from src.core.job_manager import JobManager, load_transcript
from src.core.project_manager import create_or_open_project
from src.transcription.importer import load_import, parse_stamp, parse_transcript, store_import, stored_import
from src.transcription.transcriber import transcript_cache_key
from tests.conftest import needs_ffmpeg

SRT = """1
00:00:01,000 --> 00:00:04,000
<b>Hello</b> there my friend.

2
00:00:05,500 --> 00:00:08,000
{\\an8}मैं वहाँ गया।
"""

YOUTUBE_VTT = """WEBVTT
Kind: captions
Language: en

00:00:00.200 --> 00:00:02.100 align:start position:0%
 
hello<00:00:00.500><c> there</c><00:00:00.900><c> friend</c>

00:00:02.100 --> 00:00:02.110 align:start position:0%
hello there friend
 

00:00:02.110 --> 00:00:04.000 align:start position:0%
hello there friend
how<00:00:02.600><c> are</c><00:00:03.000><c> you</c>
"""


def test_parse_stamp_formats():
    assert parse_stamp("00:00:01,500") == 1.5 and parse_stamp("01:02.250") == 62.25 and parse_stamp("1:00:00.000") == 3600


def test_srt_cleans_tags_and_estimates_word_timing():
    t, notes = parse_transcript(SRT, 60.0)
    assert [s.text for s in t.segments] == ["Hello there my friend.", "मैं वहाँ गया।"]
    assert t.model == "imported" and not t.has_word_timestamps and any("estimated" in n for n in notes)
    assert t.mean_word_confidence() is None, "imported text has no recogniser confidence to judge"
    words = t.segments[0].words
    assert words[0].start == 1.0 and words[-1].end == pytest.approx(4.0) and all(a.end <= b.start + 1e-6 for a, b in zip(words, words[1:]))
    assert t.sentence_ends() == [pytest.approx(4.0), pytest.approx(8.0)], "a danda ends a sentence"
    assert t.text_in_range(0, 3).startswith("Hello")


def test_youtube_vtt_drops_rolling_repeats_and_keeps_exact_word_times():
    t, notes = parse_transcript(YOUTUBE_VTT, 60.0)
    assert [s.text for s in t.segments] == ["hello there friend", "how are you"]
    assert t.has_word_timestamps and not any("estimated" in n for n in notes)
    assert [(w.text, w.start) for w in t.segments[0].words] == [("hello", 0.2), ("there", 0.5), ("friend", 0.9)]
    assert t.segments[1].words[-1].end == 4.0


def test_devanagari_text_is_recognised_as_hindi_and_setting_wins():
    assert parse_transcript("00:00:01.000 --> 00:00:02.000\nमैं वहाँ गया और वापस आया\n", 10.0)[0].language == "hi"
    assert parse_transcript("00:00:01.000 --> 00:00:02.000\nhello\n", 10.0)[0].language == "unknown"
    assert parse_transcript("00:00:01.000 --> 00:00:02.000\nhello\n", 10.0, "en")[0].language == "en"


def test_plain_text_is_refused_with_the_reason():
    with pytest.raises(TranscriptionError) as ei:
        parse_transcript("just some words with no timing at all", 60.0)
    assert "timestamps" in str(ei.value) and ".srt" in ei.value.recovery


def test_timings_that_do_not_fit_the_video_are_caught():
    with pytest.raises(TranscriptionError, match="inside this video"):
        parse_transcript("00:10:00,000 --> 00:10:02,000\nToo late\n", 60.0)
    late = "00:00:01,000 --> 00:00:02,000\nFine\n\n00:10:00,000 --> 00:10:02,000\nToo late\n"
    t, notes = parse_transcript(late, 60.0)
    assert len(t.segments) == 1 and any("after the end" in n for n in notes) and any("stops at" in n for n in notes)


def test_overlapping_cues_are_kept_in_order():
    t, _ = parse_transcript("00:00:01,000 --> 00:00:05,000\nFirst line\n\n00:00:04,000 --> 00:00:07,000\nSecond line\n", 60.0)
    assert t.segments[1].start == t.segments[0].end == 5.0


def test_file_encodings(tmp_path):
    for name, data in (("a.srt", SRT.encode("utf-8-sig")), ("b.srt", SRT.encode("utf-16"))):
        path = tmp_path / name
        path.write_bytes(data)
        assert load_import(path, 60.0)[0].segments[0].text == "Hello there my friend."
    bad = tmp_path / "c.srt"
    bad.write_bytes(SRT.encode("utf-8") + b"\xff\xfe\xff")
    with pytest.raises(TranscriptionError, match="UTF-8"):
        load_import(bad, 60.0)


def test_store_import_validates_and_replaces(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    assert stored_import(root) is None
    with pytest.raises(TranscriptionError, match="not found"):
        store_import(tmp_path / "missing.srt", root)
    txt = tmp_path / "t.txt"
    txt.write_text("words", encoding="utf-8")
    with pytest.raises(TranscriptionError, match="Unsupported"):
        store_import(txt, root)
    srt = tmp_path / "t.srt"
    srt.write_text(SRT, encoding="utf-8")
    assert store_import(f'"{srt}"', root).name == "imported_transcript.srt"
    vtt = tmp_path / "t.vtt"
    vtt.write_text(YOUTUBE_VTT, encoding="utf-8")
    store_import(vtt, root)
    assert stored_import(root).suffix == ".vtt" and not (root / "imported_transcript.srt").exists()


class FailingWhisper:
    device, compute_type, fallback_note = "cpu", "int8", None

    def transcribe(self, *a, **k):
        raise AssertionError("Whisper must not run when a transcript is imported")

    def close(self):
        pass


@needs_ffmpeg
def test_run_uses_imported_transcript_and_keeps_it_for_later_runs(tmp_path, sample_info, runner):
    srt = tmp_path / "t.srt"
    srt.write_text(SRT, encoding="utf-8")
    store = create_or_open_project(sample_info, tmp_path / "projects")
    cfg = load_config(overrides={"transcription": {"transcript_file": str(srt)}, "llm": {"use_llm": False}}, local_path=None)
    job = JobManager(store, cfg, runner, whisper=FailingWhisper())
    assert job.run(stages=["transcribe"]).success
    assert load_transcript(store).model == "imported" and "imported transcript" in store.state.stages["transcribe"].message
    srt.unlink()
    later = load_config(overrides={"llm": {"use_llm": False}}, local_path=None)
    rep = JobManager(store, later, runner, whisper=FailingWhisper()).run(stages=["transcribe"])
    assert rep.success and rep.cached == ["transcribe"], "a later run reuses it without the original file or Whisper"
    (store.root / "imported_transcript.srt").write_text(SRT.replace("Hello", "Goodbye"), encoding="utf-8")
    assert transcript_cache_key(sample_info, later, store.root / "imported_transcript.srt") != store.state.stages["transcribe"].cache_key, "new text, new key"


@needs_ffmpeg
def test_bad_imported_file_stops_the_run_with_the_reason(tmp_path, sample_info, runner):
    store = create_or_open_project(sample_info, tmp_path / "projects")
    cfg = load_config(overrides={"transcription": {"transcript_file": str(tmp_path / "missing.srt")}}, local_path=None)
    rep = JobManager(store, cfg, runner, whisper=FailingWhisper()).run()
    assert not rep.success and "not found" in rep.error and rep.recovery
