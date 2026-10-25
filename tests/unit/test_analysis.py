import numpy as np
import pytest
import soundfile as sf

from config import AudioConfig
from src.analysis.audio_analysis import analyze_audio_file, analyze_rms
from src.analysis.frame_extraction import extract_thumbnails, read_frames, sample_times, score_stats, visual_suitability
from src.analysis.scene_detection import build_scene_command, build_scenes, detect_scenes, parse_showinfo
from config import ScenesConfig
from tests.conftest import needs_ffmpeg

CFG = AudioConfig()


def synth_rms():
    """60 windows of 0.5 s = 30 s: quiet, loud 10-16 s, silence 20-24 s, one isolated burst at 27 s."""
    rms = np.full(60, 0.01)
    rms[20:32] = 0.4
    rms[40:48] = 1e-6
    rms[54] = 0.5
    return rms


def test_events_on_synthetic_energy():
    a = analyze_rms(synth_rms(), 30.0, CFG)
    sus = [e for e in a.events if e.kind == "sustained"]
    assert len(sus) == 1 and abs(sus[0].start - 10) <= 0.5 and abs(sus[0].end - 16) <= 0.5
    assert any(e.kind == "peak" and abs(e.start - 27) <= 1 for e in a.events)
    assert any(e.kind == "change_up" and 9 <= e.start <= 11 for e in a.events)
    assert any(e.kind == "change_down" for e in a.events)
    assert len(a.silences) == 1 and abs(a.silences[0].start - 20) < 0.6 and abs(a.silences[0].end - 24) < 0.6
    assert a.silence_ratio(20, 24) > 0.9 and a.silence_ratio(0, 5) == 0
    assert a.max_energy(10, 16) > 0.9 and a.mean_energy(0, 5) < 0.3
    assert a.events_in(0, 5) == []


def test_flat_audio_does_not_manufacture_peaks():
    rng = np.random.default_rng(0)
    rms = 0.05 * (1 + 0.02 * rng.standard_normal(200))
    a = analyze_rms(rms, 100.0, CFG)
    assert not [e for e in a.events if e.kind in ("sustained", "peak")], a.events
    assert any("dynamic range" in w for w in a.warnings)


def test_empty_audio_warns():
    a = analyze_rms(np.zeros(0), 0.0, CFG)
    assert not a.has_audio and a.warnings


def test_quality_warnings():
    a = analyze_rms(np.full(100, 1e-4), 50.0, CFG, {"clipping_ratio": 0.02})
    assert any("clipping" in w for w in a.warnings) and any("quiet" in w for w in a.warnings)


def test_wav_streaming_matches(tmp_path):
    sr = 16000
    t = np.arange(sr * 12) / sr
    y = 0.02 * np.sin(2 * np.pi * 200 * t)
    y[4 * sr:8 * sr] *= 20
    wav = tmp_path / "a.wav"
    sf.write(wav, y.astype("float32"), sr)
    a = analyze_audio_file(wav, CFG)
    assert abs(a.duration - 12) < 0.01 and len(a.energy) == 24
    assert a.mean_energy(4, 8) > a.mean_energy(0, 3) + 0.3
    assert any(e.kind == "sustained" for e in a.events)


def test_scene_helpers():
    stderr = "[Parsed_showinfo_2 @ 0x1] n:0 pts:200 pts_time:8.0 pos:1\nnoise\n[Parsed_showinfo_2 @ 0x1] n:1 pts_time:16.04\n"
    assert parse_showinfo(stderr) == [8.0, 16.04]
    scenes = build_scenes([8.0, 8.3, 16.04, 23.9], 24.0)
    assert [s.start for s in scenes] == [0.0, 8.0, 16.04]
    assert scenes[-1].end == 24.0, "cut within 1s of the end is dropped"


@needs_ffmpeg
def test_scene_command_and_real_detection(sample_info, runner):
    cmd = build_scene_command(sample_info, ScenesConfig(keyframes_only=True))
    assert cmd[cmd.index("-skip_frame") + 1] == "nokey" and cmd[cmd.index("-i") + 1] == sample_info.path
    sa = detect_scenes(sample_info, runner, ScenesConfig())
    assert len(sa.cuts) == 2 and abs(sa.cuts[0] - 8) < 0.2 and abs(sa.cuts[1] - 16) < 0.2
    assert sa.cuts_in(0, 10) == 1 and sa.cut_rate(0, 24) == pytest.approx(2 / 24)


@needs_ffmpeg
def test_visual_checks_and_thumbnails(sample_video, silent_video, tmp_path):
    assert list(read_frames(sample_video, [1.0, 99999.0]))[0][0] == 1.0
    vc = visual_suitability(sample_video, 0, 6)
    assert vc.samples == 4 and vc.score > 50
    from src.analysis.scene_detection import Scene

    thumbs = extract_thumbnails(sample_video, [Scene(start=0, end=8), Scene(start=8, end=16)], tmp_path / "f", 10)
    assert len(thumbs) == 2 and (tmp_path / "f" / thumbs[0][1]).stat().st_size > 100
    assert extract_thumbnails(sample_video, [], tmp_path, 5) == []


def test_score_stats_pure():
    assert score_stats([], []).score == 50
    black = score_stats([2.0, 3.0, 1.0], [0.0, 0.0, 0.0])
    assert black.score < 10 and any("black" in w.lower() for w in black.warnings)
    good = score_stats([120, 130], [300, 400])
    assert good.score == 100 and not good.warnings
    assert sample_times(0, 10, 2) == [2.5, 7.5]
