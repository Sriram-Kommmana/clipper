"""Scene-cut aware smart crop and loudness normalisation."""
import pytest

from config import SmartCropConfig, load_config
from src.core.models import Candidate
from src.core.project_manager import create_or_open_project
from src.ingestion.video_probe import inspect_video
from src.rendering import loudness, subject_tracking
from src.rendering.cutting import ClipRenderer
from src.rendering.encoding import audio_args
from src.rendering.subject_tracking import Box, plan_crop_path, sample_subject_track, split_shots
from src.storage.cache import StageCache
from tests.conftest import needs_ffmpeg
from tests.helpers import make_transcript

SC = SmartCropConfig()
SRC_W, CROP_W = 1920, 608


def samples(xs, dt=0.75):
    return [(i * dt, x) for i, x in enumerate(xs)]


# ---------------------------------------------------------------- crop path vs scene cuts
def test_split_shots():
    s = samples([0.5] * 12)                      # 0 .. 8.25 s
    assert len(split_shots(s, None)) == 1
    shots = split_shots(s, [3.0, 6.0], duration=8.25)
    assert [(a, b) for a, b, _ in shots] == [(0.0, 3.0), (3.0, 6.0), (6.0, 8.25)]
    assert sum(len(x) for _, _, x in shots) == len(s), "every sample lands in exactly one shot"
    assert split_shots(s, [-1.0, 0.0, 99.0], duration=8.25)[0][1] == 8.25, "cuts outside the clip are ignored"


def test_crop_jumps_at_a_cut_instead_of_panning_through_it():
    xs = [0.2] * 8 + [0.8] * 8                    # subject is on the left, the camera cuts (t = 6.0), then they are on the right
    s = samples(xs)
    through = plan_crop_path(s, SRC_W, CROP_W, SC)
    cut = plan_crop_path(s, SRC_W, CROP_W, SC, cuts=[6.0], duration=s[-1][0])
    # without the cut the path glides across several keypoints
    assert max(abs(b - a) / max(t1 - t0, 1e-6) for (t0, a), (t1, b) in zip(through.keypoints, through.keypoints[1:])) < 800
    # with the cut it jumps within about one frame
    kp = cut.keypoints
    assert [t for t, _ in kp] == sorted({t for t, _ in kp}), "strictly increasing times"
    jump = [(t0, t1, b - a) for (t0, a), (t1, b) in zip(kp, kp[1:]) if abs(b - a) > 300]
    assert len(jump) == 1 and jump[0][1] - jump[0][0] <= 0.05 + 1e-9 and 5.9 <= jump[0][1] <= 6.01, jump
    assert jump[0][2] > 0, "to the right"
    before = [x for t, x in kp if t < 6.0]
    assert max(before) - min(before) < 60, "holds still on the left until the cut"
    assert any("reset" in n.lower() for n in cut.notes) and not cut.fallback


def test_shot_without_detection_gets_a_centre_crop_not_the_previous_position():
    xs = [0.2] * 8 + [None] * 8
    s = samples(xs)
    plan = plan_crop_path(s, SRC_W, CROP_W, SC, cuts=[6.0], duration=s[-1][0])
    centre = (SRC_W - CROP_W) / 2
    after = [x for t, x in plan.keypoints if t >= 6.0]
    assert after and all(abs(x - centre) < 1 for x in after)
    assert any("Shot 2" in n and "centre" in n for n in plan.notes)


def test_no_cuts_is_exactly_the_old_behaviour():
    xs = [0.3] * 8 + [0.3 + 0.05 * k for k in range(1, 11)] + [0.8] * 8
    s = samples(xs)
    assert plan_crop_path(s, SRC_W, CROP_W, SC).keypoints == plan_crop_path(s, SRC_W, CROP_W, SC, cuts=[], duration=s[-1][0]).keypoints


def test_many_short_shots_keep_expression_size_bounded():
    xs = [0.2 + 0.6 * ((i // 2) % 2) for i in range(160)]
    s = samples(xs, 0.5)
    cuts = [float(c) for c in range(2, 79, 2)]
    plan = plan_crop_path(s, SRC_W, CROP_W, SC, cuts=cuts, duration=s[-1][0])
    assert len(plan.keypoints) <= (len(cuts) + 1) * (SC.max_keypoints // (len(cuts) + 1) + 4)
    assert [t for t, _ in plan.keypoints] == sorted({t for t, _ in plan.keypoints})


def test_sampling_adds_a_frame_after_each_cut_and_forgets_the_previous_subject(monkeypatch):
    seen_times = []

    def fake_frames(video, times, max_width=640):
        for t in times:
            seen_times.append(t)
            yield t, t                                     # the "frame" is just its timestamp

    class Det:
        name = "fake"

        def detect(self, frame_t):
            if frame_t < 6.0:
                return [Box(0.2, 0.5, 0.10, 0.2)]
            return [Box(0.2, 0.5, 0.10, 0.2), Box(0.8, 0.5, 0.12, 0.2)]   # after the cut a slightly larger face appears elsewhere

    monkeypatch.setattr(subject_tracking, "read_frames", fake_frames)
    out = sample_subject_track("x.mp4", 0.0, 12.0, Det(), SC, cuts=[6.0])
    assert any(abs(t - 6.15) < 1e-6 for t in seen_times), "an extra sample just after the cut"
    after = [c for t, c in out if t >= 6.0]
    assert after and all(c == 0.8 for c in after), "after a cut the previous subject no longer anchors the choice"
    kept = sample_subject_track("x.mp4", 0.0, 12.0, Det(), SC)       # same footage, but no cut known: stays with the first subject
    assert [c for t, c in kept if t >= 6.0][0] == 0.2


# ---------------------------------------------------------------- where the cuts come from
def test_shot_cuts_come_from_the_scene_stage(sample_info, runner, tmp_path):
    store = create_or_open_project(sample_info, tmp_path / "p")
    r = ClipRenderer(sample_info, store, load_config(local_path=None), runner)
    c = Candidate(id="c1", start=6.0, end=20.0, category="story", title="t")
    assert r.shot_cuts(c) == [], "no scene analysis yet"
    StageCache(store.cache_dir).save("scenes", "k", {"cuts": [6.1, 8.0, 16.0, 19.9, "bad", None], "scenes": []})
    assert r.shot_cuts(c) == [2.0, 10.0], "cuts too close to either end are dropped, junk is ignored"
    StageCache(store.cache_dir).save("scenes", "k", ["unexpected"])
    assert r.shot_cuts(c) == []


# ---------------------------------------------------------------- loudness
JSON_BLOCK = """
[Parsed_loudnorm_0 @ 0x55]
{
	"input_i" : "-27.61",
	"input_tp" : "-4.47",
	"input_lra" : "18.06",
	"input_thresh" : "-39.20",
	"output_i" : "-16.58",
	"output_tp" : "-1.50",
	"output_lra" : "14.78",
	"output_thresh" : "-27.71",
	"normalization_type" : "dynamic",
	"target_offset" : "0.58"
}
"""


def test_parse_measurement():
    m = loudness.parse_measurement("noise\n" + JSON_BLOCK)
    assert m and m.input_i == -27.61 and m.input_tp == -4.47 and m.input_lra == 18.06 and m.target_offset == 0.58
    assert loudness.parse_measurement("") is None and loudness.parse_measurement("{ not json") is None
    assert loudness.parse_measurement(JSON_BLOCK.replace('"-27.61"', '"-inf"')) is None, "digital silence cannot be normalised"
    assert loudness.parse_measurement('{"input_i": "x"}') is None


@pytest.mark.parametrize("measured,expect_filter", [(-27.6, True), (-14.2, False), (-12.0, True), (-45.0, False), (-80.0, False)])
def test_normalise_filter_decisions(measured, expect_filter):
    m = loudness.parse_measurement(JSON_BLOCK.replace("-27.61", str(measured)))
    f, note = loudness.normalise_filter(m, -14.0)
    assert (f is not None) is expect_filter and note
    if f:
        assert "linear=true" in f and f"measured_I={measured:.2f}" in f and "I=-14" in f
    assert loudness.normalise_filter(None, -14.0)[0] is None


def test_audio_args_chain():
    base = audio_args("160k", True)
    assert base[base.index("-af") + 1] == "aresample=async=1:first_pts=0"
    chained = audio_args("160k", True, "loudnorm=I=-14")
    assert chained[chained.index("-af") + 1].startswith("loudnorm=I=-14,aresample")
    assert audio_args("160k", False, "loudnorm=I=-14") == ["-an"]


def _render_cfg(**render):
    r = {"preview_width": 270, "preview_height": 480, "width": 270, "height": 480, "mode": "original", **render}
    return load_config(overrides={"render": r}, local_path=None)


def _measure_file(runner, path):
    info = inspect_video(path, runner)
    return loudness.measure(runner, info, 0.0, info.duration - 0.2, -14.0)


@pytest.mark.slow
@needs_ffmpeg
def test_export_is_normalised_to_target_and_preview_is_not(sample_info, runner, tmp_path):
    """Source 10-14 s is a tone near -22 LUFS. The export is measured and brought to about -14 LUFS; with the option off it is left as it was."""
    loud = Candidate(id="c1", start=10.0, end=14.0, category="story", title="t")
    t = make_transcript([(10.0, "x")], duration=24)
    store = create_or_open_project(sample_info, tmp_path / "p")
    src = loudness.measure(runner, sample_info, 10.0, 14.0, -14.0)
    assert src is not None and -30 < src.input_i < -16, f"fixture should need about +8 dB, got {src}"

    on = ClipRenderer(sample_info, store, _render_cfg(normalize_audio=True), runner).render(loud, t)
    assert any("LUFS" in n for n in on.notes)
    got = _measure_file(runner, on.path)
    assert got is not None and got.input_i == pytest.approx(-14.0, abs=1.5), got

    off = ClipRenderer(sample_info, store, _render_cfg(normalize_audio=False), runner).render(loud, t)
    assert not any("LUFS" in n for n in off.notes)
    raw = _measure_file(runner, off.path)
    assert raw is not None and raw.input_i == pytest.approx(src.input_i, abs=1.5)

    prev = ClipRenderer(sample_info, store, _render_cfg(normalize_audio=True), runner).render(loud, t, preview=True)
    assert not any("LUFS" in n for n in prev.notes), "previews skip the extra measuring pass"


@pytest.mark.slow
@needs_ffmpeg
def test_measurement_failure_never_stops_a_render(sample_info, runner, tmp_path, monkeypatch):
    monkeypatch.setattr(loudness, "measure", lambda *a, **k: None)
    store = create_or_open_project(sample_info, tmp_path / "p")
    res = ClipRenderer(sample_info, store, _render_cfg(), runner).render(
        Candidate(id="c2", start=10.0, end=13.0, category="story", title="t"), make_transcript([(10.0, "x")], duration=24))
    assert res.path.exists() and any("original level kept" in n for n in res.notes)


@needs_ffmpeg
def test_silent_and_no_audio_sources_skip_measuring(silent_video, runner):
    info = inspect_video(silent_video, runner)
    assert loudness.measure(runner, info, 0.0, 2.5, -14.0) is None


@pytest.mark.slow
@needs_ffmpeg
def test_smart_crop_render_with_cuts_produces_a_valid_clip(sample_info, runner, tmp_path, monkeypatch):
    """End to end through real FFmpeg: the stepped crop expression must be accepted and the clip must have the right length."""
    import json
    import subprocess

    from src.rendering import cutting

    class Det:
        name = "fake"

        def detect(self, frame):
            return [Box(0.2 if float(frame.mean()) > 200 else 0.8, 0.5, 0.1, 0.2)]     # white shot: left; other shots: right

    monkeypatch.setattr(cutting, "make_detector", lambda name: Det())
    store = create_or_open_project(sample_info, tmp_path / "p")
    StageCache(store.cache_dir).save("scenes", "k", {"cuts": [8.0, 16.0], "scenes": []})
    cfg = _render_cfg(mode="smart_crop", normalize_audio=False)
    r = ClipRenderer(sample_info, store, cfg, runner)
    c = Candidate(id="c3", start=4.0, end=20.0, category="story", title="t")
    res = r.render(c, make_transcript([(4.0, "x")], duration=24))
    assert any("reset at 2 scene cut" in n for n in res.notes), res.notes
    out = json.loads(subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", str(res.path)],
                                    capture_output=True, text=True, check=True).stdout)
    assert float(out["format"]["duration"]) == pytest.approx(16.0, abs=0.2)
    # the cache key knows about the cuts: new cuts -> new file
    StageCache(store.cache_dir).save("scenes", "k", {"cuts": [8.0], "scenes": []})
    assert r.render_path(c, [], False) != res.path
