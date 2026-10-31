import re
import shutil
import subprocess
import threading

import cv2
import numpy as np
import pytest
import soundfile as sf

from config import SmartCropConfig, load_config
from src.core.exceptions import JobCancelled
from src.core.models import Candidate
from src.core.project_manager import create_or_open_project
from src.rendering import cutting
from src.rendering.cutting import ClipRenderer, build_render_args, output_size
from src.rendering.encoding import EncoderChoice, audio_args, choose_encoder
from src.rendering.subject_tracking import Box, make_detector, pick_subject, plan_crop_path
from src.rendering.vertical_video import blur_layout, crop_layout, lerp_expression, original_layout
from tests.conftest import needs_ffmpeg
from tests.helpers import make_transcript

CFG = load_config(local_path=None)
X264 = EncoderChoice("libx264", ("-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"), False)


def py_eval(expr: str, t: float) -> float:
    """Evaluate the tiny subset of ffmpeg expression language lerp_expression emits."""
    e = expr.replace("if(lt(", "_if(_lt(")
    return eval(e, {"_if": lambda c, a, b: a if c else b, "_lt": lambda a, b: a < b, "t": t})


def test_lerp_expression_matches_piecewise_linear():
    pts = [(0.0, 100.0), (2.0, 300.0), (4.0, 300.0), (6.0, 0.0)]
    e = lerp_expression(pts)
    for t, want in [(0, 100), (1, 200), (2, 300), (3, 300), (5, 150), (6, 0), (9, 0)]:
        assert py_eval(e, t) == pytest.approx(want, abs=0.2), (t, e)
    assert lerp_expression([(0, 123.44)]) == "123.4"
    assert lerp_expression([(0, 50.0), (3, 50.2)]) == "50", "tiny movement collapses to a constant"
    with pytest.raises(ValueError):
        lerp_expression([])


def test_layout_filters():
    b = blur_layout(1920, 1080, 1080, 1920, CFG.render)
    assert b.startswith("split=2") and "gblur" in b and "overlay=(W-w)/2:(H-h)/2" in b and "force_original_aspect_ratio=decrease" in b
    assert "stretch" not in b
    same = blur_layout(1080, 1920, 1080, 1920, CFG.render)
    assert "gblur" not in same, "a source that already is 9:16 needs no blurred background"
    c = crop_layout(1920, 1080, 1080, 1920, [(0.0, 0.0), (2.0, 1e9)])
    assert "crop=w=608:h=1080" in c and "x='" in c and c.endswith("scale=1080:1920:flags=lanczos,setsar=1")
    assert py_eval(re.search(r"x='(.*)':y", c).group(1), 5) == 1920 - 608, "tracking is clamped inside the frame"
    tall = crop_layout(1000, 2000, 1080, 1920, None)
    assert "crop=w=1000:h=1778:x=0:y=111" in tall
    assert "force_divisible_by=2" in original_layout(1280, 720)


def test_output_sizes(sample_info):
    assert output_size(CFG, sample_info, False) == (1080, 1920)
    assert output_size(CFG, sample_info, True) == (540, 960)
    orig = load_config(overrides={"render": {"mode": "original"}}, local_path=None)
    assert output_size(orig, sample_info, False) == (640, 360), "never upscales in original mode"


@needs_ffmpeg
def test_build_args_are_safe_and_ordered(sample_info, tmp_path):
    args = build_render_args(sample_info, 9.0, 12.5, CFG, X264, tmp_path / "o.mp4", preview=False, ass_name="x.ass", crop_points=None)
    assert all(isinstance(a, str) for a in args)
    assert args.index("-ss") < args.index("-i") < args.index("-t"), "input-side seek => accurate on re-encode"
    assert args[args.index("-ss") + 1] == "9.000" and args[args.index("-t") + 1] == "3.500"
    graph = args[args.index("-filter_complex") + 1]
    assert graph.endswith(",ass=x.ass[vout]") and graph.startswith("[0:v:0]fps=25.000,")
    assert args[args.index("-i") + 1] == sample_info.path
    assert "-c:a" in args and args[-1].endswith("o.mp4") and "+faststart" in args
    noass = build_render_args(sample_info, 0, 5, CFG, X264, tmp_path / "o.mp4", preview=True, ass_name=None, crop_points=None)
    assert noass[noass.index("-filter_complex") + 1].endswith("[vout]") and "ass=" not in " ".join(noass)
    assert audio_args("160k", False) == ["-an"]


# ---------------------------------------------------------------- smart crop planning
def samples(xs, dt=0.75):
    return [(i * dt, x) for i, x in enumerate(xs)]


SC = SmartCropConfig()


def test_crop_plan_fallbacks_and_stability():
    p = plan_crop_path(samples([None] * 10), 1920, 608, SC)
    assert p.fallback and p.keypoints == [(0.0, (1920 - 608) / 2)] and p.notes
    p = plan_crop_path(samples([None] * 9 + [0.5]), 1920, 608, SC)
    assert p.fallback, "below min_detection_ratio"
    still = plan_crop_path(samples([0.5 + 0.01 * ((-1) ** i) for i in range(20)]), 1920, 608, SC)
    xs = {round(x) for _, x in still.keypoints}
    assert not still.fallback and len(xs) == 1, f"jitter inside the dead-band must not move the crop: {still.keypoints}"


def test_crop_plan_follows_real_movement_smoothly_and_in_bounds():
    xs = [0.3] * 8 + [0.3 + 0.05 * k for k in range(1, 11)] + [0.8] * 8
    p = plan_crop_path(samples(xs), 1920, 608, SC)
    ys = [x for _, x in p.keypoints]
    assert ys[0] < ys[-1] - 300, "crop follows the subject to the right"
    assert all(0 <= y <= 1920 - 608 for y in ys)
    assert max(abs(a - b) for a, b in zip(ys, ys[1:])) < 400, "no jump cuts"
    assert [t for t, _ in p.keypoints] == sorted(t for t, _ in p.keypoints)
    capped = plan_crop_path(samples([0.2 + 0.6 * (i % 2) for i in range(200)], 0.1), 1920, 608, SmartCropConfig(max_keypoints=6, deadband_pct=0.0, smoothing=1.0))
    assert len(capped.keypoints) <= 6


@pytest.mark.parametrize("name", ["face", "person"])
def test_real_detectors_load_and_run(name):
    # make_detector hides failures behind a centre crop, so a broken OpenCV must fail here instead.
    detector = make_detector(name)
    assert detector is not None
    assert detector.detect(np.zeros((360, 640, 3), dtype=np.uint8)) == []


def test_pick_subject_prefers_big_then_stable():
    big, small = Box(0.3, 0.5, 0.3, 0.4), Box(0.8, 0.5, 0.05, 0.06)
    assert pick_subject([small, big], None) is big and pick_subject([], 0.5) is None
    a, b = Box(0.2, 0.5, 0.2, 0.3), Box(0.7, 0.5, 0.19, 0.29)
    assert pick_subject([a, b], 0.72) is b, "comparable sizes: stay with the previous subject"


# ---------------------------------------------------------------- real FFmpeg renders
def ffprobe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)], capture_output=True, text=True, check=True).stdout
    import json
    return json.loads(out)


def cfg_fast(**render):
    r = {"preview_width": 270, "preview_height": 480, "width": 270, "height": 480, **render}
    return load_config(overrides={"render": r, "performance": {"mode": "normal"}}, local_path=None)


def make_renderer(tmp_path, info, runner, cfg=None):
    store = create_or_open_project(info, tmp_path / "proj")
    return ClipRenderer(info, store, cfg or cfg_fast(), runner), store


def candidate(start, end, **kw):
    return Candidate(id="c" + str(int(start * 10)), start=start, end=end, category="story", title="t", **kw)


@pytest.mark.slow
@needs_ffmpeg
def test_blur_render_properties_and_cache(sample_info, runner, tmp_path):
    rend, store = make_renderer(tmp_path, sample_info, runner)
    t = make_transcript([(8.5, "Hello there my friend.")], duration=24)
    c = candidate(8.0, 14.0)
    res = rend.render(c, t, preview=False)
    info = ffprobe(res.path)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert (v["width"], v["height"], v["codec_name"], v["pix_fmt"]) == (270, 480, "h264", "yuv420p")
    assert a["codec_name"] == "aac"
    assert float(info["format"]["duration"]) == pytest.approx(6.0, abs=0.12)
    assert not res.cached and res.path.parent == store.renders_dir and not list(store.renders_dir.glob("*.part*"))
    again = rend.render(c, t, preview=False)
    assert again.cached and again.path == res.path
    moved = rend.render(candidate(8.0, 13.0).model_copy(update={"id": c.id}), t, preview=False)
    assert not moved.cached and moved.path != res.path
    assert not res.path.exists(), "stale render of the same clip/kind is cleaned up"
    assert store.state.render_seconds > 0


@pytest.mark.slow
@needs_ffmpeg
def test_cut_is_frame_and_audio_accurate(sample_info, runner, tmp_path):
    """Source audio jumps from quiet to loud at exactly 10.0 s; a cut starting at 9.0 s must show it at ~1.0 s."""
    rend, _ = make_renderer(tmp_path, sample_info, runner, cfg_fast(mode="original"))
    t = make_transcript([(9.0, "x")], duration=24)
    res = rend.render(candidate(9.0, 12.0), t)
    wav = tmp_path / "o.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(res.path), "-ac", "1", "-ar", "16000", str(wav)], check=True)
    y, sr = sf.read(wav)
    win = int(0.05 * sr)
    rms = np.array([np.sqrt(np.mean(y[i:i + win] ** 2)) for i in range(0, len(y) - win, win)])
    assert (rms > 0.04).any() and rms[0] < 0.02, rms[:3]
    onset = int(np.argmax(rms > 0.04)) * 0.05
    assert onset == pytest.approx(1.0, abs=0.15), onset


@pytest.mark.slow
@needs_ffmpeg
def test_burned_karaoke_highlight_is_synchronised(sample_info, runner, tmp_path):
    """Word 'Hello' is active at 9.0-9.4 s (source) => yellow pixels at clip time 1.1 s, none before the first word."""
    cfg = load_config(overrides={"render": {"preview_width": 540, "preview_height": 960, "width": 540, "height": 960},
                                 "subtitles": {"pop_animation": False}}, local_path=None)
    rend, _ = make_renderer(tmp_path, sample_info, runner, cfg)
    t = make_transcript([(9.0, "Hello wonderful world today.")], word_dur=0.5, gap=0.0, duration=24)   # words at 9.0,9.5,10.0,10.5
    res = rend.render(candidate(8.0, 12.0), t)

    def yellow(sec):
        cap = cv2.VideoCapture(str(res.path))
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ok, f = cap.read()
        assert ok
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (20, 150, 200), (35, 255, 255))      # highlight colour #FFE135
        white = cv2.inRange(hsv, (0, 0, 235), (180, 30, 255))
        cap.release()
        return int(mask.sum() / 255), int(white.sum() / 255)

    y_before, w_before = yellow(0.5)      # before any caption
    y_hello, w_hello = yellow(1.25)       # 'Hello' active
    y_world, _ = yellow(2.25)             # 'world' active -> different word highlighted, still yellow present
    assert y_before == 0 and w_before < w_hello
    assert y_hello > 100 and y_world > 100


@pytest.mark.slow
@needs_ffmpeg
def test_smart_crop_follows_keypoints_and_centre_fallback(sample_info, runner, tmp_path):
    cfg = cfg_fast(mode="smart_crop")
    cfg_none = load_config(overrides={"render": cfg.render.model_dump(), "smart_crop": {"detector": "none"}}, local_path=None)
    rend, _ = make_renderer(tmp_path, sample_info, runner, cfg_none)
    t = make_transcript([(16.5, "a b c")], duration=24)
    res = rend.render(candidate(16.0, 20.0), t)
    assert res.notes and "centre" in res.notes[0].lower()
    assert ffprobe(res.path)["streams"][0]["height"] == 480
    # explicit moving crop: first frame from the left edge, last from the right edge of the testsrc2 scene
    args = build_render_args(sample_info, 16.0, 20.0, cfg, X264, tmp_path / "moving.mp4", preview=False, ass_name=None,
                             crop_points=[(0.0, 0.0), (3.9, 1e9)])
    runner.run(args)
    cap = cv2.VideoCapture(str(tmp_path / "moving.mp4"))
    ok1, f1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_MSEC, 3500)
    ok2, f2 = cap.read()
    assert ok1 and ok2 and np.abs(f1.astype(int) - f2.astype(int)).mean() > 5, "crop window moved across the source"


@pytest.mark.slow
@needs_ffmpeg
def test_no_audio_source_and_hardware_fallback(silent_video, runner, tmp_path, monkeypatch):
    from src.ingestion.video_probe import inspect_video

    info = inspect_video(silent_video, runner)
    rend, _ = make_renderer(tmp_path, info, runner, cfg_fast())
    t = make_transcript([(0.0, "x")], duration=3)
    res = rend.render(candidate(0.5, 2.5), t)
    assert [s["codec_type"] for s in ffprobe(res.path)["streams"]] == ["video"]
    # pretend NVENC passed its probe but fails when used -> automatic x264 retry, reported in notes
    fake = EncoderChoice("h264_nvenc", ("-c:v", "h264_nvenc", "-pix_fmt", "yuv420p"), True)
    real_choose = cutting.choose_encoder
    monkeypatch.setattr(cutting, "choose_encoder", lambda r, c, preview, allow_hardware=True: real_choose(r, c, preview=preview, allow_hardware=False) if not allow_hardware else fake)
    if shutil.which("nvidia-smi") is None:
        res2 = rend.render(candidate(0.5, 2.0), t, force=True)
        assert res2.encoder == "libx264" and any("GPU encoder failed" in n for n in res2.notes)


@pytest.mark.slow
@needs_ffmpeg
def test_cancel_stops_render_cleanly(sample_info, runner, tmp_path):
    rend, store = make_renderer(tmp_path, sample_info, runner)
    ev = threading.Event()
    ev.set()
    with pytest.raises(JobCancelled):
        rend.render(candidate(0.0, 20.0), make_transcript([(1.0, "x")], duration=24), cancel=ev)
    assert not list(store.renders_dir.glob("*.mp4")) and not list(store.renders_dir.glob("*.part*"))


@needs_ffmpeg
def test_choose_encoder_respects_config(runner):
    never = load_config(overrides={"render": {"hardware_encoding": "never"}}, local_path=None)
    assert choose_encoder(runner, never.render, preview=False).name == "libx264"
    prev = choose_encoder(runner, never.render, preview=True)
    assert "ultrafast" in prev.video_args and str(never.render.preview_crf) in prev.video_args
