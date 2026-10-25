import pytest

from src.core.exceptions import InvalidMediaError
from src.ingestion.input_validator import compute_fingerprint, validate_input_path
from src.ingestion.video_probe import inspect_video, parse_probe
from tests.conftest import needs_ffmpeg


def test_validate_rejects_bad_inputs(tmp_path):
    with pytest.raises(InvalidMediaError, match="No video"):
        validate_input_path("")
    with pytest.raises(InvalidMediaError, match="not found"):
        validate_input_path(tmp_path / "missing.mp4")
    with pytest.raises(InvalidMediaError, match="Not a file"):
        d = tmp_path / "dir.mp4"
        d.mkdir()
        validate_input_path(d)
    bad = tmp_path / "clip.avi"
    bad.write_bytes(b"x")
    with pytest.raises(InvalidMediaError, match="Unsupported"):
        validate_input_path(bad)
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(InvalidMediaError, match="empty"):
        validate_input_path(empty)
    with pytest.raises(InvalidMediaError):
        validate_input_path("a\x00b.mp4")


def test_validate_returns_absolute_even_for_dash_names(tmp_path, monkeypatch):
    f = tmp_path / "-dangerous.mp4"
    f.write_bytes(b"123")
    monkeypatch.chdir(tmp_path)
    p = validate_input_path("-dangerous.mp4")
    assert p.is_absolute() and not p.name.startswith("/")


def test_fingerprint_depends_on_content_not_name(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    a.write_bytes(b"A" * 5000)
    b.write_bytes(b"A" * 5000)
    assert compute_fingerprint(a) == compute_fingerprint(b)
    b.write_bytes(b"A" * 4999 + b"B")
    assert compute_fingerprint(a) != compute_fingerprint(b)


def test_parse_probe_variants(tmp_path):
    p = tmp_path / "v.mp4"
    p.write_bytes(b"x")
    base = {
        "format": {"duration": "12.5", "size": "1000", "format_name": "mov,mp4", "bit_rate": "800"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080, "avg_frame_rate": "30000/1001"},
            {"codec_type": "audio", "codec_name": "aac", "channels": 2, "sample_rate": "48000"},
        ],
    }
    info = parse_probe(base, p)
    assert (info.width, info.height, info.has_audio) == (1920, 1080, True)
    assert abs(info.fps - 29.97) < 0.01 and info.duration == 12.5
    # rotated phone video: displayed size swaps
    rot = {**base, "streams": [{**base["streams"][0], "side_data_list": [{"rotation": -90}]}, base["streams"][1]]}
    r = parse_probe(rot, p)
    assert (r.width, r.height, r.rotation) == (1080, 1920, 270)
    # no audio -> warning
    noaud = {**base, "streams": [base["streams"][0]]}
    assert any("No audio" in w for w in parse_probe(noaud, p).warnings)
    # no video / attached-pic only / zero duration
    with pytest.raises(InvalidMediaError, match="no video"):
        parse_probe({**base, "streams": [base["streams"][1]]}, p)
    pic = {"codec_type": "video", "codec_name": "mjpeg", "width": 10, "height": 10, "disposition": {"attached_pic": 1}}
    with pytest.raises(InvalidMediaError, match="no video"):
        parse_probe({**base, "streams": [pic]}, p)
    with pytest.raises(InvalidMediaError, match="duration"):
        parse_probe({**base, "format": {"duration": "0"}}, p)


@needs_ffmpeg
def test_inspect_real_video(sample_video, runner):
    info = inspect_video(sample_video, runner)
    assert (info.width, info.height) == (640, 360)
    assert abs(info.duration - 24.0) < 0.3 and info.has_audio and info.video_codec == "h264"
    assert len(info.fingerprint) == 20


@needs_ffmpeg
def test_inspect_silent_and_corrupt(silent_video, runner, tmp_path):
    info = inspect_video(silent_video, runner)
    assert not info.has_audio and any("No audio" in w for w in info.warnings)
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"this is definitely not a video" * 100)
    with pytest.raises(InvalidMediaError, match="could not be read"):
        inspect_video(corrupt, runner)
