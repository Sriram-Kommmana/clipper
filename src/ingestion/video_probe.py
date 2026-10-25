"""Video metadata extraction via ffprobe."""
from __future__ import annotations

from pathlib import Path

from src.core.exceptions import FFmpegError, InvalidMediaError
from src.core.models import VideoInfo
from src.ingestion.input_validator import compute_fingerprint, validate_input_path
from src.rendering.ffmpeg_runner import FFmpegRunner


def _fps(stream: dict) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        val = stream.get(key) or "0/0"
        try:
            num, den = val.split("/")
            if float(den) > 0 and float(num) > 0:
                return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            continue
    return 0.0


def _rotation(stream: dict) -> int:
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                return int(round(float(sd["rotation"]))) % 360
            except (TypeError, ValueError):
                pass
    try:
        return int(stream.get("tags", {}).get("rotate", 0)) % 360
    except (TypeError, ValueError):
        return 0


def parse_probe(data: dict, path: Path) -> VideoInfo:
    """Convert raw ffprobe JSON to VideoInfo, or raise InvalidMediaError explaining what is wrong."""
    streams = data.get("streams", []) or []
    videos = [
        s for s in streams
        if s.get("codec_type") == "video" and not (s.get("disposition", {}) or {}).get("attached_pic")
    ]
    if not videos:
        raise InvalidMediaError(f"{path.name} has no video stream.", recovery="This tool needs a file with moving video.")
    v = videos[0]
    audios = [s for s in streams if s.get("codec_type") == "audio"]
    fmt = data.get("format", {}) or {}
    try:
        duration = float(fmt.get("duration") or v.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        raise InvalidMediaError(
            f"Could not determine a duration for {path.name} (file may be truncated or corrupt).",
            recovery="Re-export the video, or repair it with: ffmpeg -i broken.mp4 -c copy fixed.mp4",
        )
    width, height = int(v.get("width") or 0), int(v.get("height") or 0)
    if width <= 0 or height <= 0:
        raise InvalidMediaError(f"{path.name} reports an invalid frame size ({width}x{height}).")
    rot = _rotation(v)
    if rot in (90, 270):
        width, height = height, width
    info = VideoInfo(
        path=str(path),
        filename=path.name,
        size_bytes=int(fmt.get("size") or path.stat().st_size),
        container=str(fmt.get("format_name", "")),
        duration=duration,
        width=width,
        height=height,
        fps=_fps(v),
        video_codec=str(v.get("codec_name", "")),
        audio_codec=audios[0].get("codec_name") if audios else None,
        has_audio=bool(audios),
        audio_channels=int(audios[0].get("channels") or 0) if audios else 0,
        audio_sample_rate=int(audios[0].get("sample_rate") or 0) if audios else 0,
        rotation=rot,
        bit_rate=int(fmt.get("bit_rate") or 0),
    )
    if not audios:
        info.warnings.append("No audio stream: speech cannot be transcribed, so no clips can be generated.")
    if info.duration < 30:
        info.warnings.append("Very short video (<30 s): few or no clips will be found.")
    if info.fps and info.fps < 10:
        info.warnings.append(f"Low frame rate ({info.fps:.1f} fps).")
    return info


def inspect_video(raw_path: str | Path, runner: FFmpegRunner, *, fingerprint: bool = True) -> VideoInfo:
    """Validate path -> ffprobe -> VideoInfo (+fingerprint). Raises InvalidMediaError / FFmpegError."""
    path = validate_input_path(raw_path)
    try:
        data = runner.probe_json(path)
    except FFmpegError as exc:
        raise InvalidMediaError(
            f"{path.name} could not be read as a media file (corrupt, truncated, or unsupported codec).",
            details=exc.details,
        ) from exc
    info = parse_probe(data, path)
    if fingerprint:
        info.fingerprint = compute_fingerprint(path)
    return info
