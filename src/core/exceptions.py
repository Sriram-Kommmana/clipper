"""Exception hierarchy. Every error carries an actionable `recovery` hint shown in the UI/CLI."""
from __future__ import annotations


class ClipperError(Exception):
    """Base class. `recovery` tells the user what to try next."""

    default_recovery = ""

    def __init__(self, message: str, *, recovery: str | None = None, details: str | None = None):
        super().__init__(message)
        self.recovery = recovery if recovery is not None else self.default_recovery
        self.details = details

    def user_message(self) -> str:
        msg = str(self)
        if self.recovery:
            msg += f"\n  How to fix: {self.recovery}"
        return msg


class ConfigError(ClipperError):
    default_recovery = "Fix configs/local.yaml (or the setting you changed); compare with configs/default.yaml."


class InvalidMediaError(ClipperError):
    default_recovery = "Choose a readable .mp4/.mov/.mkv file with a video stream. Re-export or re-download it if it is corrupt."


class FFmpegNotFoundError(ClipperError):
    default_recovery = (
        "Install FFmpeg (Windows: 'winget install Gyan.FFmpeg' or download from ffmpeg.org) and make sure "
        "ffmpeg/ffprobe are on PATH, or set paths.ffmpeg_path / paths.ffprobe_path in configs/local.yaml."
    )


class FFmpegError(ClipperError):
    default_recovery = "See the log for the FFmpeg error tail. Corrupt media or a missing codec are the usual causes."


class TranscriptionError(ClipperError):
    default_recovery = "Check that the video has an audio track; try transcription.device: cpu or a smaller model."


class NoAudioError(TranscriptionError):
    default_recovery = (
        "The file has no audio stream, so speech cannot be transcribed. Use a source with audio; "
        "visual-only clipping is not supported."
    )


class ResourceExhaustedError(ClipperError):
    default_recovery = (
        "Switch performance.mode to 'low', use a smaller model (transcription.model: base, llm.model: qwen3:1.7b), "
        "set transcription.device: cpu, close other GPU applications, then resume the project."
    )


class LLMUnavailableError(ClipperError):
    default_recovery = (
        "Install Ollama from ollama.com, start it, then run: ollama pull qwen3:4b  "
        "(or set llm.use_llm: false for the reduced, non-semantic fallback)."
    )


class LLMModelMissingError(LLMUnavailableError):
    pass


class LLMOutputError(ClipperError):
    default_recovery = "The local model returned invalid output repeatedly. Try a larger model (qwen3:8b) or lower llm.section_max_words."


class CandidateValidationError(ClipperError):
    default_recovery = "Adjust the clip boundaries so they lie inside the video and satisfy the duration limits."


class RenderError(ClipperError):
    default_recovery = "Check the FFmpeg log for this clip; try render.hardware_encoding: never or render.mode: blur."


class JobCancelled(ClipperError):
    default_recovery = "Intermediate results were kept. Press Run/Resume to continue where it stopped."


class ProjectError(ClipperError):
    default_recovery = "Open the project folder and check project.json; delete the cache folder to force a clean re-run."
