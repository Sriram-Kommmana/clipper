"""Local speech-to-text with faster-whisper behind a swappable backend interface."""
from __future__ import annotations

import gc
import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Protocol

from config import AppConfig
from src.core.exceptions import JobCancelled, NoAudioError, ResourceExhaustedError, TranscriptionError
from src.core.models import VideoInfo
from src.rendering.ffmpeg_runner import FFmpegRunner
from src.storage.cache import StageCache, make_key
from src.storage.project_store import ProjectStore, atomic_write_text
from src.transcription.importer import load_import
from src.transcription.transcript_models import Transcript, TranscriptSegment, Word
from src.utils.hardware_detection import is_gpu_memory_error, register_nvidia_dll_dirs

log = logging.getLogger(__name__)

STAGE_VERSION = 2
LOW_CONFIDENCE = 0.7
MIN_LANGUAGE_PROB = 0.02
ProgressFn = Callable[[float, str], None]


class WhisperBackend(Protocol):
    device: str
    compute_type: str

    def transcribe(
        self, wav: Path, *, language: str | None, duration: float,
        on_progress: Callable[[float], None] | None, cancel: threading.Event | None,
    ) -> Transcript: ...

    def close(self) -> None: ...


class FasterWhisperBackend:
    """faster-whisper with device/compute auto-selection and a visible CPU fallback.

    Models are loaded with local_files_only=True: no hidden downloads at run time. Fetch weights once
    with `python scripts/download_models.py`.
    """

    def __init__(self, cfg: AppConfig, threads: int):
        self.cfg = cfg.transcription
        self.beam_size = cfg.effective().transcription.beam_size
        self.threads = threads
        self.device = ""
        self.compute_type = ""
        self._model = None
        self.fallback_note: str | None = None
        self._language_probs: list[tuple[str, float]] = []

    # ---- loading -------------------------------------------------------------------------------
    def _plan(self) -> list[tuple[str, str]]:
        register_nvidia_dll_dirs()
        try:
            import ctranslate2

            cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception as exc:
            log.warning("CUDA probe failed (%s); assuming CPU", exc)
            cuda = False
        want = self.cfg.device
        if want == "auto":
            want = "cuda" if cuda else "cpu"
        if want == "cuda" and not cuda:
            log.warning("transcription.device=cuda but no CUDA device is usable; using CPU")
            self.fallback_note = "CUDA unavailable - transcribing on CPU (slower)."
            want = "cpu"
        plan: list[tuple[str, str]] = []
        if want == "cuda":
            ct = self.cfg.compute_type
            plan += [("cuda", ct if ct != "auto" else "float16")]
            if ct in ("auto", "float16"):
                plan.append(("cuda", "int8_float16"))
        if want == "cpu" or self.cfg.allow_cpu_fallback:
            ct = self.cfg.compute_type
            plan.append(("cpu", ct if (ct in ("int8", "float32") and want == "cpu") else "int8"))
        return plan

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionError("faster-whisper is not installed.", recovery="pip install -r requirements.txt") from exc
        last: Exception | None = None
        for device, compute in self._plan():
            try:
                log.info("Loading Whisper '%s' on %s (%s)", self.cfg.model, device, compute)
                self._model = WhisperModel(
                    self.cfg.model, device=device, compute_type=compute,
                    cpu_threads=self.threads if device == "cpu" else 0,
                    num_workers=1, local_files_only=True,
                )
                if (device, compute) != self._plan()[0]:
                    self.fallback_note = f"Fell back to {device}/{compute} after an earlier attempt failed."
                self.device, self.compute_type = device, compute
                return
            except Exception as exc:
                last = exc
                msg = str(exc).lower()
                if "local_files_only" in msg or "cannot find" in msg or "offline" in msg or "not found" in msg and "model" in msg:
                    raise TranscriptionError(
                        f"Whisper model '{self.cfg.model}' is not downloaded.",
                        recovery=f"Run once (needs internet): python scripts/download_models.py --whisper {self.cfg.model}",
                    ) from exc
                log.warning("Could not load on %s/%s: %s", device, compute, exc)
        if last is not None and is_gpu_memory_error(last):
            raise ResourceExhaustedError(f"Could not load Whisper '{self.cfg.model}': {last}") from last
        raise TranscriptionError(f"Could not load Whisper '{self.cfg.model}': {last}") from last

    # ---- inference -----------------------------------------------------------------------------
    def transcribe(self, wav, *, language, duration, on_progress, cancel) -> Transcript:
        self._load()
        try:
            result = self._run(wav, language, duration, on_progress, cancel)
            return result if language else self._retry_if_misdetected(result, wav, duration, on_progress, cancel)
        except JobCancelled:
            raise
        except Exception as exc:
            if self.device == "cuda" and self.cfg.allow_cpu_fallback and is_gpu_memory_error(exc):
                log.warning("GPU transcription failed (%s) - retrying on CPU int8", exc)
                self.close()
                self.cfg = self.cfg.model_copy(update={"device": "cpu", "compute_type": "int8"})
                self._load()
                self.fallback_note = f"GPU failed ({str(exc)[:80]}); transcribed on CPU instead."
                return self._run(wav, language, duration, on_progress, cancel)
            if is_gpu_memory_error(exc):
                raise ResourceExhaustedError(f"GPU error during transcription: {exc}") from exc
            raise TranscriptionError(f"Transcription failed: {exc}") from exc

    def _retry_if_misdetected(self, first: Transcript, wav, duration, on_progress, cancel) -> Transcript:
        """Hindi with English words can be detected as English, which translates instead of transcribing and shows up
        as low word confidence. Then try the next most likely language and keep whichever transcript is more confident."""
        confidence = first.mean_word_confidence()
        if confidence is None or confidence >= LOW_CONFIDENCE:
            return first
        other = next((lang for lang, prob in self._language_probs if lang != first.language and prob >= MIN_LANGUAGE_PROB), None)
        if other is None:
            return first
        log.warning("Word confidence %.0f%% with detected language '%s'; trying '%s'", 100 * confidence, first.language, other)
        second = self._run(wav, other, duration, on_progress, cancel)
        if (second.mean_word_confidence() or 0.0) <= confidence:
            return first
        self.fallback_note = (f"The language was detected as '{first.language}', but the transcript was much clearer as '{other}', so that was used. "
                              f"Set the spoken language to '{other}' to skip the first attempt next time.")
        return second

    def _run(self, wav, language, duration, on_progress, cancel) -> Transcript:
        assert self._model is not None
        seg_iter, info = self._model.transcribe(
            str(wav), language=language, beam_size=self.beam_size, vad_filter=self.cfg.vad_filter,
            vad_parameters={"min_silence_duration_ms": 500}, word_timestamps=True,
            condition_on_previous_text=False,
        )
        if not language:
            self._language_probs = list(info.all_language_probs or [])
        total = info.duration or duration or 1.0
        segments: list[TranscriptSegment] = []
        for seg in seg_iter:
            if cancel is not None and cancel.is_set():
                raise JobCancelled("Cancelled during transcription")
            # drop hallucinated text over non-speech (a known Whisper failure mode)
            if getattr(seg, "no_speech_prob", 0) > 0.9 and getattr(seg, "avg_logprob", 0) < -1.0:
                continue
            words = [
                Word(text=w.word.strip(), start=max(0.0, w.start), end=max(w.start, w.end), prob=float(w.probability))
                for w in (seg.words or []) if w.word.strip()
            ]
            segments.append(TranscriptSegment(id=len(segments), start=max(0.0, seg.start), end=max(seg.start, seg.end),
                                              text=seg.text.strip(), words=words))
            if on_progress:
                on_progress(min(1.0, seg.end / total))
        return Transcript(
            language=info.language, language_probability=float(info.language_probability), duration=float(total),
            model=self.cfg.model, device=self.device, compute_type=self.compute_type,
            has_word_timestamps=any(s.words for s in segments), segments=segments,
        )

    def close(self) -> None:
        """Release the model (and VRAM) before the LLM stages start."""
        self._model = None
        gc.collect()


# ---- stage-level orchestration (backend-agnostic, unit-tested with a fake backend) ----------------
def transcript_cache_key(info: VideoInfo, cfg: AppConfig, imported: Path | None = None) -> str:
    t = cfg.transcription
    if imported is not None:
        return make_key("transcript-import", STAGE_VERSION, info.fingerprint, t.language, hashlib.sha1(imported.read_bytes()).hexdigest())
    return make_key("transcript", STAGE_VERSION, info.fingerprint, t.model, t.language, t.vad_filter, cfg.effective().transcription.beam_size)


def extract_audio_wav(info: VideoInfo, wav: Path, runner: FFmpegRunner, *, on_progress=None, cancel=None) -> Path:
    """16 kHz mono PCM, the format Whisper and the audio analyser both want."""
    if not info.has_audio:
        raise NoAudioError(f"{info.filename} has no audio track.")
    wav.parent.mkdir(parents=True, exist_ok=True)
    tmp = wav.with_suffix(".part.wav")
    runner.run(
        ["-y", "-loglevel", "error", "-i", info.path, "-vn", "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", *runner.thread_args(), str(tmp)],
        duration=info.duration, on_progress=on_progress, cancel=cancel,
    )
    tmp.replace(wav)
    return wav


def transcribe_video(
    info: VideoInfo, store: ProjectStore, cfg: AppConfig, runner: FFmpegRunner,
    *, backend: WhisperBackend | None = None, on_progress: ProgressFn | None = None,
    cancel: threading.Event | None = None, imported: Path | None = None,
) -> tuple[Transcript, bool]:
    """Returns (transcript, from_cache). Writes cache/transcript.json and cache/transcript.srt."""
    cache = StageCache(store.cache_dir)
    key = transcript_cache_key(info, cfg, imported)
    hit = cache.load("transcript", key)
    if hit is not None:
        try:
            return Transcript.model_validate(hit), True
        except ValueError:
            log.warning("Cached transcript failed validation; recomputing")

    report = on_progress or (lambda f, m: None)
    if imported is not None:
        report(0.0, "Reading the imported transcript")
        transcript, notes = load_import(imported, info.duration, cfg.transcription.language)
        for note in notes:
            store.add_warning(note)
        cache.save("transcript", key, transcript.model_dump(mode="json"))
        atomic_write_text(store.cache_dir / "transcript.srt", transcript.to_srt())
        return transcript, False

    wav = store.work_dir / "audio_16k.wav"
    report(0.0, "Extracting audio")
    extract_audio_wav(info, wav, runner, on_progress=lambda f: report(0.1 * f, "Extracting audio"), cancel=cancel)

    own_backend = backend is None
    backend = backend or FasterWhisperBackend(cfg, cfg.resolved_threads())
    started = time.monotonic()
    try:
        report(0.1, f"Transcribing with Whisper '{cfg.transcription.model}'")
        transcript = backend.transcribe(
            wav, language=cfg.transcription.language, duration=info.duration,
            on_progress=lambda f: report(0.1 + 0.9 * f, "Transcribing"), cancel=cancel,
        )
    finally:
        if own_backend:
            backend.close()
    note = getattr(backend, "fallback_note", None)
    if note:
        store.add_warning(note)
    if not transcript.segments or not transcript.text_in_range(0, info.duration + 1).strip():
        raise TranscriptionError(
            "No speech was detected in this video.",
            recovery="Check that the audio is audible and speech is present; try transcription.vad_filter: false or a larger model.",
        )
    log.info("Transcribed %d segments (%s, %s/%s) in %.1fs", len(transcript.segments), transcript.language,
             transcript.device, transcript.compute_type, time.monotonic() - started)
    confidence = transcript.mean_word_confidence()
    if confidence is not None and confidence < LOW_CONFIDENCE:
        store.add_warning(f"The transcript looks unreliable (average word confidence {confidence:.0%}), and clip choices depend on it. "
                          "Try a larger Whisper model, set the spoken language, or import an .srt/.vtt transcript.")
    cache.save("transcript", key, transcript.model_dump(mode="json"))
    atomic_write_text(store.cache_dir / "transcript.srt", transcript.to_srt())
    return transcript, False
