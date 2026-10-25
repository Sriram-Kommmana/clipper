"""Safe FFmpeg/FFprobe execution: argument lists only (never a shell), progress, cancellation, timeouts."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from src.core.exceptions import FFmpegError, FFmpegNotFoundError, JobCancelled

ProgressFn = Callable[[float], None]

_WIN = os.name == "nt"


def find_binary(name: str, configured: str | None = None) -> str:
    """Locate ffmpeg/ffprobe: explicit config path > <NAME>_PATH env var > PATH."""
    candidates = [configured, os.environ.get(f"{name.upper()}_PATH")]
    for c in candidates:
        if c:
            p = Path(c)
            if p.is_dir():
                p = p / (name + (".exe" if _WIN else ""))
            if p.is_file():
                return str(p)
            raise FFmpegNotFoundError(f"Configured {name} path does not exist: {c}")
    found = shutil.which(name)
    if not found:
        raise FFmpegNotFoundError(f"'{name}' was not found on PATH.")
    return found


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str


class FFmpegRunner:
    def __init__(
        self,
        ffmpeg: str | None = None,
        ffprobe: str | None = None,
        *,
        low_priority: bool = False,
        threads: int | None = None,
    ):
        self.ffmpeg = find_binary("ffmpeg", ffmpeg)
        self.ffprobe = find_binary("ffprobe", ffprobe)
        self.low_priority = low_priority
        self.threads = threads

    # ---- ffprobe -------------------------------------------------------------------------------
    def probe_json(self, path: Path, timeout: float = 60) -> dict:
        cmd = [self.ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL, **_popen_kwargs(False)
            )
        except subprocess.TimeoutExpired as exc:
            raise FFmpegError(f"ffprobe timed out after {timeout:.0f}s on {path}") from exc
        if proc.returncode != 0:
            raise FFmpegError(
                f"ffprobe could not read {path.name}",
                details=proc.stderr.decode("utf-8", "replace")[-1500:],
            )
        try:
            return json.loads(proc.stdout.decode("utf-8", "replace"))
        except ValueError as exc:
            raise FFmpegError("ffprobe returned unparsable output") from exc

    # ---- ffmpeg --------------------------------------------------------------------------------
    def build(self, args: Sequence[str], *, progress: bool = False) -> list[str]:
        cmd = [self.ffmpeg, "-hide_banner", "-nostdin"]
        if progress:
            cmd += ["-progress", "pipe:1", "-nostats"]
        cmd += [str(a) for a in args]
        return cmd

    def run(
        self,
        args: Sequence[str],
        *,
        duration: float | None = None,
        on_progress: ProgressFn | None = None,
        cancel: threading.Event | None = None,
        timeout: float | None = None,
        cwd: Path | None = None,
        keep_stderr: bool = False,
    ) -> RunResult:
        """Run ffmpeg with `args` (without the executable). Raises FFmpegError on non-zero exit.

        `duration` (seconds of *output*) enables a 0-1 progress callback; `cancel` aborts the process.
        """
        cmd = self.build(args, progress=on_progress is not None and duration is not None)
        err = tempfile.TemporaryFile()  # stderr -> file: no pipe-buffer deadlocks, full text available
        out_lines: list[str] = []
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=err, stdin=subprocess.DEVNULL,
                cwd=str(cwd) if cwd else None, text=True, encoding="utf-8", errors="replace",
                **_popen_kwargs(self.low_priority),
            )
        except OSError as exc:
            err.close()
            raise FFmpegError(f"Could not start ffmpeg: {exc}") from exc
        self._apply_priority(proc)

        def pump() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                out_lines.append(line)
                if on_progress and duration and line.startswith("out_time_"):
                    try:
                        # out_time_us / out_time_ms are both microseconds in ffmpeg's progress output
                        micros = int(line.split("=", 1)[1])
                        on_progress(max(0.0, min(1.0, micros / 1e6 / duration)))
                    except ValueError:
                        pass

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        started = time.monotonic()
        try:
            while proc.poll() is None:
                if cancel is not None and cancel.is_set():
                    proc.terminate()
                    try:
                        proc.wait(5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise JobCancelled("Cancelled while FFmpeg was running")
                if timeout and time.monotonic() - started > timeout:
                    proc.kill()
                    raise FFmpegError(f"ffmpeg timed out after {timeout:.0f}s")
                time.sleep(0.05)
            reader.join(2)
            err.seek(0)
            stderr = err.read().decode("utf-8", "replace")
        finally:
            if proc.poll() is None:
                proc.kill()
            err.close()
        if proc.returncode != 0:
            raise FFmpegError(
                f"ffmpeg failed (exit code {proc.returncode})",
                details=stderr[-2000:],
            )
        if on_progress:
            on_progress(1.0)
        return RunResult(proc.returncode, "".join(out_lines), stderr if keep_stderr else stderr[-2000:])

    def _apply_priority(self, proc: subprocess.Popen) -> None:
        if not self.low_priority:
            return
        try:
            import psutil

            p = psutil.Process(proc.pid)
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if _WIN else 10)
        except Exception:  # priority is best-effort
            pass

    def thread_args(self) -> list[str]:
        return ["-threads", str(self.threads)] if self.threads else []


def _popen_kwargs(low_priority: bool) -> dict:
    if _WIN:
        flags = 0x08000000  # CREATE_NO_WINDOW
        if low_priority:
            flags |= 0x00004000  # BELOW_NORMAL_PRIORITY_CLASS
        return {"creationflags": flags}
    return {}
