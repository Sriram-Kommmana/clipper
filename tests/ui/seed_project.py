import sys, subprocess
sys.path.insert(0, "/home/user/clipper")
from pathlib import Path
from config import load_config
from src.rendering.ffmpeg_runner import FFmpegRunner
from src.ingestion.video_probe import inspect_video
from src.core.project_manager import create_or_open_project
from src.core.job_manager import JobManager
from tests.fakes import ScriptedLLM
from tests.integration.test_pipeline import FakeWhisper, llm_responder
S = Path(sys.argv[1])  # scratch folder
src = S / "demo.mp4"
subprocess.run(["ffmpeg","-y","-loglevel","error","-f","lavfi","-i","testsrc2=s=1280x720:r=25:d=24","-f","lavfi","-i","sine=f=220:d=24","-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac",str(src)],check=True)
r = FFmpegRunner(); info = inspect_video(src, r)
store = create_or_open_project(info, S / "uiproj")
rep = JobManager(store, load_config(local_path=None), r, llm=ScriptedLLM(llm_responder()), whisper=FakeWhisper()).run()
print("seeded", rep.success, store.root)
