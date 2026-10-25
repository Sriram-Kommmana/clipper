"""Hardware / dependency diagnostic.  Run on the machine that will do the processing:

    python scripts/check_environment.py            # human readable
    python scripts/check_environment.py --json     # machine readable
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.hardware_detection import build_report, format_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    args = ap.parse_args()
    report = build_report(args.ollama_host)
    print(json.dumps(report, indent=2) if args.json else format_report(report))
    ff = report["ffmpeg"]
    return 0 if ff.get("ffmpeg") and ff.get("ffprobe") else 1


if __name__ == "__main__":
    raise SystemExit(main())
