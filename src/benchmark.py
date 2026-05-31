from __future__ import annotations

import argparse
import json
import logging
import platform
import tempfile
import time
from pathlib import Path

from src.render import render_sequence
from src.tracker import ByteTrackTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def hardware_info() -> dict:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(args.source)

    tracker = ByteTrackTracker(device=args.device)
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / "benchmark.mp4"
        start = time.perf_counter()
        render_sequence(args.source, output_file, tracker, max_frames=args.frames)
        elapsed = time.perf_counter() - start

    fps = args.frames / elapsed if elapsed else 0.0
    report = {
        "source": str(args.source),
        "frames": args.frames,
        "elapsed_seconds": round(elapsed, 3),
        "fps": round(fps, 2),
        "hardware": hardware_info(),
    }
    logging.getLogger(__name__).debug(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()