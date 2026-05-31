from __future__ import annotations

import argparse
import json
import logging
import platform
import tempfile
import time
from pathlib import Path

from src.detector import DetectorConfig
from src.render import render_source
from src.tracker import ByteTrackTracker, TrackerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Sequence folder of *.jpg frames or a video file")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark_report.json"))

    parser.add_argument("--weights", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--classes", type=int, nargs="+", default=[0])
    parser.add_argument("--tracker", type=Path, default=Path("bytetrack.yaml"))
    return parser.parse_args()


def hardware_info() -> dict:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(args.source)

    detector_config = DetectorConfig(
        weights=args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        person_class_ids=tuple(args.classes),
    )
    tracker_config = TrackerConfig(tracker_yaml=args.tracker, detector=detector_config)
    tracker = ByteTrackTracker(config=tracker_config, device=args.device)
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / "benchmark.mp4"
        start = time.perf_counter()
        render_source(args.source, output_file, tracker, max_frames=args.frames)
        elapsed = time.perf_counter() - start

    fps = args.frames / elapsed if elapsed else 0.0
    report = {
        "source": str(args.source),
        "frames": args.frames,
        "elapsed_seconds": round(elapsed, 3),
        "fps": round(fps, 2),
        "hardware": hardware_info(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    logging.getLogger(__name__).info(json.dumps(report, indent=2))
    logging.getLogger(__name__).info("Wrote report: %s", args.output)


if __name__ == "__main__":
    main()