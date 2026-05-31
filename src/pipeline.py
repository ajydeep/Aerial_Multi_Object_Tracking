from __future__ import annotations

import argparse
from pathlib import Path
import logging

from src.detector import DetectorConfig
from src.render import RenderConfig, render_source
from src.tracker import ByteTrackTracker, TrackerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Sequence folder of *.jpg frames or a video file")
    parser.add_argument("--output", type=Path, default=Path("result.mp4"), help="Output mp4 path")
    parser.add_argument("--device", type=str, default=None, help="ultralytics device, e.g. cpu, cuda:0, mps")

    parser.add_argument("--weights", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--classes", type=int, nargs="+", default=[0], help="COCO class ids to track (default: person=0)")
    parser.add_argument("--tracker", type=Path, default=Path("bytetrack.yaml"), help="Tracker yaml name or path")

    parser.add_argument("--trail-length", type=int, default=20)
    parser.add_argument("--fps", type=float, default=30.0, help="Used for image sequences; for videos input FPS is used")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def next_output_path(output_file: Path) -> Path:
    if not output_file.exists():
        return output_file

    suffix = output_file.suffix
    stem = output_file.stem
    parent = output_file.parent
    index = 1
    while True:
        candidate = parent / f"{stem}{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


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

    # If the user didn't provide an explicit output, default to outputs/<source_basename>.mp4
    if args.output == Path("result.mp4"):
        basename = args.source.name if args.source.is_dir() else args.source.stem
        default_out = Path("outputs") / f"{basename}.mp4"
        output_file = next_output_path(default_out)
    else:
        output_file = args.output

    render_config = RenderConfig(trail_length=args.trail_length, fps=args.fps)
    render_source(args.source, output_file, tracker, config=render_config, max_frames=args.max_frames)
    logging.getLogger(__name__).info("Wrote output video: %s", output_file)


if __name__ == "__main__":
    main()