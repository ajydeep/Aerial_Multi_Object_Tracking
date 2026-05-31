from __future__ import annotations

import argparse
import logging
from pathlib import Path
import csv
from typing import Iterable

import cv2

from src.detector import DetectorConfig
from src.tracker import ByteTrackTracker, TrackerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        "--sequence",
        dest="source",
        type=Path,
        required=True,
        help="Path to sequence images folder or a video file",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/pred_mot.txt"))
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--weights", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--classes", type=int, nargs="+", default=[0])
    parser.add_argument("--tracker", type=Path, default=Path("bytetrack.yaml"))
    return parser.parse_args()


def write_mot_txt(rows, out: Path):
    # MOT plain text: frame, id, left, top, width, height, score, class, -,-
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)


def iter_source_frames(source: Path, max_frames: int) -> Iterable[tuple[int, object]]:
    if source.is_dir():
        frames = sorted(source.glob("*.jpg"))
        for idx, frame_path in enumerate(frames[:max_frames], start=1):
            yield idx, str(frame_path)
        return

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(source)
    idx = 0
    while idx < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        idx += 1
        yield idx, frame
    cap.release()


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

    rows = []
    for idx, frame in iter_source_frames(args.source, args.max_frames):
        res = tracker.track(frame)
        boxes = getattr(res, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else [0] * len(xyxy)
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [0.0] * len(xyxy)
        for (b, tid, c) in zip(xyxy, ids, confs):
            x1, y1, x2, y2 = b
            left, top = int(x1), int(y1)
            width, height = int(x2 - x1), int(y2 - y1)
            rows.append([idx, int(tid), left, top, width, height, float(c), 1, -1, -1])

    write_mot_txt(rows, args.output)
    logging.getLogger(__name__).info("Wrote MOT predictions: %s", args.output)

    # Try to run motmetrics if installed
    try:
        import motmetrics as mm  # type: ignore

        logging.getLogger(__name__).info("motmetrics available: you can compute MOTA/IDF1 using the library. See README.")
    except Exception:
        logging.getLogger(__name__).info("motmetrics not available. Install with: pip install motmetrics")


if __name__ == "__main__":
    main()
