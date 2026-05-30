from __future__ import annotations

import argparse
from pathlib import Path
import csv
import tempfile
import subprocess

from src.tracker import ByteTrackTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, required=True, help="Path to sequence images folder")
    parser.add_argument("--output", type=Path, default=Path("outputs/pred_mot.txt"))
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def write_mot_txt(rows, out: Path):
    # MOT plain text: frame, id, left, top, width, height, score, class, -,-
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)


def main() -> None:
    args = parse_args()
    if not args.sequence.exists():
        raise FileNotFoundError(args.sequence)

    tracker = ByteTrackTracker(device=args.device)
    frames = sorted(args.sequence.glob("*.jpg"))
    rows = []
    for idx, frame_path in enumerate(frames[: args.max_frames], start=1):
        frame = str(frame_path)
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
    print(f"Wrote {args.output}")

    # Try to run motmetrics if installed
    try:
        import motmetrics as mm  # type: ignore

        print("motmetrics available: you can compute MOTA/IDF1 using the library. See README.")
    except Exception:
        print("motmetrics not available. Install with: pip install motmetrics")


if __name__ == "__main__":
    main()
