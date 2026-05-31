import argparse
import logging
from pathlib import Path

from src.detector import DetectorConfig
from src.eval_tracking import iter_source_frames, write_mot_txt
from src.render import RenderConfig, render_source
from src.tracker import ByteTrackTracker, TrackerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an annotated tracking video and export MOT-format predictions.")
    parser.add_argument("--source", type=Path, required=True, help="Sequence folder of *.jpg frames or a video file")
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--output-mot", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
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
    return parser.parse_args()


def default_outputs(source: Path) -> tuple[Path, Path]:
    basename = source.name if source.is_dir() else source.stem
    return (Path("outputs") / f"{basename}.mp4", Path("outputs") / f"pred_mot_{basename}.txt")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(args.source)

    output_video, output_mot = default_outputs(args.source)
    if args.output_video is not None:
        output_video = args.output_video
    if args.output_mot is not None:
        output_mot = args.output_mot

    detector_config = DetectorConfig(
        weights=args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        person_class_ids=tuple(args.classes),
    )
    tracker_config = TrackerConfig(tracker_yaml=args.tracker, detector=detector_config)

    # 1) Render annotated video
    tracker_for_video = ByteTrackTracker(config=tracker_config, device=args.device)
    render_config = RenderConfig(trail_length=args.trail_length, fps=args.fps)
    render_source(args.source, output_video, tracker_for_video, config=render_config, max_frames=args.max_frames)
    logging.getLogger(__name__).info("Wrote output video: %s", output_video)

    # 2) Export MOT predictions (fresh tracker state)
    tracker_for_mot = ByteTrackTracker(config=tracker_config, device=args.device)
    rows = []
    max_frames = args.max_frames if args.max_frames is not None else 500_000_000
    for frame_index, frame in iter_source_frames(args.source, int(max_frames)):
        res = tracker_for_mot.track(frame)
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
            rows.append([int(frame_index), int(tid), left, top, width, height, float(c), 1, -1, -1])

    write_mot_txt(rows, output_mot)
    logging.getLogger(__name__).info("Wrote MOT predictions: %s", output_mot)


if __name__ == "__main__":
    main()