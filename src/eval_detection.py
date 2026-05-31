from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.detector import YOLODetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, required=True, help="Path to sequence images folder")
    parser.add_argument("--annotation", type=Path, required=True, help="Path to VisDrone annotation txt file for sequence")
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    # boxes are left, top, width, height
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    union = boxAArea + boxBArea - interArea
    if union <= 0:
        return 0.0
    return interArea / union


def load_annotations(annotation_file: Path):
    gt_by_frame = {}
    with annotation_file.open() as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue
            frame = int(parts[0])
            left = int(parts[2])
            top = int(parts[3])
            w = int(parts[4])
            h = int(parts[5])
            cls = int(parts[7])
            # only person (1) considered
            if cls != 1:
                continue
            gt_by_frame.setdefault(frame, []).append((left, top, w, h))
    return gt_by_frame


def main() -> None:
    args = parse_args()
    if not args.sequence.exists():
        raise FileNotFoundError(args.sequence)
    if not args.annotation.exists():
        raise FileNotFoundError(args.annotation)

    detector = YOLODetector(device=args.device)
    gt = load_annotations(args.annotation)

    frames = sorted(args.sequence.glob("*.jpg"))
    TP = FP = FN = 0
    results = []
    for idx, frame_path in enumerate(frames[: args.max_frames], start=1):
        frame = str(frame_path)
        res = detector.predict(frame)
        # extract xyxy and conf
        boxes = []
        if getattr(res, "boxes", None) is not None and res.boxes.xyxy is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else np.zeros(len(xyxy))
            for b, c in zip(xyxy, confs):
                x1, y1, x2, y2 = b
                boxes.append(((float(x1), float(y1), float(x2 - x1), float(y2 - y1)), float(c)))

        # sort boxes by score desc
        boxes.sort(key=lambda x: x[1], reverse=True)

        frame_gt = gt.get(idx, [])
        matched_gt = set()
        for (pred_box, score) in boxes:
            # find best gt
            best_i = -1
            best_iou = 0.0
            for i, g in enumerate(frame_gt):
                if i in matched_gt:
                    continue
                val = iou(pred_box, g)
                if val > best_iou:
                    best_iou = val
                    best_i = i
            if best_i >= 0 and best_iou >= args.iou:
                TP += 1
                matched_gt.add(best_i)
            else:
                FP += 1

        FN += max(0, len(frame_gt) - len(matched_gt))
        results.append({"frame": idx, "predictions": len(boxes), "gt": len(frame_gt)})

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    report = {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_frame": results,
    }
    logging.getLogger(__name__).debug(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
