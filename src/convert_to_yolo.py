from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2

from src.data.visdrone import build_person_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--subsample", type=int, default=2, help="Keep every Nth frame (1 = all)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def convert(data_root: Path, out: Path, val_split: float = 0.1, subsample: int = 2, seed: int = 42) -> None:
    random.seed(seed)
    manifest = build_person_manifest(data_root, {1})

    labels_dir = out / "labels"
    images_dir = out / "images"
    ensure_dir(labels_dir / "train")
    ensure_dir(labels_dir / "val")
    ensure_dir(images_dir / "train")
    ensure_dir(images_dir / "val")

    for seq in manifest["sequences"]:
        frames_dir = Path(seq["frames_dir"])
        # collect detections by frame
        dets_by_frame: dict[int, list] = {}
        for d in seq["detections"]:
            dets_by_frame.setdefault(d["frame_id"], []).append(d)

        frame_files = sorted(frames_dir.glob("*.jpg"))
        for idx, frame_path in enumerate(frame_files, start=1):
            if subsample > 1 and (idx % subsample) != 1:
                continue
            subset = "train" if random.random() > val_split else "val"
            rel_name = f"{seq['sequence']}_{idx:06d}.jpg"
            dest_image = images_dir / subset / rel_name
            shutil.copy2(frame_path, dest_image)

            detections = dets_by_frame.get(idx, [])
            label_lines = []
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
            for d in detections:
                left, top, w, h = d["left"], d["top"], d["width"], d["height"]
                x_center = (left + w / 2) / img_w
                y_center = (top + h / 2) / img_h
                w_n = w / img_w
                h_n = h / img_h
                label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_n:.6f} {h_n:.6f}\n")

            label_file = labels_dir / subset / (dest_image.stem + ".txt")
            label_file.write_text("".join(label_lines))

    # write data.yaml
    data_yaml = {
        "path": str(out),
        "train": str(images_dir / "train"),
        "val": str(images_dir / "val"),
        "nc": 1,
        "names": ["person"],
    }
    (out / "data.yaml").write_text(json.dumps(data_yaml, indent=2))


if __name__ == "__main__":
    args = parse_args()
    convert(args.data, args.out, val_split=args.val_split, subsample=args.subsample, seed=args.seed)
