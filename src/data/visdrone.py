from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Detection:
    frame_id: int
    track_id: int
    left: int
    top: int
    width: int
    height: int
    score: float
    category_id: int
    truncation: int
    occlusion: int

    def to_dict(self) -> dict:
        return asdict(self)


def parse_annotations(annotation_file: Path, class_ids: set[int]) -> list[Detection]:
    detections: list[Detection] = []
    with annotation_file.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 10:
                continue
            detection = Detection(
                frame_id=int(row[0]),
                track_id=int(row[1]),
                left=int(row[2]),
                top=int(row[3]),
                width=int(row[4]),
                height=int(row[5]),
                score=float(row[6]),
                category_id=int(row[7]),
                truncation=int(row[8]),
                occlusion=int(row[9]),
            )
            if detection.category_id in class_ids:
                detections.append(detection)
    return detections


def iter_sequence_files(data_root: Path) -> list[Path]:
    return sorted((data_root / "annotations").glob("*.txt"))


def build_person_manifest(data_root: Path, class_ids: set[int]) -> dict:
    sequences = []
    for annotation_file in iter_sequence_files(data_root):
        sequence_name = annotation_file.stem
        sequences.append(
            {
                "sequence": sequence_name,
                "frames_dir": str(data_root / "sequences" / sequence_name),
                "annotation_file": str(annotation_file),
                "detections": [detection.to_dict() for detection in parse_annotations(annotation_file, class_ids)],
            }
        )
    return {"data_root": str(data_root), "class_ids": sorted(class_ids), "sequences": sequences}