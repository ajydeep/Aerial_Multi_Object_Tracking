from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2

from src.tracker import ByteTrackTracker


@dataclass(frozen=True)
class RenderConfig:
    trail_length: int = 20
    fps: float = 30.0


def list_frames(sequence_dir: Path) -> list[Path]:
    return sorted(sequence_dir.glob("*.jpg"))


def get_track_items(result):
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return []

    ids = getattr(boxes, "id", None)
    if ids is None:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    track_ids = ids.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None

    items = []
    for index, box in enumerate(xyxy):
        score = float(confs[index]) if confs is not None else 0.0
        items.append((track_ids[index], box, score))
    return items


def draw_tracks(frame, items, trails, trail_length):
    for track_id, box, score in items:
        left, top, right, bottom = map(int, box)
        center_x = int((left + right) / 2)
        center_y = int((top + bottom) / 2)
        trails[track_id].append((center_x, center_y))

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {track_id} {score:.2f}",
            (left, max(0, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        trail_points = list(trails[track_id])[-trail_length:]
        for start, end in zip(trail_points, trail_points[1:]):
            cv2.line(frame, start, end, (255, 180, 0), 2)


def render_sequence(
    sequence_dir: Path,
    output_file: Path,
    tracker: ByteTrackTracker,
    config: RenderConfig | None = None,
    max_frames: int | None = None,
) -> None:
    render_config = config or RenderConfig()
    frames = list_frames(sequence_dir)
    if not frames:
        raise FileNotFoundError(sequence_dir)

    first_frame = cv2.imread(str(frames[0]))
    if first_frame is None:
        raise ValueError(frames[0])

    height, width = first_frame.shape[:2]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_file), cv2.VideoWriter_fourcc(*"mp4v"), render_config.fps, (width, height))

    trails = defaultdict(lambda: deque(maxlen=render_config.trail_length))
    for frame_index, frame_path in enumerate(frames):
        if max_frames is not None and frame_index >= max_frames:
            break
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        result = tracker.track(frame)
        items = get_track_items(result)
        draw_tracks(frame, items, trails, render_config.trail_length)
        writer.write(frame)

    writer.release()