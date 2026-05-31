from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import random

import cv2

from src.tracker import ByteTrackTracker


@dataclass(frozen=True)
class RenderConfig:
    trail_length: int = 20
    fps: float = 30.0


def color_for_id(track_id: int) -> tuple[int, int, int]:
    rng = random.Random(int(track_id))
    # Bright-ish BGR for visibility.
    b = 64 + rng.randint(0, 191)
    g = 64 + rng.randint(0, 191)
    r = 64 + rng.randint(0, 191)
    return (b, g, r)


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

        color = color_for_id(track_id)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        label = f"ID {track_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        line_type = cv2.LINE_AA

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = left
        text_y = top - 6
        if text_y - text_h - baseline < 0:
            text_y = bottom + text_h + 6

        height, width = frame.shape[:2]
        text_x = max(2, min(int(text_x), max(2, width - text_w - 2)))
        text_y = max(text_h + baseline + 2, min(int(text_y), max(text_h + baseline + 2, height - 2)))

        cv2.putText(frame, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2, lineType=line_type)
        cv2.putText(frame, label, (text_x, text_y), font, font_scale, color, thickness, lineType=line_type)

        trail_points = list(trails[track_id])[-trail_length:]
        for start, end in zip(trail_points, trail_points[1:]):
            cv2.line(frame, start, end, color, 2)


def render_video(
    video_file: Path,
    output_file: Path,
    tracker: ByteTrackTracker,
    config: RenderConfig | None = None,
    max_frames: int | None = None,
) -> None:
    render_config = config or RenderConfig()
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise FileNotFoundError(video_file)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = render_config.fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        ret, frame0 = cap.read()
        if not ret or frame0 is None:
            cap.release()
            raise ValueError(f"Unable to read first frame from {video_file}")
        height, width = frame0.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_file), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))

    trails = defaultdict(lambda: deque(maxlen=render_config.trail_length))
    frame_index = 0
    while True:
        if max_frames is not None and frame_index >= max_frames:
            break
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        result = tracker.track(frame)
        items = get_track_items(result)
        draw_tracks(frame, items, trails, render_config.trail_length)
        writer.write(frame)
        frame_index += 1

    writer.release()
    cap.release()


def render_source(
    source: Path,
    output_file: Path,
    tracker: ByteTrackTracker,
    config: RenderConfig | None = None,
    max_frames: int | None = None,
) -> None:
    if source.is_dir():
        render_sequence(source, output_file, tracker, config=config, max_frames=max_frames)
        return
    if source.is_file():
        render_video(source, output_file, tracker, config=config, max_frames=max_frames)
        return
    raise FileNotFoundError(source)


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