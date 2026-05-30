from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.detector import DetectorConfig, YOLODetector


@dataclass(frozen=True)
class TrackerConfig:
    tracker_yaml: Path = Path("bytetrack.yaml")
    detector: DetectorConfig = DetectorConfig()


class ByteTrackTracker:
    def __init__(self, config: TrackerConfig | None = None, device: str | None = None) -> None:
        self.config = config or TrackerConfig()
        self.device = device
        self.detector = YOLODetector(self.config.detector, device=device)
        self._model = None

    def _load_model(self):
        if self._model is None:
            self._model = self.detector._load_model()
        return self._model

    def track(self, frame):
        model = self._load_model()
        return model.track(
            frame,
            persist=True,
            tracker=str(self.config.tracker_yaml),
            imgsz=self.config.detector.imgsz,
            conf=self.config.detector.conf,
            iou=self.config.detector.iou,
            classes=list(self.config.detector.person_class_ids),
            max_det=self.config.detector.max_det,
            device=self.device,
            verbose=False,
        )[0]