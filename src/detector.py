from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DetectorConfig:
    weights: Path = Path("yolov8n.pt")
    imgsz: int = 1280
    conf: float = 0.35
    iou: float = 0.5
    max_det: int = 300
    person_class_ids: tuple[int, ...] = (0,)


class YOLODetector:
    def __init__(self, config: DetectorConfig | None = None, device: str | None = None) -> None:
        self.config = config or DetectorConfig()
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.config.weights))
        return self._model

    def predict(self, frame):
        model = self._load_model()
        return model.predict(
            frame,
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            iou=self.config.iou,
            classes=list(self.config.person_class_ids),
            max_det=self.config.max_det,
            device=self.device,
            verbose=False,
        )[0]