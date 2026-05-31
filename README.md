# Aerial Multi Object Tracking
Lightweight object detection and tracking on drone footage (VisDrone validation sequences).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data
Place `VisDrone2019-MOT-val` at the repository root.

## Detector and Tracker

The detector uses `yolov8n.pt` with a larger input size for small objects.
Tracking uses ByteTrack through Ultralytics.

Default tracking class is COCO `person` (`--classes 0`). To track vehicles, pass COCO class ids like car `2`, motorcycle `3`, bus `5`, truck `7`.

## Render (Output Video)

`--source` can be a directory of `*.jpg` frames (VisDrone sequence folder) or a video file.

```bash
python -m src.pipeline --source VisDrone2019-MOT-val/sequences/uav0000086_00000_v --output outputs/output_video.mp4
```



## Summary Report

**Deliverables**
- **Code**: This repo contains the detection, tracking, rendering and helper scripts. Key files: [src/detector.py](src/detector.py), [src/tracker.py](src/tracker.py), [src/render.py](src/render.py), [src/pipeline.py](src/pipeline.py).
- **Output Video**: `outputs/output_video.mp4` (bounding boxes, unique IDs, and short trajectory tail per track).
- **Summary Report**: this section.

**1) Architecture & rationale**
- **Detector**: `YOLOv8n` (ultralytics). Baseline uses COCO pre-trained weights (`yolov8n.pt`) and a larger input size to improve small-object recall; optional fine-tuning is supported via [src/train.py](src/train.py).
- **Tracker**: ByteTrack (via Ultralytics `model.track`) — Kalman-based motion model + IoU matching with a track buffer for short-term persistence.

**2) Small-object handling**
- **Inference image size**: run inference with a larger input (default `imgsz=1280`) so small objects occupy more pixels.
- **Filtering**: restrict to the target classes at the detector level (`--classes ...`) to reduce false positives.
- **Optional fine-tuning**: export VisDrone to YOLO format and fine-tune `YOLOv8n` to improve small-object recall.

**3) ID-switching & occlusion handling**
- **Persistence through occlusions**: ByteTrack keeps tracks alive for a short buffer and uses motion prediction to re-associate when detections return.
- **Reduce missed detections**: higher inference resolution and tuned confidence/IoU thresholds reduce dropouts that often trigger ID switches.
- **Ego-motion**: the motion model helps for moderate camera motion; for aggressive drone movement, a practical next step is global motion compensation (frame-to-frame stabilization) before tracking.

**4) Edge deployment plan (NVIDIA Jetson / ARM)**
- **Model export**: export `YOLOv8n` to ONNX, then convert to TensorRT engines (FP16 / INT8) for Jetson. Use `ultralytics` export utilities or `torch.onnx` -> `trtexec`.
- **Quantization & pruning**: prefer FP16 first; if needed, calibrate and deploy INT8 for further speed/size; consider pruning to reduce conv filters.
- **Pipeline**: run detector in TensorRT, tracker in lightweight C++/Python binding (ByteTrack translated), process frames at reduced resolution or lower frame rate for real-time constraints.
