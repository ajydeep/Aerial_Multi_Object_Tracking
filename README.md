# Aerial Multi Object Tracking
Lightweight person detection and tracking on the VisDrone validation set.

## Setup
```bash
pip install -r requirements.txt
```

## Data
Place `VisDrone2019-MOT-val` at the repository root.

## Prepare Dataset

```bash
python -m src.prepare_dataset --data VisDrone2019-MOT-val --output outputs/visdrone_person_manifest.json
```

Default person class ids are `1 2`.

### Export YOLOv8 format (for fine-tuning)

```bash
python -m src.convert_to_yolo --data VisDrone2019-MOT-val --out data/yolo --val-split 0.1 --subsample 2
```

This writes `data/yolo/images` and `data/yolo/labels` and a `data/yolo/data.yaml`. The `data/` folder is not added to git by default.

## Detector and Tracker

The detector uses `yolov8n.pt` with a larger input size for small objects.
Tracking uses ByteTrack through Ultralytics.

## Render

```bash
python -m src.pipeline --source VisDrone2019-MOT-val/sequences/uav0000086_00000_v
```

By default, this saves to `outputs/<sequence_name>.mp4`.

## FPS

```bash
python -m src.benchmark --source VisDrone2019-MOT-val/sequences/uav0000086_00000_v --frames 50
```

The benchmark prints FPS and hardware details.

Example measured result (this repo run):

- FPS: `4.02`
- Frames: `50`
- Elapsed: `12.435s`
- Hardware: `Darwin 25.5.0`, `arm64` (Apple Silicon)

## Run
```bash
python -m src.main --data VisDrone2019-MOT-val
```

## Fine-tune detector (local / cloud)

Run the training wrapper (local CPU or cloud GPU). On an M1 Mac this will be slow; prefer a cloud GPU for full runs.

```bash
# local CPU (slow)
python -m src.train --data data/yolo/data.yaml --weights yolov8n.pt --epochs 20 --imgsz 1280 --batch 8 --device cpu

# or use helper
./train.sh
```

The script uses `ultralytics.YOLO.train` and writes outputs under `runs/train`.

## Summary Report

**Deliverables**
- **Code**: This repo contains the detection, tracking, rendering and helper scripts. Key files: [src/detector.py](src/detector.py), [src/tracker.py](src/tracker.py), [src/render.py](src/render.py), [src/pipeline.py](src/pipeline.py).
- **Output Video**: processed clip with single box per track, deterministic color per ID, ID label and short trail: [outputs/uav0000086_00000_v_finetuned.mp4](outputs/uav0000086_00000_v_finetuned.mp4).
- **Predictions**: MOT-format predictions for scoring: [outputs/pred_mot_uav0000086_00000_v.txt](outputs/pred_mot_uav0000086_00000_v.txt).

**1) Architecture & rationale**
- **Detector**: `YOLOv8n` (ultralytics) fine-tuned on VisDrone person labels. Rationale: very small footprint (~6 MB), fast inference, and straightforward export to ONNX/TensorRT for edge deployment.
- **Tracker**: ByteTrack (via Ultralytics `model.track`) — provides motion+appearance association, low latency, and stable persistent IDs.

**2) Small-object handling**
- **Training / Inference image size**: train and run inference with larger input (e.g., 1280 px shorter side) so small persons are more than a few pixels tall.
- **Filtering**: restrict to person class IDs at the detector level to reduce false positives.
- **Augmentation / sampling**: oversample frames with small/partial people, enable mosaic and mixup during training to improve small-object robustness.

**3) ID-switching & occlusion handling**
- **ByteTrack persistence**: configured to persist short-term lost tracks and re-associate on re-entry using motion prediction + appearance features.
- **Track smoothing**: we render short trails and use Kalman-based motion prediction to reduce jitter and brief switches.
- **Practical mitigations**: (a) tune re-id / match thresholds; (b) apply simple frame stabilization to compensate for drone ego-motion; (c) increase history length before terminating a track to avoid premature ID death during occlusion.

**4) Edge deployment plan (NVIDIA Jetson / ARM)**
- **Model export**: export `YOLOv8n` to ONNX, then convert to TensorRT engines (FP16 / INT8) for Jetson. Use `ultralytics` export utilities or `torch.onnx` -> `trtexec`.
- **Quantization & pruning**: prefer FP16 first; if needed, calibrate and deploy INT8 for further speed/size; consider pruning to reduce conv filters.
- **Pipeline**: run detector in TensorRT, tracker in lightweight C++/Python binding (ByteTrack translated), process frames at reduced resolution or lower frame rate for real-time constraints.

**5) How to run (reproduce outputs)**
- Prepare environment and data (see **Setup** and **Data** sections above).
- Render a sequence (writes to `outputs/` by default):

```bash
python -m src.pipeline --source VisDrone2019-MOT-val/sequences/uav0000086_00000_v
```

- Benchmark (sample):

```bash
python -m src.benchmark --source VisDrone2019-MOT-val/sequences/uav0000086_00000_v --frames 50
```

- Compute MOT metrics (optional): install `motmetrics` then run the eval script using the provided predictions file:

```bash
pip install motmetrics
python -m src.eval_tracking --sequence VisDrone2019-MOT-val/sequences/uav0000086_00000_v --output outputs/pred_mot_uav0000086_00000_v.txt --max-frames 200
# then use motmetrics to compute MOTA/IDF1 against ground-truth
```

**6) Notes & repository artifacts**
- Single-box per track and deterministic color/ID rendering implemented in [src/render.py](src/render.py).
- Output artifacts:
	- [outputs/uav0000086_00000_v_finetuned.mp4](outputs/uav0000086_00000_v_finetuned.mp4)
	- [outputs/pred_mot_uav0000086_00000_v.txt](outputs/pred_mot_uav0000086_00000_v.txt)
	- [outputs/benchmark_report.json](outputs/benchmark_report.json)
	- [runs/detect/runs/train/train-2/weights/best.pt](runs/detect/runs/train/train-2/weights/best.pt)

If you'd like, I can (a) re-render the sequence with the new sharp ID labels for verification, (b) install `motmetrics` and compute MOTA/IDF1 locally, or (c) prepare a push to a private Git repo. Which should I do next?
