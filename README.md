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
python -m src.pipeline --source VisDrone2019-MOT-val/sequences/uav0000086_00000_v --output outputs/uav0000086_00000_v.mp4
```

## FPS

```bash
python -m src.benchmark --source VisDrone2019-MOT-val/sequences/uav0000086_00000_v --frames 50
```

The benchmark prints FPS and hardware details.

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
