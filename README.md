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
