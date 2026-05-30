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

## Run
```bash
python -m src.main --data VisDrone2019-MOT-val
```
