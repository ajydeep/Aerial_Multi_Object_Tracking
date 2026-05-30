#!/usr/bin/env bash
# simple helper to run a short fine-tune (adjust args as needed)
python3 -m src.train --data data/yolo/data.yaml --weights yolov8n.pt --epochs 20 --imgsz 1280 --batch 8 --device cpu
