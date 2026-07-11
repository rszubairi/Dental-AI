"""Train the tooth detection + FDI numbering model (Stage 1, YOLOv8).

Requires a YOLO-format dataset such as the one produced by
datasets/dentex/prepare_stage1_dataset.py (see that script and
datasets/tooth_detection/README.md for format details).

Usage:
    python train_tooth_detection.py --data ../../../datasets/dentex/yolo/dataset.yaml
"""

from __future__ import annotations

import argparse

import torch
from ultralytics import YOLO

# Batch size thresholds are conservative to keep training stable on <8GB GPUs.
VRAM_BATCH_THRESHOLD_GB = 8
BATCH_SIZE_LOW_VRAM = 4
BATCH_SIZE_HIGH_VRAM = 8

# lr=1e-3 decaying via cosine schedule to 1e-5 -> lrf is the final/initial lr ratio.
LEARNING_RATE = 1e-3
FINAL_LR_FRACTION = 1e-5 / 1e-3
WEIGHT_DECAY = 5e-4


def detect_batch_size() -> int:
    if not torch.cuda.is_available():
        return BATCH_SIZE_LOW_VRAM
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return BATCH_SIZE_HIGH_VRAM if total_vram_gb >= VRAM_BATCH_THRESHOLD_GB else BATCH_SIZE_LOW_VRAM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to YOLO dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--base-model", default="yolov8s.pt")
    parser.add_argument("--batch", type=int, default=None, help="Overrides VRAM auto-detection")
    parser.add_argument("--project", default="../models/tooth_detection/runs/stage1")
    args = parser.parse_args()

    batch = args.batch if args.batch is not None else detect_batch_size()

    model = YOLO(args.base_model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=batch,
        project=args.project,
        name="train",
        save_period=5,
        amp=True,  # FP16 mixed precision
        optimizer="AdamW",
        lr0=LEARNING_RATE,
        lrf=FINAL_LR_FRACTION,
        cos_lr=True,
        weight_decay=WEIGHT_DECAY,
        # Conservative augmentation: OPGs have fixed anatomical orientation and cannot
        # be freely rotated/flipped vertically or mixed via mosaic without breaking
        # clinical context.
        fliplr=0.3,
        flipud=0.0,
        degrees=5.0,
        translate=0.05,
        scale=0.10,
        hsv_v=0.30,
        hsv_h=0.0,
        hsv_s=0.0,
        mosaic=0.0,
    )


if __name__ == "__main__":
    main()
