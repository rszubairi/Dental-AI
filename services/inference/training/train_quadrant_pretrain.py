"""Pretrain a YOLOv8s backbone on the DENTEX quadrant subset (coarse 4-class jaw
quadrant localization, 693 images) before fine-tuning on the 32-class FDI task.

This is a separate label space from Stage 1's 32-class FDI detector -- quadrant 1-4
vs FDI 11-48 -- so it is trained as its own small model. Only its backbone/neck
weights are meant to be reused: pass its checkpoint as --base-model to
train_tooth_detection.py, and Ultralytics will transfer matching layers while
reinitializing the detection head to match the target nc, the same way loading
yolov8s.pt's 80-class COCO checkpoint works today.

Usage:
    python train_quadrant_pretrain.py --data ../../../datasets/dentex/yolo_quadrant_pretrain/dataset.yaml
    python train_tooth_detection.py --data ... --base-model runs/pretrain_quadrant/train/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from pipeline_state import is_step_complete, mark_step_complete, reset_step

STEP_NAME = "pretrain_quadrant"

VRAM_BATCH_THRESHOLD_GB = 8
BATCH_SIZE_LOW_VRAM = 4
BATCH_SIZE_HIGH_VRAM = 8


def detect_batch_size() -> int:
    if not torch.cuda.is_available():
        return BATCH_SIZE_LOW_VRAM
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return BATCH_SIZE_HIGH_VRAM if total_vram_gb >= VRAM_BATCH_THRESHOLD_GB else BATCH_SIZE_LOW_VRAM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to YOLO dataset.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--base-model", default="yolov8s.pt")
    parser.add_argument("--batch", type=int, default=None, help="Overrides VRAM auto-detection")
    parser.add_argument("--project", default="runs/pretrain_quadrant")
    parser.add_argument(
        "--reset-step",
        action="store_true",
        help="Delete existing pretrain output and retrain from scratch, even if already complete",
    )
    args = parser.parse_args()

    project_dir = Path(args.project)

    if args.reset_step:
        reset_step(STEP_NAME, project_dir)
    elif is_step_complete(STEP_NAME):
        print(
            f"{STEP_NAME} already complete per .dental_ai_state.json. "
            "Pass --reset-step to retrain from scratch."
        )
        return

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
        amp=True,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=1e-5 / 1e-3,
        cos_lr=True,
        weight_decay=5e-4,
        # Same conservative OPG augmentation policy as Stage 1.
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

    best_checkpoint = project_dir / "train" / "weights" / "best.pt"
    val_map50 = 0.0
    try:
        metrics = model.val()
        val_map50 = float(metrics.box.map50)
    except Exception as exc:  # noqa: BLE001 - best-effort metric capture, training already succeeded
        print(f"Could not compute final val mAP@0.5 for state tracking: {exc}")

    mark_step_complete(STEP_NAME, str(best_checkpoint), "map50", val_map50)
    print(f"Pretrain checkpoint ready at {best_checkpoint}")
    print("Pass it to train_tooth_detection.py via --base-model to fine-tune on FDI classes.")


if __name__ == "__main__":
    main()
