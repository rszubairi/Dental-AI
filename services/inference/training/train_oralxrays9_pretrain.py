"""Pretrain a YOLOv8s backbone on the oralxrays9 dataset (9-class clinical
condition detection -- Apical Periodontitis, Decay, Wisdom Tooth, Missing Tooth,
Dental Filling, Root Canal Filling, Implant, Porcelain Crown, Ceramic Bridge --
12,449 panoramic X-rays) before fine-tuning on Stage 1's 32-class FDI task.

Purpose: Stage 1's FDI fine-tune only has 507 training images (datasets/Dentex/yolo),
which we traced this session to a real generalization gap (strong metrics on its own
internal val split, much weaker recall on DENTEX's true held-out set). This dataset
is ~17x larger and the same imaging modality (panoramic X-ray, not bitewing/intraoral),
so pretraining the backbone/neck on it first should give Stage 1's fine-tune a much
better starting point -- same role as train_quadrant_pretrain.py, just far larger.

This is a separate label space from Stage 1's FDI classes -- only the backbone/neck
weights are meant to be reused: pass this checkpoint as --base-model to
train_tooth_detection.py, and Ultralytics will transfer matching layers while
reinitializing the detection head to match Stage 1's nc=32, the same way loading
yolov8s.pt's 80-class COCO checkpoint (or the quadrant-pretrain checkpoint) works today.

fliplr is kept at 0 here for consistency with the rest of this pipeline and because
Stage 1's downstream task needs the backbone to preserve left/right positional
asymmetry (FDI quadrants are side-sensitive) -- even though none of oralxrays9's own
9 classes encode side, training this backbone toward flip-invariance could still
work against that downstream requirement. mosaic is also kept off: compositing 4
unrelated X-rays into one chimeric image doesn't reflect real panoramic anatomy.

Usage:
    python train_oralxrays9_pretrain.py --data ../../../datasets/oralxrays9/yolo/dataset.yaml
    python train_tooth_detection.py --data ... --base-model runs/pretrain_oralxrays9/train/weights/best.pt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

from pipeline_state import is_step_complete, mark_step_complete, reset_step

STEP_NAME = "pretrain_oralxrays9"

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
    # Fewer epochs than train_quadrant_pretrain.py's default (50): this dataset has
    # ~12.6x more training images (8,736 vs 693), so each epoch covers far more data.
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--base-model", default="yolov8s.pt")
    parser.add_argument("--batch", type=int, default=None, help="Overrides VRAM auto-detection")
    parser.add_argument("--project", default="runs/pretrain_oralxrays9")
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
        # See module docstring: fliplr/mosaic kept off to match the rest of this
        # pipeline's conservative OPG augmentation policy, not because oralxrays9's
        # own classes are side-sensitive.
        fliplr=0.0,
        flipud=0.0,
        degrees=5.0,
        translate=0.05,
        scale=0.10,
        hsv_v=0.30,
        hsv_h=0.0,
        hsv_s=0.0,
        mosaic=0.0,
    )

    canonical_dir = project_dir / "train"
    best_checkpoint = canonical_dir / "weights" / "best.pt"

    # See train_tooth_detection.py's identical block for why this is needed:
    # newer ultralytics versions insert a RUNS_DIR/<task>/ prefix in front of a
    # relative `project=` path, which silently broke this path assumption and
    # reset_step's archive-before-delete safety net.
    actual_save_dir = model.trainer.save_dir
    if actual_save_dir.resolve() != canonical_dir.resolve():
        print(
            f"Note: ultralytics saved outputs to {actual_save_dir} instead of the expected "
            f"{canonical_dir}. Copying weights to the canonical location."
        )
        canonical_weights = canonical_dir / "weights"
        canonical_weights.mkdir(parents=True, exist_ok=True)
        for fname in ("best.pt", "last.pt"):
            src = actual_save_dir / "weights" / fname
            if src.exists():
                shutil.copy2(src, canonical_weights / fname)

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
