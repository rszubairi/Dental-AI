"""Train the sinus floor / nerve canal / bone crest landmark regression model
(Stage 2B, U-Net heatmap regression).

BLOCKED: requires client-provided OPGs with landmark annotations (minimum 200 images).
See datasets/landmarks/README.md for the expected format. This script will refuse to
run against fewer than MIN_IMAGES annotated images.

Expected dataset layout:
    <data_dir>/images/<stem>.png
    <data_dir>/annotations/<stem>.json
        {
          "bone_crest":   [[x, y], ...],   # normalized 0-1 image coordinates
          "sinus_floor":  [[x, y], ...],
          "nerve_canal":  [[x, y], ...]
        }
Each list may be empty if that landmark isn't present/visible in a given image.

Usage:
    python train_landmark_regression.py --data ../../../datasets/landmarks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pipeline_state import is_step_complete, mark_step_complete, reset_step
from unet_landmark_model import UNetLandmark

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.heatmap import extract_local_peak, generate_gaussian_heatmap, radial_error_mm  # noqa: E402
from preprocessing.training_transforms import apply_clahe  # noqa: E402

STEP_NAME = "stage2b"
MIN_IMAGES = 200

CANVAS_SIZE = 640
GAUSSIAN_SIGMA = 10.0
MM_PER_PIXEL = 0.27
CHANNEL_NAMES = ["bone_crest", "sinus_floor", "nerve_canal"]
# Radius (px) searched around each ground-truth point for a local response peak
# when scoring MRE. Ground-truth points along these curves are spaced ~10px
# apart on this canvas (skeleton sampling at generation time), so the window
# must stay well under that spacing -- otherwise the search drifts onto a
# neighbouring point on the same curve instead of scoring the queried one.
LOCAL_PEAK_WINDOW = 6.0
# Plain MSE on these heatmaps is dominated by background (landmarks cover a
# tiny fraction of the 640x640 canvas), so the loss-minimizing solution is a
# low, diffuse blob rather than a sharp peak. Weighting pixels by how "hot"
# their target value is keeps the loss focused near actual landmark locations.
HEATMAP_LOSS_POS_WEIGHT = 50.0

BATCH_SIZE = 4
EPOCHS = 80
PATIENCE = 20
TARGET_MRE_MM = 2.0


class LandmarkDataset(Dataset):
    def __init__(self, data_dir: Path, stems: list[str], augment: bool = False) -> None:
        self.data_dir = data_dir
        self.stems = stems
        self.augment = augment

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[tuple[float, float]]]]:
        stem = self.stems[idx]
        image = cv2.imread(str(self.data_dir / "images" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        image = apply_clahe(image)
        image = cv2.resize(image, (CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_LINEAR)

        with open(self.data_dir / "annotations" / f"{stem}.json", encoding="utf-8") as f:
            ann = json.load(f)

        norm_points = {name: [tuple(p) for p in ann.get(name, [])] for name in CHANNEL_NAMES}

        if self.augment:
            if np.random.rand() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1])
                norm_points = {
                    name: [(1.0 - x, y) for x, y in pts] for name, pts in norm_points.items()
                }
            gain = np.random.uniform(0.85, 1.15)
            bias = np.random.uniform(-15, 15)
            image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

        channels = []
        points_by_name = {}
        for name in CHANNEL_NAMES:
            points = [(x * CANVAS_SIZE, y * CANVAS_SIZE) for x, y in norm_points[name]]
            points_by_name[name] = points
            channels.append(generate_gaussian_heatmap(CANVAS_SIZE, points, sigma=GAUSSIAN_SIGMA))
        target = np.stack(channels, axis=0)

        image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        target_tensor = torch.from_numpy(target).float()
        return image_tensor, target_tensor, points_by_name


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, list[tuple[float, float]]]]],
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, list[tuple[float, float]]]]]:
    images = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    points = [b[2] for b in batch]
    return images, targets, points


def discover_stems(data_dir: Path) -> list[str]:
    image_stems = {p.stem for p in (data_dir / "images").glob("*.png")}
    annotated_stems = {p.stem for p in (data_dir / "annotations").glob("*.json")}
    return sorted(image_stems & annotated_stems)


def weighted_heatmap_mse(pred: torch.Tensor, target: torch.Tensor, pos_weight: float = HEATMAP_LOSS_POS_WEIGHT) -> torch.Tensor:
    """MSE with per-pixel weight scaled by target "hotness", so errors near
    true landmark locations dominate the loss instead of the vast background."""
    weight = 1.0 + target * (pos_weight - 1.0)
    return (weight * (pred - target) ** 2).mean()


def evaluate_mre(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    errors = []
    with torch.no_grad():
        for images, _targets, points_batch in loader:
            images = images.to(device)
            preds = model(images).cpu().numpy()
            for i in range(preds.shape[0]):
                for c, name in enumerate(CHANNEL_NAMES):
                    gt_points = points_batch[i].get(name, [])
                    for gt_xy in gt_points:
                        pred_xy = extract_local_peak(preds[i, c], gt_xy, LOCAL_PEAK_WINDOW)
                        errors.append(radial_error_mm(pred_xy, gt_xy, MM_PER_PIXEL))
    return float(np.mean(errors)) if errors else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to landmark dataset root")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="runs/stage2b")
    parser.add_argument(
        "--base-filters",
        type=int,
        default=32,
        help="U-Net base filter count. Lower reduces model capacity to fight overfitting on small datasets.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable train-time augmentation (horizontal flip + brightness/contrast jitter).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <out>/checkpoint.pt (model, optimizer, epoch, best MRE) if present.",
    )
    parser.add_argument(
        "--reset-step",
        action="store_true",
        help="Delete existing stage2b output and retrain from scratch, even if already complete",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=MIN_IMAGES,
        help=(
            f"Override the {MIN_IMAGES}-image floor (testing only -- with too few images "
            "the val split may be tiny/empty and MRE numbers are not meaningful)."
        ),
    )
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)

    if args.reset_step:
        reset_step(STEP_NAME, out_dir)
    elif is_step_complete(STEP_NAME):
        print(
            f"{STEP_NAME} already complete per .dental_ai_state.json. "
            "Pass --reset-step to retrain from scratch."
        )
        return

    stems = discover_stems(data_dir)
    if len(stems) < args.min_images:
        print(
            f"BLOCKED: found {len(stems)} annotated images under {data_dir}, "
            f"but Stage 2B requires a minimum of {args.min_images}. "
            "This stage cannot train until client-provided landmark annotations arrive "
            "(see datasets/landmarks/README.md)."
        )
        return

    split_idx = round(len(stems) * 0.85)
    train_stems, val_stems = stems[:split_idx], stems[split_idx:]

    train_loader = DataLoader(
        LandmarkDataset(data_dir, train_stems, augment=not args.no_augment),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        LandmarkDataset(data_dir, val_stems),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetLandmark(base_filters=args.base_filters).to(device)
    criterion = weighted_heatmap_mse
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir.mkdir(parents=True, exist_ok=True)

    best_mre = float("inf")
    epochs_without_improvement = 0
    start_epoch = 1

    checkpoint_path = out_dir / "checkpoint.pt"
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_mre = checkpoint["best_mre"]
        epochs_without_improvement = checkpoint["epochs_without_improvement"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from {checkpoint_path} at epoch {start_epoch} (best_mre={best_mre:.3f})")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets, _points in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        val_mre = evaluate_mre(model, val_loader, device)
        avg_loss = running_loss / len(train_stems)
        print(f"epoch {epoch}/{args.epochs} loss={avg_loss:.6f} val_mre_mm={val_mre:.3f}")

        torch.save(model.state_dict(), out_dir / "last.pt")

        if val_mre < best_mre:
            best_mre = val_mre
            epochs_without_improvement = 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_mre": best_mre,
                "epochs_without_improvement": epochs_without_improvement,
            },
            checkpoint_path,
        )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    print(f"Best val MRE: {best_mre:.3f}mm (target < {TARGET_MRE_MM}mm)")
    mark_step_complete(STEP_NAME, str(out_dir / "best.pt"), "mre_mm", best_mre)


if __name__ == "__main__":
    main()
