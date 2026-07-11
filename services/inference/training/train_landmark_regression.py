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
from preprocessing.heatmap import extract_peak, generate_gaussian_heatmap, radial_error_mm  # noqa: E402
from preprocessing.training_transforms import apply_clahe  # noqa: E402

STEP_NAME = "stage2b"
MIN_IMAGES = 200

CANVAS_SIZE = 640
GAUSSIAN_SIGMA = 10.0
MM_PER_PIXEL = 0.27
CHANNEL_NAMES = ["bone_crest", "sinus_floor", "nerve_canal"]

BATCH_SIZE = 4
EPOCHS = 80
PATIENCE = 20
TARGET_MRE_MM = 2.0


class LandmarkDataset(Dataset):
    def __init__(self, data_dir: Path, stems: list[str]) -> None:
        self.data_dir = data_dir
        self.stems = stems

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        stem = self.stems[idx]
        image = cv2.imread(str(self.data_dir / "images" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        image = apply_clahe(image)
        image = cv2.resize(image, (CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_LINEAR)

        with open(self.data_dir / "annotations" / f"{stem}.json", encoding="utf-8") as f:
            ann = json.load(f)

        channels = []
        for name in CHANNEL_NAMES:
            points = [(x * CANVAS_SIZE, y * CANVAS_SIZE) for x, y in ann.get(name, [])]
            channels.append(generate_gaussian_heatmap(CANVAS_SIZE, points, sigma=GAUSSIAN_SIGMA))
        target = np.stack(channels, axis=0)

        image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        target_tensor = torch.from_numpy(target).float()
        return image_tensor, target_tensor


def discover_stems(data_dir: Path) -> list[str]:
    image_stems = {p.stem for p in (data_dir / "images").glob("*.png")}
    annotated_stems = {p.stem for p in (data_dir / "annotations").glob("*.json")}
    return sorted(image_stems & annotated_stems)


def evaluate_mre(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    errors = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            preds = model(images).cpu().numpy()
            targets = targets.numpy()
            for i in range(preds.shape[0]):
                for c in range(preds.shape[1]):
                    if targets[i, c].max() == 0:
                        continue  # landmark absent from this sample; nothing to score
                    pred_xy = extract_peak(preds[i, c])
                    gt_xy = extract_peak(targets[i, c])
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
        "--reset-step",
        action="store_true",
        help="Delete existing stage2b output and retrain from scratch, even if already complete",
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
    if len(stems) < MIN_IMAGES:
        print(
            f"BLOCKED: found {len(stems)} annotated images under {data_dir}, "
            f"but Stage 2B requires a minimum of {MIN_IMAGES}. "
            "This stage cannot train until client-provided landmark annotations arrive "
            "(see datasets/landmarks/README.md)."
        )
        return

    split_idx = round(len(stems) * 0.85)
    train_stems, val_stems = stems[:split_idx], stems[split_idx:]

    train_loader = DataLoader(
        LandmarkDataset(data_dir, train_stems), batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        LandmarkDataset(data_dir, val_stems), batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetLandmark().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir.mkdir(parents=True, exist_ok=True)

    best_mre = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
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
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    print(f"Best val MRE: {best_mre:.3f}mm (target < {TARGET_MRE_MM}mm)")
    mark_step_complete(STEP_NAME, str(out_dir / "best.pt"), "mre_mm", best_mre)


if __name__ == "__main__":
    main()
