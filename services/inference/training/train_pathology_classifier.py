"""Train the per-tooth pathology classifier (Stage 2A, EfficientNet-B3).

Requires a class-per-folder crop dataset such as the one produced by
datasets/dentex/prepare_stage2a_dataset.py:

    <data_dir>/train/{healthy,caries,deep_caries,periapical_lesion,impacted}/*.png
    <data_dir>/val/{healthy,caries,deep_caries,periapical_lesion,impacted}/*.png

Usage:
    python train_pathology_classifier.py --data ../../../datasets/dentex/stage2a
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from pipeline_state import is_step_complete, mark_step_complete, reset_step

STEP_NAME = "stage2a"

CLASSES = ["healthy", "caries", "deep_caries", "periapical_lesion", "impacted"]
INPUT_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 15
LEARNING_RATE = 1e-4
FINAL_LR = 1e-6


def build_train_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_eval_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_model(device: torch.device) -> nn.Module:
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASSES))
    return model.to(device)


def compute_class_weights(dataset: datasets.ImageFolder, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(len(CLASSES))
    for _, label in dataset.samples:
        counts[label] += 1
    counts = counts.clamp(min=1)
    weights = counts.sum() / (len(CLASSES) * counts)
    return weights.to(device)


def _binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    """Mann-Whitney U based AUC; avoids adding a scikit-learn dependency."""
    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]
    if pos_scores.size == 0 or neg_scores.size == 0:
        return None
    diff = pos_scores[:, None] - neg_scores[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (pos_scores.size * neg_scores.size))


def _macro_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """One-vs-rest AUC per class, averaged. Classes absent from this split are skipped."""
    aucs = []
    for class_idx in range(len(CLASSES)):
        auc = _binary_auc((labels == class_idx).astype(int), probs[:, class_idx])
        if auc is not None:
            aucs.append(auc)
    return float(np.mean(aucs)) if aucs else 0.0


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return _macro_auc(labels, probs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to stage2a crop dataset root")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--out", default="runs/stage2a")
    parser.add_argument(
        "--reset-step",
        action="store_true",
        help="Delete existing stage2a output and retrain from scratch, even if already complete",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)

    if args.reset_step:
        reset_step(STEP_NAME, out_dir)
    elif is_step_complete(STEP_NAME):
        print(
            f"{STEP_NAME} already complete per .dental_ai_state.json. "
            "Pass --reset-step to retrain from scratch."
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = datasets.ImageFolder(str(Path(args.data) / "train"), transform=build_train_transforms())
    val_ds = datasets.ImageFolder(str(Path(args.data) / "val"), transform=build_eval_transforms())
    assert set(train_ds.classes) == set(CLASSES), f"Expected class folders {CLASSES}, found {train_ds.classes}"
    assert train_ds.classes == val_ds.classes, (
        f"train/val class folder sets differ: {train_ds.classes} vs {val_ds.classes}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # ImageFolder assigns indices alphabetically, which may not match CLASSES' declared
    # order above -- persist the actual mapping so inference can decode predictions correctly.
    with open(out_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(train_ds.classes, f)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(device)
    class_weights = compute_class_weights(train_ds, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=FINAL_LR)

    best_auc = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        scheduler.step()

        val_auc = evaluate(model, val_loader, device)
        avg_loss = running_loss / len(train_ds)
        print(f"epoch {epoch}/{args.epochs} loss={avg_loss:.4f} val_macro_auc={val_auc:.4f}")

        torch.save(model.state_dict(), out_dir / "last.pt")

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    print(f"Best val macro AUC-ROC: {best_auc:.4f}")
    mark_step_complete(STEP_NAME, str(out_dir / "best.pt"), "macro_auc", best_auc)


if __name__ == "__main__":
    main()
