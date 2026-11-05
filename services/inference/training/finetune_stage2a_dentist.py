"""Fine-tune a Stage 2A checkpoint on dentist-reviewed X-rays (dentist_caries_labels.json).

Her labels are per tooth, caries or not: she doesn't separate caries from deep
caries, and says nothing about other pathology. So her teeth are trained only on
P(caries) + P(deep_caries) -- up for a caries tooth, down for a caries-free one --
leaving how the rest splits across healthy/impacted/periapical to the DENTEX
crops, which are mixed into every step so the model doesn't forget them.

There are only a dozen reviewed X-rays, and they're also the only dentist-checked
test set, so the fine-tune is first scored by cross-validation: it trains on all
but a fold of X-rays and scores the fold it never saw, against the unchanged
checkpoint on the same teeth. Then it trains once more on all of them for the
checkpoint to use.

Teeth come from the Stage 1 detector, cut as the checkpoint's crop.json says
(inference.py). A tooth she names that Stage 1 doesn't find can't be used.

Usage (from services/inference/training):
    python finetune_stage2a_dentist.py --base runs/stage2a_crown256_focal_v2 \
        --data ../../../datasets/Dentex/stage2a_crown256_v2 --out runs/stage2a_crown256_focal_v3_dentist
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inference  # noqa: E402
from preprocessing.image_loader import load_image  # noqa: E402
from preprocessing.training_transforms import apply_clahe, to_rgb  # noqa: E402
from train_pathology_classifier import (  # noqa: E402
    CLASSES,
    FocalLoss,
    build_eval_transforms,
    build_model,
    build_train_transforms,
    compute_class_weights,
    evaluate,
)

LABELS_PATH = Path(__file__).resolve().parent / "dentist_caries_labels.json"
VALIDATION_XRAYS = (
    Path(__file__).resolve().parents[3]
    / "datasets/Dentex/validation_data/validation_data/quadrant_enumeration_disease/xrays"
)
CUT_OFFS = (0.6, 0.7, 0.8)
SEED = 42


def dentist_teeth(base: Path) -> list[dict]:
    """One entry per tooth Stage 1 finds on each reviewed X-ray: its crop (as the
    checkpoint expects) and whether she reported caries on it."""
    inference.PATHOLOGY_CLASSES_PATH = str(base / "classes.json")
    crop_recipe = inference.load_pathology_crop()
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))["caries_teeth"]
    detector = inference.ToothDetectionPipeline()
    teeth = []
    for filename, caries in labels.items():
        raw = (VALIDATION_XRAYS / filename).read_bytes()
        detections, _, _, _ = detector.run_with_image_bytes(raw, filename=filename)
        enhanced = apply_clahe(load_image(raw, filename=filename))
        found = set()
        for det in detections:
            x0, y0, x1, y1 = det["bbox"]
            crop = to_rgb(inference.extract_pathology_crop(enhanced, (x0, y0, x1 - x0, y1 - y0), crop_recipe))
            teeth.append({"xray": filename, "tooth": det["fdi_number"], "caries": det["fdi_number"] in caries, "crop": crop})
            found.add(det["fdi_number"])
        for tooth in sorted(set(caries) - found):
            print(f"  {filename}: Stage 1 didn't find tooth {tooth}; can't train or score on it")
    return teeth


def caries_mass(logits: torch.Tensor, caries_idx: list[int]) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, caries_idx].sum(dim=1).clamp(1e-6, 1 - 1e-6)


def finetune(base: Path, dentex: datasets.ImageFolder, teeth: list[dict], args, device) -> torch.nn.Module:
    model = build_model(device)
    model.load_state_dict(torch.load(base / "best.pt", map_location=device))
    caries_idx = [dentex.classes.index("caries"), dentex.classes.index("deep_caries")]
    dentex_loss = FocalLoss(weight=compute_class_weights(dentex, device))
    dentex_batches = DataLoader(dentex, batch_size=args.batch_size, shuffle=True, num_workers=0)
    train_tf = build_train_transforms(args.input_size)
    positives = [t for t in teeth if t["caries"]]
    negatives = [t for t in teeth if not t["caries"]]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(SEED)
    from PIL import Image

    model.train()
    for epoch in range(args.epochs):
        for step, (images, labels) in enumerate(dentex_batches):
            if step >= args.steps_per_epoch:
                break
            # Half caries, half caries-free: her 17 caries teeth would otherwise be
            # swamped by the ~300 caries-free ones.
            half = args.dentist_batch // 2
            batch = [rng.choice(positives) for _ in range(half)] + [rng.choice(negatives) for _ in range(half)]
            dentist_images = torch.stack([train_tf(Image.fromarray(t["crop"])) for t in batch]).to(device)
            is_caries = torch.tensor([t["caries"] for t in batch], dtype=torch.float32, device=device)

            mass = caries_mass(model(dentist_images), caries_idx)
            dentist_loss = -(is_caries * mass.log() + (1 - is_caries) * (1 - mass).log()).mean()
            loss = dentex_loss(model(images.to(device)), labels.to(device)) + args.dentist_weight * dentist_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"  epoch {epoch + 1}/{args.epochs} loss={loss.item():.4f}")
    model.eval()
    return model


def predict(model: torch.nn.Module, teeth: list[dict], classes: list[str], input_size: int, device) -> list[dict]:
    from PIL import Image

    tf = build_eval_transforms(input_size)
    out = []
    with torch.no_grad():
        for t in teeth:
            probs = torch.softmax(model(tf(Image.fromarray(t["crop"])).unsqueeze(0).to(device)), dim=1)[0].cpu().numpy()
            out.append(dict(zip(classes, probs.tolist())))
    return out


def score(teeth: list[dict], probs: list[dict], cut_off: float) -> dict:
    """inference.py's rule: a pathology is reported when it's the top class and clears the cut-off."""
    def flagged(p):
        top = max(p, key=p.get)
        return top in ("caries", "deep_caries") and p[top] >= cut_off

    found = sum(t["caries"] and flagged(p) for t, p in zip(teeth, probs))
    false_flags = sum((not t["caries"]) and flagged(p) for t, p in zip(teeth, probs))
    return {"found": found, "caries_teeth": sum(t["caries"] for t in teeth), "false_flags": false_flags}


def folds(xrays: list[str], labels: dict, n: int) -> list[list[str]]:
    """Split X-rays into n folds, spreading caries and caries-free X-rays evenly."""
    with_caries = [x for x in xrays if labels[x]]
    without = [x for x in xrays if not labels[x]]
    return [with_caries[i::n] + without[i::n] for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Checkpoint dir to fine-tune (best.pt, classes.json, crop.json)")
    parser.add_argument("--data", required=True, help="Its DENTEX crop dataset root (train/, val/)")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--tooth-checkpoint",
        default="runs/stage1_rect_v1/train/weights/best.pt",
        help="Stage 1 detector that finds the teeth (Streamlit's default)",
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dentist-batch", type=int, default=16)
    parser.add_argument("--dentist-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    inference.MODEL_CHECKPOINT_PATH = args.tooth_checkpoint
    base, out_dir = Path(args.base), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.input_size = json.loads((base / "crop.json").read_text())["size"]
    classes = json.loads((base / "classes.json").read_text())

    dentex = datasets.ImageFolder(str(Path(args.data) / "train"), transform=build_train_transforms(args.input_size))
    dentex_val = datasets.ImageFolder(str(Path(args.data) / "val"), transform=build_eval_transforms(args.input_size))
    assert dentex.classes == classes, f"{args.data} classes {dentex.classes} != checkpoint's {classes}"
    assert set(classes) == set(CLASSES)

    print("Cutting the reviewed teeth...")
    teeth = dentist_teeth(base)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))["caries_teeth"]
    print(f"  {len(teeth)} teeth, {sum(t['caries'] for t in teeth)} with caries")

    base_model = build_model(device)
    base_model.load_state_dict(torch.load(base / "best.pt", map_location=device))
    base_model.eval()
    base_probs = predict(base_model, teeth, classes, args.input_size, device)

    held_out_probs: list[dict | None] = [None] * len(teeth)
    for i, fold in enumerate(folds(list(labels), labels, args.folds)):
        print(f"Fold {i + 1}/{args.folds}: holding out {fold}")
        train_teeth = [t for t in teeth if t["xray"] not in fold]
        model = finetune(base, dentex, train_teeth, args, device)
        idx = [j for j, t in enumerate(teeth) if t["xray"] in fold]
        for j, p in zip(idx, predict(model, [teeth[j] for j in idx], classes, args.input_size, device)):
            held_out_probs[j] = p

    report = {"cross_validation": {}}
    for cut_off in CUT_OFFS:
        report["cross_validation"][str(cut_off)] = {
            "base": score(teeth, base_probs, cut_off),
            "fine_tuned_on_other_xrays": score(teeth, held_out_probs, cut_off),
        }
        print(f"cut-off {cut_off}: {report['cross_validation'][str(cut_off)]}")

    print("Training on all reviewed X-rays...")
    final = finetune(base, dentex, teeth, args, device)
    torch.save(final.state_dict(), out_dir / "best.pt")
    for name in ("classes.json", "crop.json"):
        shutil.copy(base / name, out_dir / name)
    base_auc, _, _ = evaluate(base_model, DataLoader(dentex_val, batch_size=args.batch_size), device)
    final_auc, _, _ = evaluate(final, DataLoader(dentex_val, batch_size=args.batch_size), device)
    report["dentex_val_macro_auc"] = {"base": round(base_auc, 4), "fine_tuned": round(final_auc, 4)}
    report["args"] = {k: v for k, v in vars(args).items()}
    print(f"DENTEX val macro AUC: base {base_auc:.4f}, fine-tuned {final_auc:.4f}")
    (out_dir / "dentist_finetune_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
