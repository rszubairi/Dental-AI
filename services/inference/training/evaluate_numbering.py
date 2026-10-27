"""Score Stage 1's FDI numbering end-to-end against DENTEX ground truth.

Runs the full ToothDetectionPipeline (detector + overlap dedup + arch ordering +
missing-tooth triage) on DENTEX quadrant_enumeration images from the chosen
split(s) -- the same seeded 80/10/10 split prepare_stage1_dataset.py uses -- and
reports:

- label accuracy: of predicted boxes matching a GT tooth (IoU >= 0.5), the share
  carrying the right FDI number. Off-by-one runs (a 47 called 48 when 48 is
  absent) show up here.
- unmatched predictions (FP) / unmatched GT teeth (FN).
- absent-site precision/recall, separately for non-wisdom sites (the implant-
  candidate "missing" list) and wisdom sites (absent_wisdom_teeth).
- how well numbering_review_quadrants targets the errors: error rate in flagged
  vs unflagged quadrants, and the share of all errors that were flagged.

Use it to compare checkpoints or post-processing changes on val/test before
shipping them. With --split train --write-hard-examples, it also writes the stems
of training images with at least one misnumbered tooth, for
train_tooth_detection.py --oversample-list.

Usage (from services/inference, with its venv):
    python training/evaluate_numbering.py \
        --checkpoint training/runs/stage1/train/weights/best.pt --split val test

    python training/evaluate_numbering.py \
        --checkpoint training/runs/stage1/train/weights/best.pt --split train \
        --write-hard-examples training/hard_examples_stage1.txt
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

INFERENCE_DIR = Path(__file__).resolve().parents[1]
DENTEX_DIR = INFERENCE_DIR.parents[1] / "datasets" / "Dentex"
sys.path.insert(0, str(INFERENCE_DIR))
sys.path.insert(0, str(DENTEX_DIR))

import inference  # noqa: E402
from inference import ToothDetectionPipeline, chart_review_flags  # noqa: E402
from postprocessing.fdi_mapping import FDI_TOOTH_NUMBERS  # noqa: E402
from postprocessing.missing_tooth import WISDOM_TEETH  # noqa: E402
from postprocessing.overlap_dedup import _iou  # noqa: E402
from prepare_stage1_dataset import load_coco, split_image_ids  # noqa: E402

MATCH_IOU = 0.5


def load_ground_truth() -> tuple[Path, list[dict], dict[int, str]]:
    """Return (xrays dir, images with their GT teeth, image id -> split)."""
    src_dir = DENTEX_DIR / "training_data" / "training_data" / "quadrant_enumeration"
    coco = load_coco(src_dir / "train_quadrant_enumeration.json")
    teeth_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        teeth_by_image.setdefault(ann["image_id"], []).append(
            {"fdi_number": f"{ann['category_id_1'] + 1}{ann['category_id_2'] + 1}", "bbox": [x, y, x + w, y + h]}
        )
    images = [{**img, "gt": teeth_by_image.get(img["id"], [])} for img in coco["images"]]
    return src_dir / "xrays", images, split_image_ids([img["id"] for img in coco["images"]])


def match_to_ground_truth(pred: list[dict], gt: list[dict]) -> tuple[list[tuple[dict, dict]], int, int]:
    """Greedy one-to-one IoU matching. Returns (matched pairs, #unmatched pred, #unmatched gt)."""
    pairs = sorted(
        ((_iou(p["bbox"], g["bbox"]), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)), reverse=True
    )
    used_pred, used_gt, matched = set(), set(), []
    for iou, i, j in pairs:
        if iou < MATCH_IOU:
            break
        if i in used_pred or j in used_gt:
            continue
        used_pred.add(i)
        used_gt.add(j)
        matched.append((pred[i], gt[j]))
    return matched, len(pred) - len(used_pred), len(gt) - len(used_gt)


def score_image(detections: list[dict], gt: list[dict], counts: Counter) -> int:
    """Add one image's results to counts; return its number of misnumbered teeth."""
    matched, fp, fn = match_to_ground_truth(detections, gt)
    wrong_pairs = [(p, g) for p, g in matched if p["fdi_number"] != g["fdi_number"]]
    counts["matched"] += len(matched)
    counts["wrong"] += len(wrong_pairs)
    counts["fp"] += fp
    counts["fn"] += fn
    counts["images"] += 1
    counts["images_all_correct"] += not wrong_pairs

    gt_present = {g["fdi_number"] for g in gt}
    pred_present = {d["fdi_number"] for d in detections}
    for fdi in FDI_TOOTH_NUMBERS:
        kind = "wisdom" if fdi in WISDOM_TEETH else "nonwisdom"
        gt_absent, pred_absent = fdi not in gt_present, fdi not in pred_present
        outcome = "tp" if gt_absent and pred_absent else "fp" if pred_absent else "fn" if gt_absent else "tn"
        counts[f"{kind}_absent_{outcome}"] += 1

    flagged = set(chart_review_flags(detections)["numbering_review_quadrants"])
    for quadrant in "1234":
        state = "flagged" if quadrant in flagged else "unflagged"
        q_pairs = [(p, g) for p, g in matched if p["fdi_number"][0] == quadrant]
        q_wrong = sum(p["fdi_number"] != g["fdi_number"] for p, g in q_pairs)
        counts[f"{state}_quadrants"] += 1
        counts[f"{state}_quadrants_with_error"] += q_wrong > 0
        counts[f"{state}_wrong"] += q_wrong
    return len(wrong_pairs)


def _ratio(num: int, den: int) -> str:
    return f"{num / den:.3f}" if den else "n/a"


def print_report(counts: Counter) -> None:
    c = counts
    print(
        f"Label accuracy: {_ratio(c['matched'] - c['wrong'], c['matched'])} "
        f"({c['wrong']} misnumbered of {c['matched']} matched teeth)"
    )
    print(f"Unmatched predictions (FP): {c['fp']}  |  Unmatched GT teeth (FN): {c['fn']}")
    print(f"Images with every matched tooth numbered correctly: {c['images_all_correct']}/{c['images']}")
    for kind, label in (("nonwisdom", "Absent non-wisdom sites"), ("wisdom", "Absent wisdom sites")):
        tp, fp, fn = (c[f"{kind}_absent_{k}"] for k in ("tp", "fp", "fn"))
        print(f"{label}: precision {_ratio(tp, tp + fp)}, recall {_ratio(tp, tp + fn)} (tp {tp}, fp {fp}, fn {fn})")
    for state in ("flagged", "unflagged"):
        print(
            f"{state.capitalize()} quadrants: {c[f'{state}_quadrants']}, with >=1 numbering error: "
            f"{c[f'{state}_quadrants_with_error']} ({_ratio(c[f'{state}_quadrants_with_error'], c[f'{state}_quadrants'])})"
        )
    print(f"Share of numbering errors inside flagged quadrants: {_ratio(c['flagged_wrong'], c['wrong'])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Stage 1 weights (best.pt)")
    parser.add_argument("--split", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument(
        "--write-hard-examples",
        help="Write stems of images with >=1 misnumbered tooth here (one per line)",
    )
    args = parser.parse_args()

    inference.MODEL_CHECKPOINT_PATH = args.checkpoint
    pipeline = ToothDetectionPipeline()
    xrays_dir, images, split_map = load_ground_truth()

    counts: Counter = Counter()
    hard_examples: list[str] = []
    selected = [img for img in images if split_map[img["id"]] in args.split]
    for n, img in enumerate(selected, 1):
        path = xrays_dir / img["file_name"]
        detections, _missing, _uncertain, _image = pipeline.run_with_image_bytes(path.read_bytes(), img["file_name"])
        if score_image(detections, img["gt"], counts):
            hard_examples.append(Path(img["file_name"]).stem)
        if n % 50 == 0:
            print(f"  scored {n}/{len(selected)} images", file=sys.stderr)

    print(f"Checkpoint: {args.checkpoint}  |  splits: {', '.join(args.split)}")
    print_report(counts)

    if args.write_hard_examples:
        Path(args.write_hard_examples).write_text("\n".join(sorted(hard_examples)) + "\n", encoding="utf-8")
        print(f"Wrote {len(hard_examples)} hard-example stems to {args.write_hard_examples}")


if __name__ == "__main__":
    main()
