"""Guardrail against double-counted teeth.

Ultralytics' built-in NMS (see inference.py's model.predict call) only suppresses
duplicate boxes *within the same predicted class*. On a panoramic X-ray the model
assigns each box one of 32 FDI-number classes, so when one physical tooth produces
several overlapping boxes that get classified as *different* neighboring FDI numbers
(e.g. the same crowded region drawing boxes for both "31" and "41"), nothing in the
detector suppresses them against each other. That's how a 32-tooth adult arch turns
into 40+ "detections": the same tooth counted multiple times under different FDI
labels, which also corrupts missing-tooth inference (a mislabeled box makes a truly
missing site look "present", while the correct site it should have filled looks
"missing").

This module re-runs suppression across all classes together, on the assumption that
in panoramic X-ray space two boxes that overlap heavily are almost always the same
physical tooth, never two distinct teeth (teeth are laid out along the arch with
little mutual overlap). It keeps the highest-confidence box per overlap cluster,
then enforces the anatomical invariant that an FDI site can hold at most one tooth.
"""

from __future__ import annotations

# Two boxes overlapping this much (intersection / union) are treated as the same
# physical tooth. Chosen below Ultralytics' NMS default (0.7) since that default is
# exactly what let same-class duplicates like three ~100%-overlapping "32" boxes
# through in observed validation cases -- see run_validation.py output. Lower this if
# genuinely adjacent teeth (which can touch/overlap slightly at the gumline on a 2D
# projection) start getting merged; raise it if duplicate boxes still slip through.
DEFAULT_IOU_THRESHOLD = 0.4


def _iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter_area <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def deduplicate_overlapping_detections(
    detections: list[dict], iou_threshold: float = DEFAULT_IOU_THRESHOLD
) -> list[dict]:
    """Collapse detections that overlap heavily, regardless of their predicted FDI
    class, keeping only the highest-confidence box per physical tooth.

    Runs in two passes:
    1. Greedy cross-class NMS by IoU -- suppresses overlapping lower-confidence boxes
       no matter what FDI number each was classified as.
    2. One-box-per-FDI-number safety net -- an anatomical invariant (a given FDI site
       can't hold two teeth) in case non-overlapping duplicates of the same class
       slip past step 1 (e.g. two low-IoU boxes near the same tooth from a noisy
       prediction).
    """
    by_confidence = sorted(detections, key=lambda d: d["confidence"], reverse=True)

    kept: list[dict] = []
    for det in by_confidence:
        if any(_iou(det["bbox"], k["bbox"]) > iou_threshold for k in kept):
            continue
        kept.append(det)

    best_per_fdi: dict[str, dict] = {}
    for det in kept:
        fdi = det["fdi_number"]
        if fdi not in best_per_fdi or det["confidence"] > best_per_fdi[fdi]["confidence"]:
            best_per_fdi[fdi] = det

    return list(best_per_fdi.values())
