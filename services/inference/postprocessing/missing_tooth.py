"""Infer missing tooth sites from Stage 1 detections.

DENTEX (and Stage 1 in general) only annotates/detects teeth that are physically
present. Missing teeth are inferred by diffing the expected adult FDI set against
whatever Stage 1 actually detected. Wisdom teeth (18/28/38/48) are excluded from
the expected set since their absence is common and not itself abnormal.

A plain diff can't tell a missing tooth from a Stage 1 mistake: on DENTEX's
held-out split ~40% of diffed "missing" sites hold a real tooth that was either
not detected or numbered one position off (e.g. a 16 called 17). The detector's
own sub-threshold boxes are the best available evidence for both, so
assess_missing_teeth() uses them to split the diff into "likely missing" and
"uncertain" (flag for dentist review) -- nothing is dropped, only triaged.
"""

from __future__ import annotations

from .fdi_mapping import FDI_TOOTH_NUMBERS
from .overlap_dedup import _iou

WISDOM_TEETH = {"18", "28", "38", "48"}
EXPECTED_FDI_SET: set[str] = set(FDI_TOOTH_NUMBERS) - WISDOM_TEETH

# Thresholds below were tuned on Stage 1's val split and checked on its test split
# (likely-missing precision 0.56 -> 0.71 on test). Re-tune if Stage 1 is retrained.
#
# Possible numbering error: a sub-threshold box labelled with the missing FDI number
# sits on a tooth already detected under another number. Ultralytics' per-class NMS
# lets a second-choice label survive as its own weak box, so this is the detector
# hedging between two numbers for one tooth. ~70% of such sites are present teeth.
NUMBERING_ALT_MIN_CONFIDENCE = 0.1
NUMBERING_ALT_MIN_IOU = 0.5
# Possible detector miss: a weak box labelled with the missing FDI number sits where
# no reported tooth is. Roughly half of such sites are present teeth.
DETECTOR_MISS_MIN_CONFIDENCE = 0.03
DETECTOR_MISS_MAX_IOU = 0.3

REASON_NUMBERING = "possible_numbering_error"
REASON_DETECTOR_MISS = "possible_detector_miss"


def _sort_fdi(fdi_numbers) -> list[str]:
    return sorted(fdi_numbers, key=lambda fdi: (fdi[0], fdi[1]))


def find_missing_teeth(detected_fdi_numbers: list[str]) -> list[str]:
    """Return the sorted list of expected FDI numbers absent from detections.

    Each entry is a candidate missing-tooth site requiring implant assessment.
    """
    detected = set(detected_fdi_numbers)
    return _sort_fdi(EXPECTED_FDI_SET - detected)


def assess_missing_teeth(
    detections: list[dict], candidates: list[dict]
) -> tuple[list[str], list[dict]]:
    """Split find_missing_teeth()'s sites into likely-missing and uncertain.

    Args:
        detections: reported (deduplicated, above-threshold) detections.
        candidates: the detector's sub-threshold boxes for the same image, each
            {"fdi_number", "bbox", "confidence"}. Used only as evidence here.

    Returns:
        (likely_missing, uncertain): likely_missing is a sorted FDI list;
        uncertain is a list of {"fdi_number", "reason", "bbox", "confidence"}
        sorted by FDI number, where bbox/confidence are the sub-threshold box that
        triggered it (so a viewer can show the dentist where to look).
    """
    likely_missing: list[str] = []
    uncertain: list[dict] = []
    for fdi in find_missing_teeth([d["fdi_number"] for d in detections]):
        evidence = _uncertainty_evidence(fdi, detections, candidates)
        if evidence is None:
            likely_missing.append(fdi)
        else:
            reason, cand = evidence
            uncertain.append(
                {"fdi_number": fdi, "reason": reason, "bbox": cand["bbox"], "confidence": cand["confidence"]}
            )
    return likely_missing, uncertain


def _uncertainty_evidence(
    fdi: str, detections: list[dict], candidates: list[dict]
) -> tuple[str, dict] | None:
    for cand in candidates:
        if cand["fdi_number"] != fdi:
            continue
        overlap = max((_iou(cand["bbox"], d["bbox"]) for d in detections), default=0.0)
        if overlap >= NUMBERING_ALT_MIN_IOU and cand["confidence"] >= NUMBERING_ALT_MIN_CONFIDENCE:
            return REASON_NUMBERING, cand
        if overlap < DETECTOR_MISS_MAX_IOU and cand["confidence"] >= DETECTOR_MISS_MIN_CONFIDENCE:
            return REASON_DETECTOR_MISS, cand
    return None
