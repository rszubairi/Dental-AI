"""Dentist corrections to Stage 1's FDI numbering, captured as training data.

Stage 1 often numbers a run of teeth one position off next to an absent tooth
(see postprocessing/missing_tooth.py's find_quadrants_needing_numbering_review).
Post-processing can't recover the right numbers, but a dentist reviewing the
chart can: streamlit_app.py lets them relabel each box and saves the corrected
chart here. Each save appends one record to CORRECTIONS_LOG and stores the
decoded X-ray under CORRECTIONS_DIR/images, so
datasets/Dentex/prepare_stage1_from_corrections.py can turn confirmed charts
into extra Stage 1 training images.

These files hold patient X-rays: CORRECTIONS_DIR is git-ignored.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from postprocessing.fdi_mapping import FDI_CLASS_TO_INDEX

CORRECTIONS_DIR = Path(__file__).resolve().parent / "numbering_corrections"
CORRECTIONS_LOG = CORRECTIONS_DIR / "corrections.jsonl"
# Choice in the correction editor for a box that isn't a tooth (e.g. a duplicate
# box on a tooth that already has one). Such boxes are dropped from the chart.
NOT_A_TOOTH = "not a tooth"


def validate_corrections(corrected_fdis: list[str]) -> list[str]:
    """Return human-readable problems with a corrected chart (empty if valid)."""
    problems = [f"Unknown FDI number: {fdi}" for fdi in corrected_fdis if fdi != NOT_A_TOOTH and fdi not in FDI_CLASS_TO_INDEX]
    counts = Counter(fdi for fdi in corrected_fdis if fdi != NOT_A_TOOTH)
    problems += [f"FDI {fdi} is assigned to {n} boxes" for fdi, n in sorted(counts.items()) if n > 1]
    return problems


def apply_corrections(detections: list[dict], corrected_fdis: list[str]) -> list[dict]:
    """Return detections renumbered per corrected_fdis (same order), dropping
    boxes marked NOT_A_TOOTH. Each kept detection records the model's original
    number as "model_fdi_number" when it differs (kept across repeat corrections)."""
    corrected = []
    for det, fdi in zip(detections, corrected_fdis, strict=True):
        if fdi == NOT_A_TOOTH:
            continue
        new_det = dict(det, fdi_number=fdi)
        new_det.pop("model_fdi_number", None)
        if fdi != model_fdi(det):
            new_det["model_fdi_number"] = model_fdi(det)
        corrected.append(new_det)
    return corrected


def model_fdi(detection: dict) -> str:
    """The number Stage 1 originally gave this detection, before any correction."""
    return detection.get("model_fdi_number", detection["fdi_number"])


def save_corrected_chart(
    image: np.ndarray,
    filename: str,
    checkpoint: str,
    detections: list[dict],
    corrected_fdis: list[str],
    complete: bool,
) -> dict:
    """Store the X-ray and append the corrected chart to CORRECTIONS_LOG.

    Args:
        image: the decoded grayscale X-ray the boxes are in (pipeline output).
        detections: the model's chart as shown to the dentist.
        corrected_fdis: the dentist's number (or NOT_A_TOOTH) for each detection.
        complete: the dentist confirmed every tooth present in the X-ray has a
            box. Only complete charts are used for training -- an unboxed tooth
            would otherwise be learned as background.

    Returns the appended record.
    """
    (CORRECTIONS_DIR / "images").mkdir(parents=True, exist_ok=True)
    ok, png = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode the X-ray as PNG")
    digest = hashlib.sha1(png.tobytes()).hexdigest()
    image_rel = f"images/{digest}.png"
    image_path = CORRECTIONS_DIR / image_rel
    if not image_path.exists():
        image_path.write_bytes(png.tobytes())

    record = {
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_filename": filename,
        "image": image_rel,
        "checkpoint": checkpoint,
        "complete": complete,
        "teeth": [
            {"bbox": [round(v, 1) for v in det["bbox"]], "model_fdi": model_fdi(det), "fdi": fdi}
            for det, fdi in zip(detections, corrected_fdis, strict=True)
        ],
    }
    with open(CORRECTIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def load_latest_charts(log_path: Path | None = None) -> list[dict]:
    """Return the most recent saved chart per X-ray (a later save of the same
    image supersedes earlier ones). log_path defaults to CORRECTIONS_LOG."""
    log_path = log_path or CORRECTIONS_LOG
    latest: dict[str, dict] = {}
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                latest[record["image"]] = record
    return list(latest.values())
