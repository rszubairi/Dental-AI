"""Streamlit viewer for manually testing/verifying the trained Dental AI models.

Run with the SYSTEM python (not services/inference/.venv) since streamlit lives
there and the venv is deliberately kept isolated from it (see Documentation/runbook.md
step 0 -- mixing streamlit into the torch/ultralytics venv risks protobuf conflicts):

    streamlit run services/inference/streamlit_app.py

Lets you pick an X-ray from datasets/dentex's validation set (or upload any
image/DICOM), run Stage 1 (tooth detection + FDI numbering), Stage 2A (pathology
classification) and/or Stage 2B (landmarks) against it, and see the boxes/points
drawn on the image plus the raw per-tooth results -- for eyeballing model
quality against known validation cases without going through the full
Convex/Next.js app.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent  # services/inference
REPO_ROOT = BASE_DIR.parent.parent
TRAINING_DIR = BASE_DIR / "training"
VALIDATION_DIR = (
    REPO_ROOT
    / "datasets"
    / "dentex"
    / "validation_data"
    / "validation_data"
    / "quadrant_enumeration_disease"
    / "xrays"
)

# inference.py resolves checkpoint paths from these env vars at import time
# (same vars as Documentation/runbook.md step 8) -- set defaults to the
# checkpoints already trained in this repo before importing it.
os.environ.setdefault("TOOTH_DETECTION_CHECKPOINT", str(TRAINING_DIR / "runs/stage1/train/weights/best.pt"))
os.environ.setdefault("PATHOLOGY_CLASSIFIER_CHECKPOINT", str(TRAINING_DIR / "runs/stage2a/best.pt"))
os.environ.setdefault("PATHOLOGY_CLASSIFIER_CLASSES", str(TRAINING_DIR / "runs/stage2a/classes.json"))
os.environ.setdefault("LANDMARK_CHECKPOINT", str(TRAINING_DIR / "runs/stage2b/best.pt"))
# NOTE: runs/stage2b/best.pt was trained with --base-filters 32 (the training
# script's own default -- see train_landmark_regression.py), NOT 16. The
# runbook's step 8 and inference.py's own fallback both say 16, which mismatches
# this checkpoint's actual weight shapes and fails to load; 32 is correct here.
os.environ.setdefault("LANDMARK_BASE_FILTERS", "32")

sys.path.insert(0, str(BASE_DIR))

import inference  # noqa: E402
from inference import (  # noqa: E402
    LandmarkRegressionPipeline,
    PathologyClassificationPipeline,
    ToothDetectionPipeline,
)
from preprocessing.image_loader import load_image  # noqa: E402

PATHOLOGY_COLORS = {
    "healthy": (0, 200, 0),
    "caries": (0, 165, 255),
    "deep_caries": (0, 0, 255),
    "periapical_lesion": (255, 0, 255),
    "impacted": (255, 128, 0),
    None: (255, 255, 0),
}
LANDMARK_COLORS = {
    "bone_crest": (0, 255, 255),
    "sinus_floor": (255, 0, 255),
    "nerve_canal": (0, 128, 255),
}


@st.cache_resource(show_spinner=False)
def _load_pipelines() -> tuple[ToothDetectionPipeline, PathologyClassificationPipeline, LandmarkRegressionPipeline]:
    return ToothDetectionPipeline(), PathologyClassificationPipeline(), LandmarkRegressionPipeline()


def _list_validation_images() -> list[Path]:
    if not VALIDATION_DIR.exists():
        return []
    return sorted(VALIDATION_DIR.glob("*.png"))


def _draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for det in detections:
        x0, y0, x1, y1 = [int(round(v)) for v in det["bbox"]]
        color = PATHOLOGY_COLORS.get(det.get("pathology"), PATHOLOGY_COLORS[None])
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        label = det["fdi_number"]
        if det.get("pathology") and det["pathology"] != "healthy":
            label += f" {det['pathology']} {det['pathology_confidence']:.2f}"
        cv2.putText(canvas, label, (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def _draw_landmarks(canvas: np.ndarray, landmarks: dict[str, list[list[float]]]) -> np.ndarray:
    h, w = canvas.shape[:2]
    for name, points in landmarks.items():
        color = LANDMARK_COLORS.get(name, (255, 255, 255))
        for x_norm, y_norm in points:
            cv2.circle(canvas, (int(x_norm * w), int(y_norm * h)), 3, color, -1)
    return canvas


def main() -> None:
    st.set_page_config(page_title="Dental AI - Model Verification", layout="wide")
    st.title("Dental AI - Model Verification Viewer")
    st.caption("Run trained checkpoints against DENTEX validation images (or your own upload) for visual QA.")

    tooth_pipeline, pathology_pipeline, landmark_pipeline = _load_pipelines()

    with st.sidebar:
        st.header("Settings")
        mode = st.radio(
            "Model",
            ["full_assessment", "tooth_detection", "landmarks"],
            format_func=lambda m: {
                "full_assessment": "Full assessment (Stage 1 + 2A + 2B)",
                "tooth_detection": "Tooth detection only (Stage 1)",
                "landmarks": "Landmarks only (Stage 2B)",
            }[m],
        )
        detection_threshold = st.slider(
            "Tooth detection confidence threshold", 0.0, 1.0, inference.DETECTION_CONFIDENCE_THRESHOLD, 0.05,
            help="Boxes below this confidence aren't returned at all. Raise it to cut false "
            "positives/near-duplicate boxes in crowded or noisy regions.",
        )
        inference.DETECTION_CONFIDENCE_THRESHOLD = detection_threshold
        pathology_threshold = st.slider(
            "Pathology confidence threshold", 0.0, 1.0, inference.PATHOLOGY_CONFIDENCE_THRESHOLD, 0.05,
            help="Below this softmax confidence, a pathology call falls back to 'healthy'.",
        )
        inference.PATHOLOGY_CONFIDENCE_THRESHOLD = pathology_threshold
        landmark_threshold = st.slider(
            "Landmark peak threshold", 0.0, 1.0, inference.LANDMARK_PEAK_THRESHOLD, 0.05,
            help="Heatmap local maxima below this value aren't reported as landmark points.",
        )
        inference.LANDMARK_PEAK_THRESHOLD = landmark_threshold
        only_caries = st.checkbox(
            "Only show caries (hide healthy teeth)",
            help="Filters detections down to caries, hiding healthy and every other pathology class. "
            "Only applies in 'Full assessment' mode, since that's the only mode with pathology labels.",
        )

        st.divider()
        source = st.radio("Image source", ["Validation set", "Upload"])

        raw_bytes: bytes | None = None
        filename = "upload.png"
        if source == "Validation set":
            val_images = _list_validation_images()
            if not val_images:
                st.error(f"No images found in {VALIDATION_DIR}")
            else:
                chosen = st.selectbox("Validation image", val_images, format_func=lambda p: p.name)
                raw_bytes = chosen.read_bytes()
                filename = chosen.name
        else:
            uploaded = st.file_uploader("Upload an X-ray", type=["png", "jpg", "jpeg", "dcm", "dicom"])
            if uploaded is not None:
                raw_bytes = uploaded.getvalue()
                filename = uploaded.name

        run_clicked = st.button("Run inference", type="primary", disabled=raw_bytes is None)

    if raw_bytes is None:
        st.info("Pick a validation image or upload one, then click **Run inference**.")
        return

    if not run_clicked:
        image = load_image(raw_bytes, filename=filename)
        st.image(image, caption=f"{filename} (not yet analyzed)", use_container_width=True)
        return

    try:
        with st.spinner("Running inference..."):
            if mode == "tooth_detection":
                detections, missing_teeth, image = tooth_pipeline.run_with_image_bytes(raw_bytes, filename)
                landmarks = None
            elif mode == "landmarks":
                image = load_image(raw_bytes, filename=filename)
                detections, missing_teeth = [], []
                landmarks = landmark_pipeline.run_with_image_bytes(raw_bytes, filename)
            else:
                raw_detections, missing_teeth, image = tooth_pipeline.run_with_image_bytes(raw_bytes, filename)
                detections = pathology_pipeline.classify_crops(image, raw_detections)
                landmarks = landmark_pipeline.run_with_image_bytes(raw_bytes, filename)
    except RuntimeError as exc:
        st.error(str(exc))
        return

    if only_caries:
        detections = [d for d in detections if d.get("pathology") == "caries"]

    canvas = _draw_detections(image, detections)
    if landmarks:
        canvas = _draw_landmarks(canvas, landmarks)
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    st.image(canvas_rgb, caption=filename, use_container_width=True)

    st.subheader("Summary")
    metric_cols = st.columns(2)
    metric_cols[0].metric("Caries detected" if only_caries else "Teeth detected", len(detections))
    metric_cols[1].metric("Missing teeth (FDI)", len(missing_teeth))
    if missing_teeth:
        st.write(", ".join(missing_teeth))

    if detections:
        st.subheader("Per-tooth detections")
        rows = [
            {
                "FDI": d["fdi_number"],
                "Detection conf.": round(d["confidence"], 3),
                "Pathology": d.get("pathology", "-"),
                "Pathology conf.": round(d["pathology_confidence"], 3) if "pathology_confidence" in d else None,
            }
            for d in detections
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    if landmarks:
        st.subheader("Landmark points")
        st.write({name: len(points) for name, points in landmarks.items()})

    with st.expander("Raw JSON"):
        st.json({"detections": detections, "missing_teeth": missing_teeth, "landmarks": landmarks})


if __name__ == "__main__":
    main()
