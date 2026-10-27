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

The X-ray view and the per-tooth table are linked through st.session_state's
"selected_idx": clicking a tooth's box (or its table row) sets it, and both the
plot and the table re-render highlighting whichever tooth is selected, alongside
a zoomed-in crop for close inspection. Hovering a box always shows its FDI
number/pathology, independent of selection.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    chart_review_flags,
)
from numbering_corrections import (  # noqa: E402
    NOT_A_TOOTH,
    apply_corrections,
    model_fdi,
    save_corrected_chart,
    validate_corrections,
)
from postprocessing.fdi_mapping import FDI_TOOTH_NUMBERS  # noqa: E402
from postprocessing.missing_tooth import find_absent_wisdom_teeth, find_missing_teeth  # noqa: E402
from postprocessing.overlap_dedup import _iou  # noqa: E402
from preprocessing.image_loader import load_image  # noqa: E402

# BGR tuples (kept BGR so landmark drawing via cv2 below doesn't need its own
# separate color table) -- converted to "rgb(r,g,b)" strings for Plotly via
# _bgr_to_rgb_str where needed.
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
SELECTED_COLOR_RGB = "rgb(255,255,255)"


def _bgr_to_rgb_str(bgr: tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"rgb({r},{g},{b})"


@st.cache_resource(show_spinner=False)
def _load_pipelines() -> tuple[ToothDetectionPipeline, PathologyClassificationPipeline, LandmarkRegressionPipeline]:
    return ToothDetectionPipeline(), PathologyClassificationPipeline(), LandmarkRegressionPipeline()


def _list_validation_images() -> list[Path]:
    if not VALIDATION_DIR.exists():
        return []
    return sorted(VALIDATION_DIR.glob("*.png"))


def _draw_landmarks(canvas_bgr: np.ndarray, landmarks: dict[str, list[list[float]]]) -> np.ndarray:
    h, w = canvas_bgr.shape[:2]
    for name, points in landmarks.items():
        color = LANDMARK_COLORS.get(name, (255, 255, 255))
        for x_norm, y_norm in points:
            cv2.circle(canvas_bgr, (int(x_norm * w), int(y_norm * h)), 3, color, -1)
    return canvas_bgr


# Uncertain sites (postprocessing/missing_tooth.py) are drawn as dashed outlines in
# this colour, distinct from every PATHOLOGY_COLORS entry, so a low-confidence
# "possible tooth" is visible without looking like a confirmed detection.
UNCERTAIN_COLOR_RGB = "rgb(255,176,0)"
UNCERTAIN_MERGE_IOU = 0.5


def _build_figure(
    canvas_rgb: np.ndarray,
    detections: list[dict],
    selected_idx: int | None,
    uncertain_teeth: list[dict] | None = None,
) -> go.Figure:
    """One go.Scatter fill-rectangle trace per tooth (trace index i+1, since the
    image itself is trace 0), so a click's curve_number maps straight back to
    detections[curve_number - 1]. hoveron="fills" makes the whole box -- not just
    its outline -- both hoverable and clickable, and the fill area scales with
    the image in data coordinates (unlike marker size, which is screen-pixel
    based and would drift out of alignment with the box when the chart is
    resized). Uncertain-site traces come after every detection trace, so their
    curve_numbers fall outside detections and clicks on them are ignored."""
    fig = px.imshow(canvas_rgb)
    fig.data[0].hoverinfo = "skip"  # don't show raw per-pixel RGB values on hover

    for i, det in enumerate(detections):
        x0, y0, x1, y1 = det["bbox"]
        is_selected = i == selected_idx
        base_color = PATHOLOGY_COLORS.get(det.get("pathology"), PATHOLOGY_COLORS[None])
        line_color = SELECTED_COLOR_RGB if is_selected else _bgr_to_rgb_str(base_color)
        fill_opacity = 0.35 if is_selected else 0.08

        pathology = det.get("pathology")
        conf_line = f"Detection conf: {det['confidence']:.2f}<br>"
        if pathology is not None:
            conf_line += f"Pathology: {pathology} ({det['pathology_confidence']:.2f})"
        hover = f"<b>FDI {det['fdi_number']}</b><br>{conf_line}<extra></extra>"

        fig.add_trace(
            go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, y1, y1, y0],
                mode="lines",
                fill="toself",
                fillcolor=line_color.replace("rgb", "rgba").replace(")", f",{fill_opacity})"),
                line=dict(color=line_color, width=4 if is_selected else 2),
                hoveron="fills",
                hovertemplate=hover,
                showlegend=False,
            )
        )

    reason_labels = {
        "possible_numbering_error": "may be a neighbouring tooth numbered one position off",
        "possible_detector_miss": "may be present but under-detected",
    }
    # One weak box often backs several neighbouring sites (e.g. the detector hedging
    # between 32 and 33 for the same blurred tooth) -- draw it once, labelled "32/33?".
    groups: list[list[dict]] = []
    for site in uncertain_teeth or []:
        if site.get("bbox") is None:
            continue
        group = next((g for g in groups if _iou(g[0]["bbox"], site["bbox"]) > UNCERTAIN_MERGE_IOU), None)
        if group is None:
            groups.append([site])
        else:
            group.append(site)

    for group in groups:
        x0, y0, x1, y1 = group[0]["bbox"]
        label = "/".join(site["fdi_number"] for site in group)
        reasons = "<br>".join(
            f"{site['fdi_number']}: {reason_labels.get(site['reason'], site['reason'])}"
            f" (conf {site['confidence']:.2f})"
            for site in group
        )
        hover = f"<b>FDI {label}? (uncertain, below reporting threshold)</b><br>{reasons}<extra></extra>"
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, y1, y1, y0],
                mode="lines",
                fill="toself",
                fillcolor=UNCERTAIN_COLOR_RGB.replace("rgb", "rgba").replace(")", ",0.04)"),
                line=dict(color=UNCERTAIN_COLOR_RGB, width=2, dash="dash"),
                hoveron="fills",
                hovertemplate=hover,
                showlegend=False,
            )
        )
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=y1,
            text=f"{label}?",
            showarrow=False,
            yshift=-10,
            font=dict(color=UNCERTAIN_COLOR_RGB, size=12),
        )

    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="pan",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _zoomed_crop(image: np.ndarray, bbox: list[float], pad_frac: float = 0.4) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = bbox
    pad_x, pad_y = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
    cx0, cy0 = max(0, int(x0 - pad_x)), max(0, int(y0 - pad_y))
    cx1, cy1 = min(w, int(x1 + pad_x)), min(h, int(y1 + pad_y))
    crop = image[cy0:cy1, cx0:cx1]
    target_w = 420
    scale = target_w / crop.shape[1] if crop.shape[1] else 1.0
    return cv2.resize(crop, (target_w, max(1, int(crop.shape[0] * scale))), interpolation=cv2.INTER_LINEAR)


def _style_table(df: pd.DataFrame, selected_idx: int | None) -> "pd.io.formats.style.Styler":
    def highlight(row: pd.Series) -> list[str]:
        if selected_idx is not None and row.name == selected_idx:
            return ["background-color: #ffe08a"] * len(row)
        return [""] * len(row)

    return df.style.apply(highlight, axis=1)


def _render_numbering_editor(result: dict) -> None:
    """Let the reviewing dentist renumber teeth, then save the chart as Stage 1
    training data (see numbering_corrections.py)."""
    detections = result["detections"]
    flagged = result["review_flags"]["numbering_review_quadrants"]
    with st.expander("Correct FDI numbering", expanded=bool(flagged)):
        if result.get("dentist_verified"):
            st.success("Numbering verified and saved. Edit and save again to amend it.")
        st.caption(
            "Teeth are listed upper arch then lower arch, each in X-ray left-to-right order, so a run "
            f"numbered one position off reads in sequence. Choose '{NOT_A_TOOTH}' for a duplicate box."
        )
        order = sorted(
            range(len(detections)),
            key=lambda i: (detections[i]["fdi_number"][0] in "34", sum(detections[i]["bbox"][0::2])),
        )
        editor_df = pd.DataFrame(
            [
                {
                    "Arch": "lower" if detections[i]["fdi_number"][0] in "34" else "upper",
                    "Centre x (px)": round(sum(detections[i]["bbox"][0::2]) / 2),
                    "Model FDI": model_fdi(detections[i]),
                    "FDI": detections[i]["fdi_number"],
                }
                for i in order
            ]
        )
        # Keyed by chart version so a saved correction reloads the editor from the new chart.
        editor_key = f"{result['filename']}_{result.get('chart_version', 0)}"
        edited = st.data_editor(
            editor_df,
            column_config={
                "FDI": st.column_config.SelectboxColumn(options=FDI_TOOTH_NUMBERS + [NOT_A_TOOTH], required=True)
            },
            disabled=["Arch", "Centre x (px)", "Model FDI"],
            hide_index=True,
            use_container_width=True,
            key=f"fdi_editor_{editor_key}",
        )
        complete = st.checkbox(
            "Every tooth visible in this X-ray has a box (required to use this chart for training)",
            key=f"fdi_complete_{editor_key}",
        )
        if not st.button("Save corrected chart", key=f"fdi_save_{editor_key}"):
            return

        corrected_fdis = [""] * len(detections)
        for row, det_idx in enumerate(order):
            corrected_fdis[det_idx] = edited["FDI"].iloc[row]
        problems = validate_corrections(corrected_fdis)
        if problems:
            st.error("Not saved: " + "; ".join(problems))
            return
        save_corrected_chart(
            result["image"], result["filename"], inference.MODEL_CHECKPOINT_PATH, detections, corrected_fdis, complete
        )
        corrected = apply_corrections(detections, corrected_fdis)
        fdi_numbers = [d["fdi_number"] for d in corrected]
        result.update(
            detections=corrected,
            # The dentist has confirmed the chart, so the triage and review flags no longer apply.
            missing_teeth=find_missing_teeth(fdi_numbers),
            uncertain_teeth=[],
            review_flags={"absent_wisdom_teeth": find_absent_wisdom_teeth(fdi_numbers), "numbering_review_quadrants": []},
            dentist_verified=True,
            chart_version=result.get("chart_version", 0) + 1,
        )
        st.session_state["selected_idx"] = None
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Dental AI - Model Verification", layout="wide")
    st.title("Dental AI - Model Verification Viewer")
    st.caption("Run trained checkpoints against DENTEX validation images (or your own upload) for visual QA.")

    with st.sidebar:
        st.header("Settings")

        # Stage 1 Checkpoint selection
        available_stage1_checkpoints = {
            "Enhanced (Rectangular 512x1024)": str(TRAINING_DIR / "runs/stage1_rect_v1/train/weights/best.pt"),
            "Baseline (Square 640x640)": str(TRAINING_DIR / "runs/stage1/train/weights/best.pt"),
        }
        existing_stage1 = {
            name: path for name, path in available_stage1_checkpoints.items() if Path(path).exists()
        }
        if existing_stage1:
            selected_s1 = st.selectbox("Stage 1 (Tooth Detection)", list(existing_stage1.keys()), index=0)
            if inference.MODEL_CHECKPOINT_PATH != existing_stage1[selected_s1]:
                inference.MODEL_CHECKPOINT_PATH = existing_stage1[selected_s1]
                _load_pipelines.clear()
                st.rerun()

        # Stage 2A Checkpoint selection
        available_stage2a_checkpoints = {
            "Enhanced (Crown 256x256 + Focal Loss)": {
                "weights": str(TRAINING_DIR / "runs/stage2a_crown256_focal/best.pt"),
                "classes": str(TRAINING_DIR / "runs/stage2a_crown256_focal/classes.json"),
            },
            "Baseline (128x128 Class-Weighted)": {
                "weights": str(TRAINING_DIR / "runs/stage2a/best.pt"),
                "classes": str(TRAINING_DIR / "runs/stage2a/classes.json"),
            },
        }
        existing_stage2a = {
            name: paths
            for name, paths in available_stage2a_checkpoints.items()
            if Path(paths["weights"]).exists()
        }
        if existing_stage2a:
            selected_s2a = st.selectbox("Stage 2A (Pathology / Caries)", list(existing_stage2a.keys()), index=0)
            target_weights = existing_stage2a[selected_s2a]["weights"]
            target_classes = existing_stage2a[selected_s2a]["classes"]
            if inference.PATHOLOGY_CHECKPOINT_PATH != target_weights:
                inference.PATHOLOGY_CHECKPOINT_PATH = target_weights
                inference.PATHOLOGY_CLASSES_PATH = target_classes
                _load_pipelines.clear()
                st.rerun()

    tooth_pipeline, pathology_pipeline, landmark_pipeline = _load_pipelines()

    with st.sidebar:
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

    if run_clicked:
        try:
            with st.spinner("Running inference..."):
                if mode == "tooth_detection":
                    detections, missing_teeth, uncertain_teeth, image = tooth_pipeline.run_with_image_bytes(
                        raw_bytes, filename
                    )
                    landmarks = None
                elif mode == "landmarks":
                    image = load_image(raw_bytes, filename=filename)
                    detections, missing_teeth, uncertain_teeth = [], [], []
                    landmarks = landmark_pipeline.run_with_image_bytes(raw_bytes, filename)
                else:
                    raw_detections, missing_teeth, uncertain_teeth, image = tooth_pipeline.run_with_image_bytes(
                        raw_bytes, filename
                    )
                    detections = pathology_pipeline.classify_crops(image, raw_detections)
                    landmarks = landmark_pipeline.run_with_image_bytes(raw_bytes, filename)
        except RuntimeError as exc:
            st.error(str(exc))
            return

        st.session_state["result"] = {
            "detections": detections,
            "missing_teeth": missing_teeth,
            "uncertain_teeth": uncertain_teeth,
            # Landmarks mode runs no tooth detection, so there's no chart to flag.
            "review_flags": None if mode == "landmarks" else chart_review_flags(detections),
            "landmarks": landmarks,
            "image": image,
            "filename": filename,
        }
        # A fresh run invalidates whatever tooth was selected before (it may not
        # even exist in the new detection set), and any pending click/row
        # selection from the previous result's widgets.
        st.session_state["selected_idx"] = None
        st.session_state.pop("_last_plot_sig", None)
        st.session_state.pop("_last_table_sig", None)

    result = st.session_state.get("result")
    if result is None:
        image = load_image(raw_bytes, filename=filename)
        st.image(image, caption=f"{filename} (not yet analyzed)", use_container_width=True)
        return

    detections = result["detections"]
    missing_teeth = result["missing_teeth"]
    uncertain_teeth = result.get("uncertain_teeth", [])
    landmarks = result["landmarks"]
    image = result["image"]
    filename = result["filename"]

    if only_caries:
        detections = [d for d in detections if d.get("pathology") == "caries"]

    selected_idx = st.session_state.get("selected_idx")
    if selected_idx is not None and selected_idx >= len(detections):
        selected_idx = None
        st.session_state["selected_idx"] = None

    canvas_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if landmarks:
        canvas_bgr = _draw_landmarks(canvas_bgr, landmarks)
    canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)

    if detections:
        fig = _build_figure(canvas_rgb, detections, selected_idx, None if only_caries else uncertain_teeth)
        plot_state = st.plotly_chart(fig, on_select="rerun", selection_mode="points", key="xray_chart")
        plot_points = (plot_state or {}).get("selection", {}).get("points", [])
        if plot_points:
            point = plot_points[0]
            new_idx = point.get("curve_number", 1) - 1
            sig = ("plot", point.get("curve_number"), point.get("x"), point.get("y"))
            if st.session_state.get("_last_plot_sig") != sig and 0 <= new_idx < len(detections):
                st.session_state["_last_plot_sig"] = sig
                st.session_state["selected_idx"] = new_idx
                st.rerun()
    else:
        st.image(canvas_rgb, caption=filename, use_container_width=True)

    st.subheader("Summary")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Caries detected" if only_caries else "Teeth detected", len(detections))
    metric_cols[1].metric("Missing teeth (FDI)", len(missing_teeth))
    metric_cols[2].metric("Uncertain sites (review)", len(uncertain_teeth))
    if missing_teeth:
        st.write("Likely missing: " + ", ".join(missing_teeth))
    if uncertain_teeth:
        reason_labels = {
            "possible_numbering_error": "may be a neighbouring tooth numbered one position off",
            "possible_detector_miss": "may be present but under-detected",
        }
        st.write(
            "Uncertain: "
            + ", ".join(f"{u['fdi_number']} ({reason_labels.get(u['reason'], u['reason'])})" for u in uncertain_teeth)
        )
    review_flags = result.get("review_flags")
    if review_flags:
        if review_flags["absent_wisdom_teeth"]:
            st.write("Absent third molars: " + ", ".join(review_flags["absent_wisdom_teeth"]))
        if review_flags["numbering_review_quadrants"]:
            st.warning(
                "Verify FDI numbering in quadrant(s) "
                + ", ".join(f"Q{q}" for q in review_flags["numbering_review_quadrants"])
                + ": a tooth is absent there, and the detector often numbers the remaining teeth "
                "one position off when neighbours have drifted into the space."
            )

    selected_idx = st.session_state.get("selected_idx")
    if detections:
        st.subheader("Per-tooth detections")
        st.caption("Click a box on the X-ray, or a row below, to link and zoom into that tooth.")
        rows_df = pd.DataFrame(
            [
                {
                    "FDI": d["fdi_number"],
                    "Detection conf.": round(d["confidence"], 3),
                    "Pathology": d.get("pathology", "-"),
                    "Pathology conf.": round(d["pathology_confidence"], 3) if "pathology_confidence" in d else None,
                }
                for d in detections
            ]
        )
        table_state = st.dataframe(
            _style_table(rows_df, selected_idx),
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="detections_table",
        )
        table_rows = (table_state or {}).get("selection", {}).get("rows", [])
        if table_rows:
            new_idx = table_rows[0]
            sig = ("table", new_idx)
            if st.session_state.get("_last_table_sig") != sig:
                st.session_state["_last_table_sig"] = sig
                st.session_state["selected_idx"] = new_idx
                st.rerun()

    if detections and review_flags is not None and not only_caries:
        _render_numbering_editor(result)

    selected_idx = st.session_state.get("selected_idx")
    if selected_idx is not None and 0 <= selected_idx < len(detections):
        det = detections[selected_idx]
        st.subheader(f"Zoomed inspection -- FDI {det['fdi_number']}")
        zoom_cols = st.columns([1, 2])
        with zoom_cols[0]:
            st.image(_zoomed_crop(image, det["bbox"]), caption=f"FDI {det['fdi_number']} (padded crop)")
        with zoom_cols[1]:
            st.write(f"**Detection confidence:** {det['confidence']:.3f}")
            if det.get("pathology") is not None:
                st.write(f"**Pathology:** {det['pathology']} ({det['pathology_confidence']:.3f} confidence)")
            if st.button("Clear selection"):
                st.session_state["selected_idx"] = None
                st.rerun()

    if landmarks:
        st.subheader("Landmark points")
        st.write({name: len(points) for name, points in landmarks.items()})

    with st.expander("Raw JSON"):
        st.json(
            {
                "detections": detections,
                "missing_teeth": missing_teeth,
                "uncertain_teeth": uncertain_teeth,
                "landmarks": landmarks,
            }
        )


if __name__ == "__main__":
    main()
