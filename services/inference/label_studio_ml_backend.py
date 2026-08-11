"""Label Studio ML backend for tooth detection auto-labeling (Stage 1 YOLOv8 model).

Implements the minimal Label Studio ML backend protocol directly (no
label-studio-ml SDK dependency) so it can reuse this repo's own preprocessing
code and stay consistent with services/inference/inference.py.

Endpoints:
    GET  /health   -> liveness check Label Studio pings before/after registration
    POST /setup    -> called when the backend is connected to a project
    POST /predict  -> called per batch of tasks; returns pre-annotations

Run:
    cd services/inference
    TOOTH_DETECTION_CHECKPOINT=../../runs/detect/runs/stage1/train/weights/best.pt \
        uvicorn label_studio_ml_backend:app --host 0.0.0.0 --port 9090

Then in Label Studio: Settings -> Model -> Connect Model, URL = http://localhost:9090
(see https://api.labelstud.io/api-reference/api-reference/ml/create for the registration API).
"""

from __future__ import annotations

import os

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from postprocessing.fdi_mapping import index_to_fdi  # noqa: F401  (kept for parity/debugging)
from preprocessing.image_loader import load_image
from preprocessing.training_transforms import apply_clahe, letterbox, to_rgb

MODEL_CHECKPOINT_PATH = os.environ.get(
    "TOOTH_DETECTION_CHECKPOINT", "models/tooth_detection/weights/best.pt"
)
CANVAS_SIZE = 640
CONFIDENCE_THRESHOLD = float(os.environ.get("LS_CONFIDENCE_THRESHOLD", "0.25"))

# Label Studio labeling-config names; override if your config uses different ones.
LS_FROM_NAME = os.environ.get("LS_FROM_NAME", "label")
LS_TO_NAME = os.environ.get("LS_TO_NAME", "image")
LS_IMAGE_DATA_KEY = os.environ.get("LS_IMAGE_DATA_KEY", "image")

# Needed to fetch images Label Studio stores itself (local storage / uploaded files),
# whose task URLs are host-relative and require an auth token.
LABEL_STUDIO_URL = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")
LABEL_STUDIO_ACCESS_TOKEN = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN", "")

app = FastAPI(title="Dental-AI Label Studio ML Backend")

_model = None


def _load_model():
    global _model
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO(MODEL_CHECKPOINT_PATH)
    return _model


def _resolve_image_url(url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return f"{LABEL_STUDIO_URL.rstrip('/')}{url}"


def _fetch_image_bytes(url: str) -> bytes:
    resolved = _resolve_image_url(url)
    headers = {}
    if LABEL_STUDIO_ACCESS_TOKEN and resolved.startswith(LABEL_STUDIO_URL):
        headers["Authorization"] = f"Token {LABEL_STUDIO_ACCESS_TOKEN}"
    response = httpx.get(resolved, headers=headers, timeout=30.0)
    response.raise_for_status()
    return response.content


def _predict_task(image_url: str) -> dict:
    model = _load_model()

    raw_bytes = _fetch_image_bytes(image_url)
    gray = load_image(raw_bytes, filename=image_url)
    orig_h, orig_w = gray.shape[:2]

    enhanced = apply_clahe(gray)
    canvas, scale, pad_left, pad_top = letterbox(enhanced, target_size=CANVAS_SIZE)
    rgb = to_rgb(canvas)

    results = model.predict(rgb, verbose=False)[0]

    annotation_results = []
    scores = []
    for box in results.boxes:
        confidence = float(box.conf.item())
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        class_index = int(box.cls.item())
        fdi_label = model.names[class_index]

        # Undo letterbox: canvas coords -> original image pixel coords.
        cx1, cy1, cx2, cy2 = [float(v) for v in box.xyxy[0].tolist()]
        x1 = (cx1 - pad_left) / scale
        y1 = (cy1 - pad_top) / scale
        x2 = (cx2 - pad_left) / scale
        y2 = (cy2 - pad_top) / scale

        # Label Studio RectangleLabels format: percentages of original image, top-left origin.
        annotation_results.append(
            {
                "from_name": LS_FROM_NAME,
                "to_name": LS_TO_NAME,
                "type": "rectanglelabels",
                "score": confidence,
                "value": {
                    "x": max(0.0, x1 / orig_w * 100),
                    "y": max(0.0, y1 / orig_h * 100),
                    "width": min(100.0, (x2 - x1) / orig_w * 100),
                    "height": min(100.0, (y2 - y1) / orig_h * 100),
                    "rotation": 0,
                    "rectanglelabels": [fdi_label],
                },
            }
        )
        scores.append(confidence)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return {"result": annotation_results, "score": avg_score, "model_version": "stage1-yolov8"}


class PredictRequest(BaseModel):
    tasks: list[dict]
    label_config: str | None = None
    project: str | None = None


@app.get("/health")
def health():
    return {"status": "UP", "model_class": "YOLOv8-stage1"}


@app.post("/setup")
def setup(_: dict | None = None):
    return {"model_version": "stage1-yolov8"}


@app.post("/predict")
def predict(request: PredictRequest):
    predictions = []
    for task in request.tasks:
        image_url = task["data"][LS_IMAGE_DATA_KEY]
        predictions.append(_predict_task(image_url))
    return {"results": predictions}
