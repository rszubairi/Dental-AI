"""Label Studio ML backend for landmark auto-labeling (Stage 2B U-Net model).

Pre-labels bone_crest / sinus_floor / nerve_canal as KeyPointLabels (one point
per detected local heatmap maximum) so future annotation rounds can start from
a draft curve to correct instead of drawing from scratch. Implements the
minimal Label Studio ML backend protocol directly, matching the pattern in
label_studio_ml_backend.py (Stage 1 tooth detection).

Endpoints:
    GET  /health   -> liveness check Label Studio pings before/after registration
    POST /setup    -> called when the backend is connected to a project
    POST /predict  -> called per batch of tasks; returns pre-annotations

Run:
    cd services/inference
    LANDMARK_CHECKPOINT=training/runs/stage2b/best.pt \
        uvicorn landmark_ml_backend:app --host 0.0.0.0 --port 9091

Then in Label Studio: Settings -> Model -> Connect Model, URL = http://localhost:9091
Your labeling config's KeyPointLabels must be named "bone_crest", "sinus_floor",
"nerve_canal" to match this model's output.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from inference import LandmarkRegressionPipeline

# Label Studio labeling-config names; override if your config uses different ones.
LS_FROM_NAME = os.environ.get("LS_FROM_NAME", "label")
LS_TO_NAME = os.environ.get("LS_TO_NAME", "image")
LS_IMAGE_DATA_KEY = os.environ.get("LS_IMAGE_DATA_KEY", "image")

# Needed to fetch images Label Studio stores itself (local storage / uploaded files),
# whose task URLs are host-relative and require an auth token.
LABEL_STUDIO_URL = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")
LABEL_STUDIO_ACCESS_TOKEN = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN", "")

app = FastAPI(title="Dental-AI Landmark Label Studio ML Backend")

_pipeline = LandmarkRegressionPipeline()


def _resolve_image_url(url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return f"{LABEL_STUDIO_URL.rstrip('/')}{url}"


def _predict_task(image_url: str) -> dict:
    resolved = _resolve_image_url(image_url)
    headers = {}
    if LABEL_STUDIO_ACCESS_TOKEN and resolved.startswith(LABEL_STUDIO_URL):
        headers["Authorization"] = f"Token {LABEL_STUDIO_ACCESS_TOKEN}"
    landmarks = _pipeline.run(resolved, headers=headers)

    annotation_results = []
    for structure, points in landmarks.items():
        for x, y in points:
            # Label Studio KeyPointLabels format: percentages of original image.
            annotation_results.append(
                {
                    "from_name": LS_FROM_NAME,
                    "to_name": LS_TO_NAME,
                    "type": "keypointlabels",
                    "value": {
                        "x": x * 100,
                        "y": y * 100,
                        "width": 0.3,
                        "keypointlabels": [structure],
                    },
                }
            )

    return {"result": annotation_results, "score": 0.0, "model_version": "stage2b-unet"}


class PredictRequest(BaseModel):
    tasks: list[dict]
    label_config: str | None = None
    project: str | None = None


@app.get("/health")
def health():
    return {"status": "UP", "model_class": "UNetLandmark-stage2b"}


@app.post("/setup")
def setup(_: dict | None = None):
    return {"model_version": "stage2b-unet"}


@app.post("/predict")
def predict(request: PredictRequest):
    predictions = []
    for task in request.tasks:
        image_url = task["data"][LS_IMAGE_DATA_KEY]
        predictions.append(_predict_task(image_url))
    return {"results": predictions}
