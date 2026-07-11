"""Tooth detection + FDI numbering inference pipeline.

NOTE: No trained weights exist yet. `ToothDetectionPipeline.run` will raise
until a model checkpoint is produced by services/inference/training/train_tooth_detection.py
against annotated data (see datasets/tooth_detection/README.md).
"""

from __future__ import annotations

import os

import httpx

from postprocessing.fdi_mapping import index_to_fdi
from postprocessing.missing_tooth import find_missing_teeth
from preprocessing.image_loader import load_image, normalize_for_model

MODEL_CHECKPOINT_PATH = os.environ.get(
    "TOOTH_DETECTION_CHECKPOINT", "models/tooth_detection/weights/best.pt"
)


class ToothDetectionPipeline:
    def __init__(self) -> None:
        self._model = None  # lazy-loaded on first run to keep service startup fast

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not os.path.exists(MODEL_CHECKPOINT_PATH):
            raise RuntimeError(
                f"No trained checkpoint found at {MODEL_CHECKPOINT_PATH}. "
                "Train the tooth detection model first "
                "(services/inference/training/train_tooth_detection.py) "
                "using annotated data placed in datasets/tooth_detection/."
            )
        from ultralytics import YOLO

        self._model = YOLO(MODEL_CHECKPOINT_PATH)
        return self._model

    def run(self, image_url: str) -> dict:
        model = self._load_model()

        raw_bytes = httpx.get(image_url, timeout=30.0).content
        image = load_image(raw_bytes, filename=image_url)
        normalized = normalize_for_model(image)

        results = model.predict(normalized, verbose=False)[0]

        detections: list[dict] = []
        for box in results.boxes:
            class_index = int(box.cls.item())
            detections.append(
                {
                    "fdi_number": index_to_fdi(class_index),
                    "bbox": [float(v) for v in box.xyxy[0].tolist()],
                    "confidence": float(box.conf.item()),
                }
            )

        missing_teeth = find_missing_teeth([d["fdi_number"] for d in detections])
        return {"detections": detections, "missing_teeth": missing_teeth}
