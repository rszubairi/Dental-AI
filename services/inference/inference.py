"""Tooth detection + FDI numbering, and per-tooth pathology classification, pipelines.

NOTE: No trained weights exist yet. Both pipelines' `run`/`classify_crops` will raise
until model checkpoints are produced by services/inference/training/train_tooth_detection.py
and train_pathology_classifier.py respectively.
"""

from __future__ import annotations

import json
import os

import cv2
import httpx
import numpy as np

from postprocessing.fdi_mapping import index_to_fdi
from postprocessing.missing_tooth import find_missing_teeth
from preprocessing.image_loader import load_image, normalize_for_model
from preprocessing.training_transforms import apply_clahe, extract_crop, to_rgb

MODEL_CHECKPOINT_PATH = os.environ.get(
    "TOOTH_DETECTION_CHECKPOINT", "models/tooth_detection/weights/best.pt"
)
PATHOLOGY_CHECKPOINT_PATH = os.environ.get(
    "PATHOLOGY_CLASSIFIER_CHECKPOINT", "models/pathology_classifier/weights/best.pt"
)
PATHOLOGY_CLASSES_PATH = os.environ.get(
    "PATHOLOGY_CLASSIFIER_CLASSES", "models/pathology_classifier/weights/classes.json"
)
PATHOLOGY_CROP_SIZE = 128
PATHOLOGY_CROP_PADDING = 0.2

LANDMARK_CHECKPOINT_PATH = os.environ.get(
    "LANDMARK_CHECKPOINT", "models/landmark_regression/weights/best.pt"
)
# Must match the --base-filters the checkpoint was trained with (see
# services/inference/training/train_landmark_regression.py).
LANDMARK_BASE_FILTERS = int(os.environ.get("LANDMARK_BASE_FILTERS", "16"))
LANDMARK_CANVAS_SIZE = 640
LANDMARK_CHANNEL_NAMES = ["bone_crest", "sinus_floor", "nerve_canal"]
# Local maxima below this heatmap value aren't reported as landmark points.
LANDMARK_PEAK_THRESHOLD = float(os.environ.get("LANDMARK_PEAK_THRESHOLD", "0.5"))
# Ground-truth points along these curves are spaced ~10px apart on the 640px
# canvas (see train_landmark_regression.py's LOCAL_PEAK_WINDOW comment) --
# keep local maxima at least this far apart so points aren't duplicated.
LANDMARK_MIN_PEAK_DISTANCE = 8


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
        detections, missing_teeth, _normalized_image = self.run_with_image(image_url)
        return {"detections": detections, "missing_teeth": missing_teeth}

    def run_with_image(self, image_url: str) -> tuple[list[dict], list[str], np.ndarray]:
        """Same as run(), but also returns the normalized image so callers (e.g. the
        pathology classifier) can extract crops without re-downloading/re-decoding."""
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
        return detections, missing_teeth, normalized


class PathologyClassificationPipeline:
    """Classifies pathology (caries, deep caries, periapical lesion, impacted, healthy)
    for each per-tooth crop, given bboxes from ToothDetectionPipeline."""

    def __init__(self) -> None:
        self._model = None  # lazy-loaded on first run to keep service startup fast
        self._classes: list[str] | None = None
        self._device = None
        self._transform = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not os.path.exists(PATHOLOGY_CHECKPOINT_PATH):
            raise RuntimeError(
                f"No trained checkpoint found at {PATHOLOGY_CHECKPOINT_PATH}. "
                "Train the pathology classifier first "
                "(services/inference/training/train_pathology_classifier.py) "
                "using crops produced by datasets/dentex/prepare_stage2a_dataset.py."
            )
        if not os.path.exists(PATHOLOGY_CLASSES_PATH):
            raise RuntimeError(f"Missing class mapping file at {PATHOLOGY_CLASSES_PATH}.")

        import torch
        from torchvision import models, transforms

        with open(PATHOLOGY_CLASSES_PATH, encoding="utf-8") as f:
            self._classes = json.load(f)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.efficientnet_b3()
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, len(self._classes))
        model.load_state_dict(torch.load(PATHOLOGY_CHECKPOINT_PATH, map_location=self._device))
        model.eval()
        self._model = model.to(self._device)

        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return self._model

    def classify_crops(self, image: np.ndarray, detections: list[dict]) -> list[dict]:
        """Given the grayscale image used for Stage 1 and its detections (with bboxes
        in that image's pixel space), return each detection annotated with a pathology
        prediction."""
        import torch

        model = self._load_model()
        enhanced = apply_clahe(image)

        results: list[dict] = []
        for det in detections:
            x0, y0, x1, y1 = det["bbox"]
            bbox_xywh = (x0, y0, x1 - x0, y1 - y0)
            crop = extract_crop(enhanced, bbox_xywh, PATHOLOGY_CROP_PADDING, PATHOLOGY_CROP_SIZE)
            rgb_crop = to_rgb(crop)
            tensor = self._transform(rgb_crop).unsqueeze(0).to(self._device)

            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
            class_idx = int(probs.argmax())

            results.append(
                {
                    **det,
                    "pathology": self._classes[class_idx],
                    "pathology_confidence": float(probs[class_idx]),
                }
            )
        return results


class LandmarkRegressionPipeline:
    """Predicts bone_crest / sinus_floor / nerve_canal curve points (Stage 2B,
    U-Net heatmap regression). Points are returned normalized 0-1, matching
    datasets/landmarks/README.md's annotation schema."""

    def __init__(self) -> None:
        self._model = None  # lazy-loaded on first run to keep service startup fast
        self._device = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not os.path.exists(LANDMARK_CHECKPOINT_PATH):
            raise RuntimeError(
                f"No trained checkpoint found at {LANDMARK_CHECKPOINT_PATH}. "
                "Train the landmark regression model first "
                "(services/inference/training/train_landmark_regression.py) "
                "using annotated data placed in datasets/landmarks/."
            )
        import torch

        from training.unet_landmark_model import UNetLandmark

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNetLandmark(base_filters=LANDMARK_BASE_FILTERS)
        model.load_state_dict(torch.load(LANDMARK_CHECKPOINT_PATH, map_location=self._device))
        model.eval()
        self._model = model.to(self._device)
        return self._model

    def run(self, image_url: str, headers: dict[str, str] | None = None) -> dict[str, list[list[float]]]:
        import torch
        from skimage.feature import peak_local_max

        model = self._load_model()

        response = httpx.get(image_url, headers=headers, timeout=30.0)
        response.raise_for_status()
        gray = load_image(response.content, filename=image_url)

        enhanced = apply_clahe(gray)
        canvas = cv2.resize(enhanced, (LANDMARK_CANVAS_SIZE, LANDMARK_CANVAS_SIZE), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(canvas).float().unsqueeze(0).unsqueeze(0) / 255.0
        tensor = tensor.to(self._device)

        with torch.no_grad():
            heatmaps = model(tensor).cpu().numpy()[0]

        landmarks: dict[str, list[list[float]]] = {}
        for c, name in enumerate(LANDMARK_CHANNEL_NAMES):
            peaks = peak_local_max(
                heatmaps[c],
                min_distance=LANDMARK_MIN_PEAK_DISTANCE,
                threshold_abs=LANDMARK_PEAK_THRESHOLD,
            )
            landmarks[name] = [
                [round(float(x) / LANDMARK_CANVAS_SIZE, 6), round(float(y) / LANDMARK_CANVAS_SIZE, 6)]
                for y, x in peaks
            ]
        return landmarks
