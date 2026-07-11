"""Training-time image preprocessing shared by dataset preparation and training scripts.

Mirrors the preprocessing pipeline in the client-facing dataset spec: CLAHE contrast
enhancement, aspect-preserving letterbox resize to a fixed square canvas, grayscale
to RGB replication, and COCO->YOLO bbox/coordinate remapping.
"""

from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Enhance local contrast on a single-channel uint8 image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def letterbox(
    image: np.ndarray, target_size: int = 640, pad_value: int = 0
) -> tuple[np.ndarray, float, int, int]:
    """Resize preserving aspect ratio, padding to a square target_size x target_size canvas.

    Returns (letterboxed_image, scale, pad_left, pad_top) so bboxes can be remapped.
    """
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    pad_right = target_size - new_w - pad_left
    pad_bottom = target_size - new_h - pad_top

    canvas = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=pad_value,
    )
    return canvas, scale, pad_left, pad_top


def transform_bbox_xywh(
    bbox: tuple[float, float, float, float], scale: float, pad_left: int, pad_top: int
) -> tuple[float, float, float, float]:
    """Remap a COCO-style [x, y, w, h] bbox (top-left origin) into letterboxed canvas coords."""
    x, y, w, h = bbox
    return (
        x * scale + pad_left,
        y * scale + pad_top,
        w * scale,
        h * scale,
    )


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Replicate a single-channel grayscale image into 3 identical channels."""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def preprocess_for_training(image: np.ndarray, target_size: int = 640) -> tuple[np.ndarray, float, int, int]:
    """Full pipeline: CLAHE -> letterbox -> RGB. Returns (image, scale, pad_left, pad_top)."""
    enhanced = apply_clahe(image)
    canvas, scale, pad_left, pad_top = letterbox(enhanced, target_size=target_size)
    rgb = to_rgb(canvas)
    return rgb, scale, pad_left, pad_top


def xywh_to_yolo_line(
    class_idx: int, bbox_xywh: tuple[float, float, float, float], canvas_size: int
) -> str:
    """Convert an absolute [x, y, w, h] (top-left origin) bbox to a normalized YOLO label line."""
    x, y, w, h = bbox_xywh
    cx = (x + w / 2) / canvas_size
    cy = (y + h / 2) / canvas_size
    nw = w / canvas_size
    nh = h / canvas_size
    return f"{class_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def extract_crop(
    image: np.ndarray,
    bbox_xywh: tuple[float, float, float, float],
    padding_frac: float = 0.2,
    output_size: int = 128,
) -> np.ndarray:
    """Extract a padded, resized crop around a bbox (used for Stage 2A disease crops)."""
    h, w = image.shape[:2]
    x, y, bw, bh = bbox_xywh
    pad_x, pad_y = bw * padding_frac, bh * padding_frac

    x0 = max(0, int(x - pad_x))
    y0 = max(0, int(y - pad_y))
    x1 = min(w, int(x + bw + pad_x))
    y1 = min(h, int(y + bh + pad_y))

    crop = image[y0:y1, x0:x1]
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
