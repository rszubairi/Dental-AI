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


def build_multichannel_contrast(image: np.ndarray) -> np.ndarray:
    """Build a rich 3-channel composite from a single-channel grayscale dental OPG:
    - Channel 0: Unsharp-masked contrast (sharpens enamel-cementum boundaries & interdental contacts).
    - Channel 1: High-frequency local CLAHE (reveals fine alveolar bone peaks and tooth root apices).
    - Channel 2: Gamma-adjusted curve (gamma=0.75) to illuminate underexposed molar ramus and impacted wisdom teeth.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Channel 0: Unsharp mask
    gaussian = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
    unsharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)

    # Channel 1: Local CLAHE
    clahe = apply_clahe(gray, clip_limit=2.5, tile_grid_size=(8, 8))

    # Channel 2: Gamma correction (gamma=0.75 boosts dark bony ramus and third-molar crypts)
    inv_gamma = 1.0 / 0.75
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_boost = cv2.LUT(gray, table)

    return np.dstack([unsharp, clahe, gamma_boost])


def letterbox(
    image: np.ndarray,
    target_size: int | tuple[int, int] = 640,
    pad_value: int = 0,
) -> tuple[np.ndarray, float, int, int]:
    """Resize preserving aspect ratio, padding to target canvas.

    target_size can be a single int (square target_size x target_size) or a tuple
    (target_height, target_width).

    Returns (letterboxed_image, scale, pad_left, pad_top) so bboxes can be remapped.
    """
    if isinstance(target_size, (tuple, list)):
        target_h, target_w = int(target_size[0]), int(target_size[1])
    else:
        target_h = target_w = int(target_size)

    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    pad_right = target_w - new_w - pad_left
    pad_bottom = target_h - new_h - pad_top

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
    if len(image.shape) == 3 and image.shape[2] == 3:
        return image
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def preprocess_for_training(
    image: np.ndarray,
    target_size: int | tuple[int, int] = 640,
    use_composite: bool = False,
) -> tuple[np.ndarray, float, int, int]:
    """Full pipeline: contrast enhancement -> letterbox.

    If use_composite is True, constructs a 3-channel (unsharp, CLAHE, gamma) composite.
    Otherwise applies standard CLAHE -> letterbox -> to_rgb.
    Returns (image, scale, pad_left, pad_top).
    """
    if use_composite:
        enhanced = build_multichannel_contrast(image)
    else:
        enhanced = apply_clahe(image)

    canvas, scale, pad_left, pad_top = letterbox(enhanced, target_size=target_size)
    rgb = canvas if use_composite else to_rgb(canvas)
    return rgb, scale, pad_left, pad_top


def xywh_to_yolo_line(
    class_idx: int,
    bbox_xywh: tuple[float, float, float, float],
    canvas_size: int | tuple[int, int],
) -> str:
    """Convert an absolute [x, y, w, h] (top-left origin) bbox to a normalized YOLO label line."""
    if isinstance(canvas_size, (tuple, list)):
        target_h, target_w = float(canvas_size[0]), float(canvas_size[1])
    else:
        target_h = target_w = float(canvas_size)

    x, y, w, h = bbox_xywh
    cx = (x + w / 2) / target_w
    cy = (y + h / 2) / target_h
    nw = w / target_w
    nh = h / target_h
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


def extract_crown_crop(
    image: np.ndarray,
    bbox_xywh: tuple[float, float, float, float],
    fdi_number: str | None = None,
    output_size: int = 256,
) -> np.ndarray:
    """Extract a crown-focused crop targeting enamel, dentin, and interproximal margins.
    
    In a panoramic radiograph:
    - Maxillary teeth (quadrants 1 and 2): Crowns are at the bottom of the tooth bounding box.
    - Mandibular teeth (quadrants 3 and 4): Crowns are at the top of the tooth bounding box.
    - Focusing on the coronal 60% of the tooth eliminates confounding root canal fillings,
      apical lesions, and maxillary sinus shadows from corrupting caries classification.
    """
    h, w = image.shape[:2]
    x, y, bw, bh = bbox_xywh

    # Determine if maxillary (quadrant 1 or 2) or mandibular (quadrant 3 or 4)
    is_maxillary = True
    if fdi_number and len(fdi_number) >= 2:
        quadrant = fdi_number[0]
        if quadrant in ("3", "4"):
            is_maxillary = False
    else:
        # Fallback heuristic: check vertical position relative to image center
        cy = y + bh / 2.0
        is_maxillary = cy < (h / 2.0)

    # Focus on the coronal 60% + horizontal margin for interproximal contacts
    pad_x = bw * 0.15
    x0 = max(0, int(x - pad_x))
    x1 = min(w, int(x + bw + pad_x))

    if is_maxillary:
        # Crown is towards bottom of the tooth box
        y0 = max(0, int(y + bh * 0.35))
        y1 = min(h, int(y + bh * 1.05))
    else:
        # Crown is towards top of the tooth box
        y0 = max(0, int(y - bh * 0.05))
        y1 = min(h, int(y + bh * 0.65))

    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return extract_crop(image, bbox_xywh, padding_frac=0.1, output_size=output_size)
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
