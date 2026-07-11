"""Load and normalize dental X-ray images (DICOM or standard formats) for inference."""

from __future__ import annotations

import io

import cv2
import numpy as np
import pydicom
from PIL import Image


def load_image(source: bytes, filename: str) -> np.ndarray:
    """Return a single-channel uint8 numpy array regardless of source format."""
    if filename.lower().endswith((".dcm", ".dicom")):
        return _load_dicom(source)
    return _load_standard(source)


def _load_dicom(raw: bytes) -> np.ndarray:
    ds = pydicom.dcmread(io.BytesIO(raw))
    pixels = ds.pixel_array.astype(np.float32)
    pixels -= pixels.min()
    max_val = pixels.max() or 1.0
    pixels = (pixels / max_val) * 255.0
    return pixels.astype(np.uint8)


def _load_standard(raw: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(raw)).convert("L")
    return np.array(image)


def normalize_for_model(image: np.ndarray, target_size: tuple[int, int] = (1024, 1024)) -> np.ndarray:
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    equalized = cv2.equalizeHist(resized)
    return equalized
