"""Gaussian heatmap generation/decoding for landmark regression (Stage 2B)."""

from __future__ import annotations

import numpy as np


def generate_gaussian_heatmap(
    canvas_size: int, points: list[tuple[float, float]], sigma: float = 10.0
) -> np.ndarray:
    """Render one channel: a canvas_size x canvas_size float32 heatmap with a 2D Gaussian
    centred at each (x, y) pixel coordinate, max-pooled into a single map.

    Returns an all-zero heatmap if no points are given (landmark absent from this image).
    """
    heatmap = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    if not points:
        return heatmap

    yy, xx = np.mgrid[0:canvas_size, 0:canvas_size]
    for x, y in points:
        gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        heatmap = np.maximum(heatmap, gaussian.astype(np.float32))
    return heatmap


def extract_peak(heatmap: np.ndarray) -> tuple[float, float]:
    """Return the (x, y) pixel coordinate of a heatmap's maximum value."""
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return float(x), float(y)


def extract_local_peak(heatmap: np.ndarray, center: tuple[float, float], window: float) -> tuple[float, float]:
    """Return the (x, y) pixel coordinate of the maximum value within `window`
    pixels of `center`. Used to evaluate curve-shaped landmarks (many ground
    truth points per structure): the global argmax is meaningless when a
    structure's heatmap has dozens of equally-high peaks along its length, so
    each ground truth point is instead matched against the model's response
    in its own local neighbourhood.
    """
    size = heatmap.shape[0]
    cx, cy = center
    x0, x1 = max(0, int(cx - window)), min(size, int(cx + window) + 1)
    y0, y1 = max(0, int(cy - window)), min(size, int(cy + window) + 1)
    patch = heatmap[y0:y1, x0:x1]
    if patch.size == 0:
        return center
    ly, lx = np.unravel_index(np.argmax(patch), patch.shape)
    return float(x0 + lx), float(y0 + ly)


def radial_error_mm(pred_xy: tuple[float, float], gt_xy: tuple[float, float], mm_per_pixel: float = 0.27) -> float:
    """Euclidean distance between predicted and ground-truth landmark peaks, in mm."""
    dx = pred_xy[0] - gt_xy[0]
    dy = pred_xy[1] - gt_xy[1]
    return float(np.hypot(dx, dy) * mm_per_pixel)
