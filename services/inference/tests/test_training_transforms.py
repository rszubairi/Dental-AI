import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.training_transforms import (
    apply_clahe,
    build_multichannel_contrast,
    letterbox,
    preprocess_for_training,
    transform_bbox_xywh,
    xywh_to_yolo_line,
)


def test_letterbox_square():
    dummy = np.zeros((500, 1000), dtype=np.uint8)
    canvas, scale, pad_left, pad_top = letterbox(dummy, target_size=640)
    assert canvas.shape == (640, 640)
    assert scale == 0.64
    assert pad_top == (640 - int(500 * 0.64)) // 2


def test_letterbox_rectangular():
    dummy = np.zeros((1000, 2000), dtype=np.uint8)
    canvas, scale, pad_left, pad_top = letterbox(dummy, target_size=(512, 1024))
    assert canvas.shape == (512, 1024)
    # 2000 * (1024/2000) = 1024, 1000 * (1024/2000) = 512
    assert scale == 1024 / 2000
    assert pad_left == 0
    assert pad_top == 0


def test_multichannel_contrast_shape():
    dummy = np.full((100, 200), 128, dtype=np.uint8)
    composite = build_multichannel_contrast(dummy)
    assert composite.shape == (100, 200, 3)
    assert composite.dtype == np.uint8


def test_xywh_to_yolo_line_rectangular():
    # bbox at center of a 512x1024 canvas
    bbox = (256, 128, 512, 256)
    line = xywh_to_yolo_line(5, bbox, canvas_size=(512, 1024))
    parts = line.split()
    assert parts[0] == "5"
    assert pytest.approx(float(parts[1]), 1e-4) == (256 + 256) / 1024  # cx
    assert pytest.approx(float(parts[2]), 1e-4) == (128 + 128) / 512   # cy
    assert pytest.approx(float(parts[3]), 1e-4) == 512 / 1024          # nw
    assert pytest.approx(float(parts[4]), 1e-4) == 256 / 512           # nh
