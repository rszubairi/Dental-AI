import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postprocessing.arch_graph_matcher import enforce_dental_arch_ordering


def test_enforce_dental_arch_ordering_removes_inversion():
    # Simulate upper arch sequence where an erroneous flipped 21 box appears before 11 spatially
    detections = [
        {"bbox": [100, 50, 140, 90], "fdi_number": "18", "confidence": 0.95},
        {"bbox": [200, 50, 240, 90], "fdi_number": "17", "confidence": 0.92},
        {"bbox": [300, 50, 340, 90], "fdi_number": "11", "confidence": 0.90},
        # Erroneous noise box: '21' appearing before '11' with lower confidence
        {"bbox": [280, 50, 310, 90], "fdi_number": "21", "confidence": 0.35},
        {"bbox": [400, 50, 440, 90], "fdi_number": "21", "confidence": 0.89},
        {"bbox": [500, 50, 540, 90], "fdi_number": "22", "confidence": 0.91},
    ]

    filtered = enforce_dental_arch_ordering(detections)
    fdis = [d["fdi_number"] for d in filtered]

    # Should retain 18 -> 17 -> 11 -> 21 -> 22 and discard the inverted low-confidence 21
    assert fdis == ["18", "17", "11", "21", "22"]


def test_enforce_dental_arch_ordering_preserves_both_arches():
    detections = [
        {"bbox": [100, 50, 140, 90], "fdi_number": "18", "confidence": 0.95},
        {"bbox": [400, 50, 440, 90], "fdi_number": "21", "confidence": 0.89},
        {"bbox": [100, 200, 140, 240], "fdi_number": "48", "confidence": 0.94},
        {"bbox": [400, 200, 440, 240], "fdi_number": "31", "confidence": 0.88},
    ]

    filtered = enforce_dental_arch_ordering(detections)
    assert len(filtered) == 4
    upper_fdis = [d["fdi_number"] for d in filtered if d["fdi_number"] in ["18", "21"]]
    lower_fdis = [d["fdi_number"] for d in filtered if d["fdi_number"] in ["48", "31"]]
    assert upper_fdis == ["18", "21"]
    assert lower_fdis == ["48", "31"]
