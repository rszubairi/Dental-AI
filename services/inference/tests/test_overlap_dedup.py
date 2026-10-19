import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postprocessing.overlap_dedup import deduplicate_overlapping_detections


def _det(fdi: str, bbox: list[float], confidence: float) -> dict:
    return {"fdi_number": fdi, "bbox": bbox, "confidence": confidence}


def test_no_overlap_keeps_all_detections():
    detections = [
        _det("11", [0, 0, 10, 10], 0.9),
        _det("12", [20, 0, 30, 10], 0.8),
    ]
    result = deduplicate_overlapping_detections(detections)
    assert {d["fdi_number"] for d in result} == {"11", "12"}


def test_same_class_near_duplicates_collapse_to_highest_confidence():
    # Mirrors observed validation data: one tooth, three heavily-overlapping boxes
    # all classified as "32", surviving Ultralytics' loose default NMS.
    detections = [
        _det("32", [552.5, 631.8, 591.3, 795.4], 0.69),
        _det("32", [551.1, 649.2, 589.0, 830.8], 0.48),
        _det("32", [552.9, 630.8, 589.5, 753.5], 0.26),
    ]
    result = deduplicate_overlapping_detections(detections)
    assert len(result) == 1
    assert result[0]["confidence"] == 0.69


def test_cross_class_overlap_collapses_to_one_physical_tooth():
    # The actual bug: one tooth produces boxes under two DIFFERENT FDI classes.
    # Per-class NMS can't catch this; this is the gap the guardrail closes.
    detections = [
        _det("31", [100, 100, 140, 180], 0.55),
        _det("41", [102, 98, 142, 182], 0.70),
    ]
    result = deduplicate_overlapping_detections(detections)
    assert len(result) == 1
    assert result[0]["fdi_number"] == "41"


def test_adjacent_teeth_with_slight_contact_are_not_merged():
    # Real neighboring teeth can touch slightly at the gumline but shouldn't be
    # treated as duplicates -- IoU of a small contact sliver stays well under the
    # threshold.
    detections = [
        _det("23", [0, 0, 20, 100], 0.9),
        _det("24", [19, 0, 40, 100], 0.85),
    ]
    result = deduplicate_overlapping_detections(detections)
    assert {d["fdi_number"] for d in result} == {"23", "24"}


def test_result_never_has_duplicate_fdi_numbers():
    detections = [
        _det("16", [0, 0, 50, 50], 0.9),
        _det("16", [200, 200, 250, 250], 0.4),  # non-overlapping but same FDI
    ]
    result = deduplicate_overlapping_detections(detections)
    assert len(result) == 1
    assert result[0]["confidence"] == 0.9
