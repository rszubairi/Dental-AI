import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postprocessing.missing_tooth import (
    EXPECTED_FDI_SET,
    REASON_DETECTOR_MISS,
    REASON_NUMBERING,
    assess_missing_teeth,
    find_missing_teeth,
)


def test_expected_set_excludes_wisdom_teeth():
    assert len(EXPECTED_FDI_SET) == 28
    for wisdom in ("18", "28", "38", "48"):
        assert wisdom not in EXPECTED_FDI_SET


def test_full_arch_detected_yields_no_missing_teeth():
    assert find_missing_teeth(sorted(EXPECTED_FDI_SET)) == []


def test_single_missing_tooth_detected():
    all_but_one = sorted(EXPECTED_FDI_SET - {"36"})
    assert find_missing_teeth(all_but_one) == ["36"]


def test_no_detections_flags_all_expected_teeth():
    assert find_missing_teeth([]) == sorted(EXPECTED_FDI_SET, key=lambda fdi: (fdi[0], fdi[1]))


def test_wisdom_tooth_absence_not_flagged():
    all_but_wisdom = sorted(EXPECTED_FDI_SET)  # wisdom teeth already excluded from expected set
    assert find_missing_teeth(all_but_wisdom) == []


def _det(fdi: str, bbox: list[float], confidence: float) -> dict:
    return {"fdi_number": fdi, "bbox": bbox, "confidence": confidence}


def _arch_without(*absent: str) -> list[dict]:
    """One non-overlapping box per expected tooth except `absent`."""
    return [
        _det(fdi, [i * 20.0, 0.0, i * 20.0 + 10.0, 10.0], 0.9)
        for i, fdi in enumerate(sorted(EXPECTED_FDI_SET))
        if fdi not in absent
    ]


def test_missing_site_without_evidence_is_likely_missing():
    likely, uncertain = assess_missing_teeth(_arch_without("36"), candidates=[])
    assert likely == ["36"]
    assert uncertain == []


def test_weak_alternative_label_on_detected_tooth_is_numbering_uncertainty():
    detections = _arch_without("16")
    tooth_17 = next(d for d in detections if d["fdi_number"] == "17")
    # The detector hedged: the same tooth also produced a weak "16" box.
    candidates = [_det("16", list(tooth_17["bbox"]), 0.15)]
    likely, uncertain = assess_missing_teeth(detections, candidates)
    assert likely == []
    assert uncertain == [{"fdi_number": "16", "reason": REASON_NUMBERING}]


def test_weak_box_in_empty_space_is_detector_miss_uncertainty():
    candidates = [_det("36", [5000.0, 0.0, 5010.0, 10.0], 0.05)]
    likely, uncertain = assess_missing_teeth(_arch_without("36"), candidates)
    assert likely == []
    assert uncertain == [{"fdi_number": "36", "reason": REASON_DETECTOR_MISS}]


def test_too_faint_evidence_is_ignored():
    candidates = [_det("36", [5000.0, 0.0, 5010.0, 10.0], 0.01)]
    likely, uncertain = assess_missing_teeth(_arch_without("36"), candidates)
    assert likely == ["36"]
    assert uncertain == []


def test_candidates_for_other_sites_do_not_affect_missing_site():
    candidates = [_det("46", [5000.0, 0.0, 5010.0, 10.0], 0.2)]
    likely, uncertain = assess_missing_teeth(_arch_without("36"), candidates)
    assert likely == ["36"]
    assert uncertain == []
