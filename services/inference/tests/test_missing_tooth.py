import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postprocessing.missing_tooth import EXPECTED_FDI_SET, find_missing_teeth


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
