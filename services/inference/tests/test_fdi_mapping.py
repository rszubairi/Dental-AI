import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postprocessing.fdi_mapping import FDI_TOOTH_NUMBERS, index_to_fdi


def test_fdi_tooth_numbers_cover_all_four_quadrants():
    assert len(FDI_TOOTH_NUMBERS) == 32
    assert FDI_TOOTH_NUMBERS[0] == "11"
    assert FDI_TOOTH_NUMBERS[7] == "18"
    assert FDI_TOOTH_NUMBERS[8] == "21"
    assert FDI_TOOTH_NUMBERS[-1] == "48"


def test_index_to_fdi_round_trip():
    assert index_to_fdi(0) == "11"
    assert index_to_fdi(31) == "48"


def test_index_to_fdi_rejects_unknown_index():
    try:
        index_to_fdi(999)
        assert False, "expected ValueError"
    except ValueError:
        pass
