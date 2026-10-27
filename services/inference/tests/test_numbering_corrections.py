import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numbering_corrections
from numbering_corrections import (
    NOT_A_TOOTH,
    apply_corrections,
    load_latest_charts,
    save_corrected_chart,
    validate_corrections,
)


def _det(fdi: str, x: float) -> dict:
    return {"fdi_number": fdi, "bbox": [x, 0.0, x + 10.0, 10.0], "confidence": 0.9}


def test_valid_corrections_have_no_problems():
    assert validate_corrections(["25", "26", NOT_A_TOOTH, NOT_A_TOOTH]) == []


def test_duplicate_and_unknown_numbers_are_rejected():
    problems = validate_corrections(["25", "25", "19"])
    assert "FDI 25 is assigned to 2 boxes" in problems
    assert "Unknown FDI number: 19" in problems


def test_apply_corrections_shifts_run_and_drops_non_teeth():
    # val_36-style upper-left run: model numbered 24, 26, 27; really 25, 26, 27, plus a duplicate box.
    detections = [_det("24", 0), _det("26", 20), _det("27", 40), _det("28", 45)]
    corrected = apply_corrections(detections, ["25", "26", "27", NOT_A_TOOTH])
    assert [d["fdi_number"] for d in corrected] == ["25", "26", "27"]
    assert corrected[0]["model_fdi_number"] == "24"
    assert "model_fdi_number" not in corrected[1]


def test_repeat_correction_keeps_model_original():
    once = apply_corrections([_det("24", 0)], ["25"])
    twice = apply_corrections(once, ["24"])
    assert twice[0]["fdi_number"] == "24"
    assert "model_fdi_number" not in twice[0]  # back to what the model said


def test_save_and_load_latest_chart(tmp_path, monkeypatch):
    monkeypatch.setattr(numbering_corrections, "CORRECTIONS_DIR", tmp_path)
    monkeypatch.setattr(numbering_corrections, "CORRECTIONS_LOG", tmp_path / "corrections.jsonl")
    image = np.zeros((20, 40), dtype=np.uint8)
    detections = [_det("24", 0), _det("26", 20)]

    save_corrected_chart(image, "val_36.png", "ckpt.pt", detections, ["24", "26"], complete=False)
    save_corrected_chart(image, "val_36.png", "ckpt.pt", detections, ["25", "26"], complete=True)

    charts = load_latest_charts(tmp_path / "corrections.jsonl")
    assert len(charts) == 1  # same image -> latest save wins
    assert charts[0]["complete"] is True
    assert [(t["model_fdi"], t["fdi"]) for t in charts[0]["teeth"]] == [("24", "25"), ("26", "26")]
    assert (tmp_path / charts[0]["image"]).exists()
    assert len((tmp_path / "corrections.jsonl").read_text().splitlines()) == 2
    json.loads((tmp_path / "corrections.jsonl").read_text().splitlines()[0])
