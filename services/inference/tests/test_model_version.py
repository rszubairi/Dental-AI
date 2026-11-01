import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inference
import model_version


def _use_checkpoint(monkeypatch, path: Path) -> None:
    monkeypatch.setattr(inference, "MODEL_CHECKPOINT_PATH", str(path))


def test_same_weights_and_settings_give_same_version(tmp_path, monkeypatch):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"weights v1")
    _use_checkpoint(monkeypatch, ckpt)
    assert model_version.describe("tooth_detection") == model_version.describe("tooth_detection")


def test_retrained_weights_at_same_path_change_version(tmp_path, monkeypatch):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"weights v1")
    _use_checkpoint(monkeypatch, ckpt)
    before = model_version.describe("tooth_detection")["id"]
    ckpt.write_bytes(b"weights v2, retrained")
    assert model_version.describe("tooth_detection")["id"] != before


def test_threshold_change_changes_version(tmp_path, monkeypatch):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"weights v1")
    _use_checkpoint(monkeypatch, ckpt)
    before = model_version.describe("tooth_detection")["id"]
    monkeypatch.setattr(inference, "DETECTION_CONFIDENCE_THRESHOLD", 0.5)
    assert model_version.describe("tooth_detection")["id"] != before


def test_id_carries_release_label_and_only_used_stages(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_RELEASE", "2026.11.0")
    _use_checkpoint(monkeypatch, tmp_path / "absent.pt")
    version = model_version.describe("tooth_detection")
    assert version["id"].startswith("2026.11.0+")
    assert set(version["components"]) == {"stage1", "code"}
    assert version["components"]["stage1"]["weights"] == "missing"
