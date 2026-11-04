import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inference
from preprocessing.training_transforms import extract_crop, extract_crown_crop


@pytest.fixture
def checkpoint_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "PATHOLOGY_CLASSES_PATH", str(tmp_path / "classes.json"))
    return tmp_path


def test_checkpoint_without_recipe_gets_whole_tooth_default(checkpoint_dir):
    assert inference.load_pathology_crop() == {"style": "tooth", "size": 128, "padding": 0.2}


def test_crown_recipe_cuts_the_crown_crop_it_was_trained_on(checkpoint_dir):
    (checkpoint_dir / "crop.json").write_text('{"style": "crown", "size": 256}')
    crop = inference.load_pathology_crop()
    image = np.random.default_rng(0).integers(0, 255, (800, 1600), dtype=np.uint8)
    bbox = (700.0, 100.0, 90.0, 300.0)
    out = inference.extract_pathology_crop(image, bbox, crop)
    assert out.shape == (256, 256)
    np.testing.assert_array_equal(out, extract_crown_crop(image, bbox, output_size=256))
    assert not np.array_equal(out, extract_crop(image, bbox, 0.2, 256))


def test_unknown_crop_style_is_rejected(checkpoint_dir):
    (checkpoint_dir / "crop.json").write_text('{"style": "root"}')
    with pytest.raises(RuntimeError, match="Unknown crop style"):
        inference.load_pathology_crop()
