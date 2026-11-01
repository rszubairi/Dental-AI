import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inference_runs
from inference_runs import RunLogError, build_run, log_run

VERSION = {"id": "test+0000abcd", "release": "test", "components": {"code": "abc"}}


def _fields(**overrides) -> dict:
    return {
        "mode": "tooth_detection",
        "source": "Upload",
        "filename": "pano.png",
        "raw_bytes": b"xray",
        "model_version": VERSION,
        "settings": inference_runs.current_settings("tooth_detection"),
        "duration_s": 1.23456,
        **overrides,
    }


def test_successful_run_has_counts_and_outputs_but_no_error():
    outputs = {
        "detections": [{"fdi_number": "11", "bbox": [0, 0, 1, 1], "confidence": 0.9}],
        "missing_teeth": ["12", "13"],
        "uncertain_teeth": [],
        "review_flags": {"absent_wisdom_teeth": [], "numbering_review_quadrants": []},
        "landmarks": None,
    }
    run = build_run(**_fields(outputs=outputs))
    assert (run["teethCount"], run["missingCount"], run["uncertainCount"]) == (1, 2, 0)
    assert run["modelVersion"] == "test+0000abcd"
    assert run["sampleSha256"] == inference_runs.sample_digest(b"xray")
    assert run["durationS"] == 1.235
    assert "error" not in run and run["outputs"] is outputs


def test_failed_run_has_error_and_no_outputs():
    run = build_run(**_fields(error="No trained checkpoint"))
    assert run["error"] == "No trained checkpoint"
    assert "outputs" not in run  # Convex rejects null for optional fields
    assert run["teethCount"] == 0


def test_settings_cover_only_the_stages_a_mode_runs():
    assert set(inference_runs.current_settings("tooth_detection")) == {"stage1"}
    assert set(inference_runs.current_settings("landmarks")) == {"stage2b"}
    assert set(inference_runs.current_settings("full_assessment")) == {"stage1", "stage2a", "stage2b"}


def test_log_run_uploads_sample_then_run(monkeypatch):
    calls = []
    monkeypatch.setattr(inference_runs, "_request", lambda method, path, **kw: calls.append((method, path, kw)) or {})
    outputs = {"detections": [{"fdi_number": "11", "bbox": [0, 0, 1, 1], "confidence": np.float32(0.5)}]}
    run = log_run(**_fields(outputs=outputs))

    (_, sample_path, sample_kw), (_, run_path, run_kw) = calls
    assert sample_path == "/samples" and sample_kw["content"] == b"xray"
    assert sample_kw["params"]["sha256"] == run["sampleSha256"]
    assert run_path == "/runs"
    assert json.loads(run_kw["content"])["outputs"]["detections"][0]["confidence"] == 0.5  # numpy -> float


def test_unconfigured_log_raises(monkeypatch):
    monkeypatch.delenv("CONVEX_SITE_URL", raising=False)
    monkeypatch.delenv("INFERENCE_LOG_SECRET", raising=False)
    with pytest.raises(RunLogError):
        log_run(**_fields())
