import json
import sys
from pathlib import Path

import httpx
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


def _fake_convex(monkeypatch, sample_exists: bool) -> tuple[list, list]:
    """Stub the HTTP actions and the storage upload URL; returns (action calls, uploads)."""
    calls, uploads = [], []

    def request(method, path, **kw):
        calls.append((method, path, kw))
        return {"exists": True} if sample_exists else {"exists": False, "uploadUrl": "https://upload"}

    def post(url, **kw):
        uploads.append((url, kw))
        return httpx.Response(200, json={"storageId": "kg2abc"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(inference_runs, "_request", request)
    monkeypatch.setattr(httpx, "post", post)
    return calls, uploads


def test_log_run_uploads_new_sample_via_upload_url_then_run(monkeypatch):
    calls, uploads = _fake_convex(monkeypatch, sample_exists=False)
    outputs = {"detections": [{"fdi_number": "11", "bbox": [0, 0, 1, 1], "confidence": np.float32(0.5)}]}
    run = log_run(**_fields(outputs=outputs))

    assert [path for _, path, _ in calls] == ["/samples/upload-url", "/samples", "/runs"]
    assert calls[0][2]["params"] == {"sha256": run["sampleSha256"]}
    # The bytes go to the storage upload URL (no HTTP action size limit), without the secret.
    [(url, upload_kw)] = uploads
    assert url == "https://upload" and upload_kw["content"] == b"xray"
    assert "Authorization" not in upload_kw["headers"]
    assert calls[1][2]["json"] == {"sha256": run["sampleSha256"], "storageId": "kg2abc", "filename": "pano.png"}
    assert json.loads(calls[2][2]["content"])["outputs"]["detections"][0]["confidence"] == 0.5  # numpy -> float


def test_log_run_skips_upload_of_already_stored_sample(monkeypatch):
    calls, uploads = _fake_convex(monkeypatch, sample_exists=True)
    log_run(**_fields())
    assert uploads == []
    assert [path for _, path, _ in calls] == ["/samples/upload-url", "/runs"]


def test_unconfigured_log_raises(monkeypatch):
    monkeypatch.delenv("CONVEX_SITE_URL", raising=False)
    monkeypatch.delenv("INFERENCE_LOG_SECRET", raising=False)
    with pytest.raises(RunLogError):
        log_run(**_fields())


def test_redirect_explains_wrong_site_url(monkeypatch):
    monkeypatch.setenv("CONVEX_SITE_URL", "https://dental-ai.streamlit.app")
    monkeypatch.setenv("INFERENCE_LOG_SECRET", "s")
    redirect = httpx.Response(303, headers={"location": "https://share.streamlit.io/-/auth/app"})
    monkeypatch.setattr(httpx, "request", lambda *a, **kw: redirect)
    with pytest.raises(RunLogError, match="convex.site"):
        inference_runs.list_versions()
