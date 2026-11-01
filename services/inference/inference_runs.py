"""Log of every inference run made from streamlit_app.py, kept in Convex.

Each run records the sample it ran on, the model version (model_version.py) and
every setting in effect, how long it took, and what it returned -- or the error,
for failed runs. The X-ray itself is stored once in Convex storage, keyed by the
sha256 of its bytes, so the same X-ray's runs can be found across versions.

Talks to the secret-protected HTTP routes in apps/web/convex/inferenceRuns.ts.
Configure with:
    CONVEX_SITE_URL       the deployment's HTTP actions URL (https://<name>.convex.site)
    INFERENCE_LOG_SECRET  the same value set on the deployment
                          (npx convex env set INFERENCE_LOG_SECRET ...)

Dentist corrections (numbering_corrections.py) record the run_id they correct.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid

import httpx

import inference

REQUEST_TIMEOUT_S = 60.0


class RunLogError(RuntimeError):
    """The run log is unconfigured or Convex couldn't be reached."""


def is_configured() -> bool:
    return bool(os.environ.get("CONVEX_SITE_URL") and os.environ.get("INFERENCE_LOG_SECRET"))


def _request(method: str, path: str, **kwargs) -> dict | list:
    if not is_configured():
        raise RunLogError("Inference run log is off: set CONVEX_SITE_URL and INFERENCE_LOG_SECRET.")
    url = os.environ["CONVEX_SITE_URL"].rstrip("/") + "/inference-log" + path
    headers = {"Authorization": f"Bearer {os.environ['INFERENCE_LOG_SECRET']}", **kwargs.pop("headers", {})}
    try:
        response = httpx.request(method, url, headers=headers, timeout=REQUEST_TIMEOUT_S, **kwargs)
        # Convex HTTP actions never redirect: a redirect means CONVEX_SITE_URL points
        # somewhere else (e.g. the Streamlit app's own URL, behind its login page).
        if response.is_redirect:
            raise RunLogError(
                f"CONVEX_SITE_URL ({os.environ['CONVEX_SITE_URL']}) redirected to "
                f"{response.headers.get('location', 'another page')}: it should be the Convex "
                "deployment's HTTP actions URL, https://<name>.convex.site."
            )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RunLogError(f"Convex run log returned {exc.response.status_code}: {exc.response.text}") from exc
    except httpx.HTTPError as exc:
        raise RunLogError(f"Could not reach the Convex run log: {exc}") from exc
    return response.json()


def sample_digest(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def current_settings(mode: str) -> dict:
    """Checkpoint paths and thresholds the given mode runs with, as currently set
    (Streamlit changes them at runtime)."""
    settings: dict = {}
    if mode in ("tooth_detection", "full_assessment"):
        settings["stage1"] = {
            "checkpoint": inference.MODEL_CHECKPOINT_PATH,
            "confidence_threshold": inference.DETECTION_CONFIDENCE_THRESHOLD,
            "nms_iou_threshold": inference.DETECTION_NMS_IOU_THRESHOLD,
            "candidate_confidence_floor": inference.CANDIDATE_CONFIDENCE_FLOOR,
        }
    if mode == "full_assessment":
        settings["stage2a"] = {
            "checkpoint": inference.PATHOLOGY_CHECKPOINT_PATH,
            "classes": inference.PATHOLOGY_CLASSES_PATH,
            "confidence_threshold": inference.PATHOLOGY_CONFIDENCE_THRESHOLD,
        }
    if mode in ("landmarks", "full_assessment"):
        settings["stage2b"] = {
            "checkpoint": inference.LANDMARK_CHECKPOINT_PATH,
            "peak_threshold": inference.LANDMARK_PEAK_THRESHOLD,
            "base_filters": inference.LANDMARK_BASE_FILTERS,
        }
    return settings


def build_run(
    *,
    mode: str,
    source: str,
    filename: str,
    raw_bytes: bytes,
    model_version: dict,
    settings: dict,
    duration_s: float,
    outputs: dict | None = None,
    error: str | None = None,
) -> dict:
    """The run record as stored in Convex (inferenceRuns table).

    Args:
        source: where the X-ray came from ("Validation set" / "Upload").
        model_version, settings: captured before the run, as it ran.
        outputs: detections, missing_teeth, uncertain_teeth, review_flags and
            landmarks as returned; None for a failed run, with error set instead.
    """
    outputs = outputs or {}
    run = {
        "runId": uuid.uuid4().hex,
        "ranAt": int(time.time() * 1000),
        "mode": mode,
        "source": source,
        "filename": filename,
        "sampleSha256": sample_digest(raw_bytes),
        "modelVersion": model_version["id"],
        "modelVersionComponents": model_version["components"],
        "settings": settings,
        "durationS": round(duration_s, 3),
        "teethCount": len(outputs.get("detections") or []),
        "missingCount": len(outputs.get("missing_teeth") or []),
        "uncertainCount": len(outputs.get("uncertain_teeth") or []),
    }
    # Convex rejects null for optional fields: leave them out instead.
    if error is None:
        run["outputs"] = outputs
    else:
        run["error"] = error
    return run


def log_run(*, raw_bytes: bytes, **run_fields) -> dict:
    """Store the X-ray (once per distinct X-ray) and the run in Convex. Takes
    build_run's arguments; returns the stored run. Raises RunLogError on failure."""
    run = build_run(raw_bytes=raw_bytes, **run_fields)
    _request(
        "POST",
        "/samples",
        params={"sha256": run["sampleSha256"], "filename": run["filename"]},
        content=raw_bytes,
        headers={"Content-Type": "application/octet-stream"},
    )
    _request(
        "POST",
        "/runs",
        # default=float: numpy scalars that slipped into outputs.
        content=json.dumps(run, default=float),
        headers={"Content-Type": "application/json"},
    )
    return run


def list_versions() -> list[dict]:
    """Per-version totals (inferenceVersions table), most recently run first."""
    return _request("GET", "/versions")


def list_runs(*, model_version: str | None = None, sample_sha256: str | None = None, limit: int = 200) -> list[dict]:
    """Runs of one model version, or on one X-ray, newest first -- without outputs."""
    params = {"limit": limit}
    if model_version is not None:
        params["modelVersion"] = model_version
    if sample_sha256 is not None:
        params["sampleSha256"] = sample_sha256
    return _request("GET", "/runs", params=params)


def get_run(run_id: str) -> dict:
    """One run with its outputs, plus "sampleUrl" to download its X-ray."""
    return _request("GET", "/run", params={"runId": run_id})


def fetch_sample(sample_url: str) -> bytes:
    try:
        response = httpx.get(sample_url, timeout=REQUEST_TIMEOUT_S)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RunLogError(f"Could not download the run's X-ray: {exc}") from exc
    return response.content
