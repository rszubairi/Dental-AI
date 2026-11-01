"""Identifies exactly which models and settings produced an inference result.

Checkpoint paths alone can't do this: retraining overwrites best.pt in place, and
the deployed Space downloads whatever is on the HF repo at startup. So a version
is built from what actually determines the output:

- each stage's weights, by content hash (plus classes.json for Stage 2A),
- the inference/post-processing code (a numbering fix in arch_graph_matcher.py
  changes the chart as much as new weights do),
- the thresholds in effect (env vars, or the Streamlit sliders),

and folded into one id, "<release>+<8-hex digest>", e.g. "2026.11.0+3f9a1c2b".
MODEL_RELEASE is the human label (download_checkpoints.py sets it from the pinned
HF revision); the digest changes whenever any component does, so two results with
the same id were produced by the same models, code and settings.

Dentist feedback records this id (numbering_corrections.py), so
training/compare_model_versions.py can score each version against it.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import inference

SERVICE_DIR = Path(__file__).resolve().parent
# Source files whose changes alter results. Paths relative to SERVICE_DIR.
CODE_GLOBS = ["inference.py", "postprocessing/*.py", "preprocessing/*.py", "training/unet_landmark_model.py"]
STAGES_BY_MODEL = {
    "tooth_detection": ("stage1",),
    "landmarks": ("stage2b",),
    "full_assessment": ("stage1", "stage2a", "stage2b"),
}
DIGEST_LENGTH = 12


def release() -> str:
    return os.environ.get("MODEL_RELEASE", "dev")


@lru_cache(maxsize=32)
def _hash_file(path: str, _size: int, _mtime_ns: int) -> str:
    # Size and mtime are only cache keys: a checkpoint replaced in place gets rehashed.
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files_digest(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if not os.path.exists(path):
            return "missing"
        stat = os.stat(path)
        digest.update(_hash_file(os.path.abspath(path), stat.st_size, stat.st_mtime_ns).encode())
    return digest.hexdigest()[:DIGEST_LENGTH]


@lru_cache(maxsize=1)
def code_digest() -> str:
    files = sorted({p for pattern in CODE_GLOBS for p in SERVICE_DIR.glob(pattern)})
    return _files_digest([str(p) for p in files])


def _stage_component(stage: str) -> dict[str, str]:
    if stage == "stage1":
        return {
            "weights": _files_digest([inference.MODEL_CHECKPOINT_PATH]),
            "settings": f"conf={inference.DETECTION_CONFIDENCE_THRESHOLD},iou={inference.DETECTION_NMS_IOU_THRESHOLD}",
        }
    if stage == "stage2a":
        return {
            "weights": _files_digest([inference.PATHOLOGY_CHECKPOINT_PATH, inference.PATHOLOGY_CLASSES_PATH]),
            "settings": f"conf={inference.PATHOLOGY_CONFIDENCE_THRESHOLD}",
        }
    return {
        "weights": _files_digest([inference.LANDMARK_CHECKPOINT_PATH]),
        "settings": f"peak={inference.LANDMARK_PEAK_THRESHOLD},filters={inference.LANDMARK_BASE_FILTERS}",
    }


def describe(model: str) -> dict:
    """Version of the given model ("tooth_detection" / "landmarks" / "full_assessment")
    as currently configured: {"id", "release", "components"}. Read at call time, so
    checkpoints or thresholds changed at runtime (Streamlit) are reflected."""
    components = {stage: _stage_component(stage) for stage in STAGES_BY_MODEL[model]}
    components["code"] = code_digest()
    digest = hashlib.sha256(json.dumps(components, sort_keys=True).encode()).hexdigest()[:8]
    return {"id": f"{release()}+{digest}", "release": release(), "components": components}
