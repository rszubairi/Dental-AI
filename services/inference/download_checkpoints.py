"""Download trained model checkpoints from a Hugging Face model repo at
container startup, so the Space's Docker image itself stays small and
checkpoints can be updated without rebuilding the image.

Expects a single HF repo (default: env var HF_CHECKPOINTS_REPO) containing:
    stage1_tooth_detection/best.pt
    stage2a_pathology_classifier/best.pt
    stage2a_pathology_classifier/classes.json
    stage2b_landmark_regression/best.pt

Sets the same env vars inference.py already reads (TOOTH_DETECTION_CHECKPOINT,
etc.) to the downloaded local paths, so no other code needs to change.

Versioning: set HF_CHECKPOINTS_REVISION to a tag (e.g. "2026.11.0") or commit of
the HF repo to deploy that exact set of checkpoints; unset, it deploys the repo's
current main. Every file is downloaded from one resolved commit, so a push to the
repo mid-download can't mix checkpoints from two releases. MODEL_RELEASE (the label
in every response's model_version, see model_version.py) defaults to the tag, or
to the resolved commit's short sha.

Usage (called once at container startup, before uvicorn):
    python download_checkpoints.py
"""

from __future__ import annotations

import os
import sys

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = os.environ.get("HF_CHECKPOINTS_REPO")
REVISION = os.environ.get("HF_CHECKPOINTS_REVISION")
HF_TOKEN = os.environ.get("HF_TOKEN")  # only needed if the repo is private

FILES = {
    "TOOTH_DETECTION_CHECKPOINT": "stage1_tooth_detection/best.pt",
    "PATHOLOGY_CLASSIFIER_CHECKPOINT": "stage2a_pathology_classifier/best.pt",
    "PATHOLOGY_CLASSIFIER_CLASSES": "stage2a_pathology_classifier/classes.json",
    "LANDMARK_CHECKPOINT": "stage2b_landmark_regression/best.pt",
}


def main() -> None:
    if not REPO_ID:
        print("HF_CHECKPOINTS_REPO not set -- skipping checkpoint download.", file=sys.stderr)
        return

    commit = HfApi().model_info(REPO_ID, revision=REVISION, token=HF_TOKEN).sha
    print(f"Deploying checkpoints from {REPO_ID}@{REVISION or 'main'} (commit {commit})")
    lines = []
    for env_var, filename in FILES.items():
        local_path = hf_hub_download(repo_id=REPO_ID, filename=filename, revision=commit, token=HF_TOKEN)
        print(f"Downloaded {filename} -> {local_path}")
        lines.append(f"{env_var}={local_path}")
    if not os.environ.get("MODEL_RELEASE"):
        release = REVISION if REVISION and REVISION != commit else commit[:7]
        lines.append(f"MODEL_RELEASE={release}")
    lines.append(f"HF_CHECKPOINTS_COMMIT={commit}")

    # Write to a file that the container's start command sources before
    # launching uvicorn (see Dockerfile's CMD).
    with open("/tmp/checkpoint_env.sh", "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"export {line}\n")


if __name__ == "__main__":
    main()
