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

Usage (called once at container startup, before uvicorn):
    python download_checkpoints.py
"""

from __future__ import annotations

import os
import sys

from huggingface_hub import hf_hub_download

REPO_ID = os.environ.get("HF_CHECKPOINTS_REPO")
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

    env_path = os.environ.get("GITHUB_ENV") or "/tmp/checkpoint_env"
    lines = []
    for env_var, filename in FILES.items():
        local_path = hf_hub_download(repo_id=REPO_ID, filename=filename, token=HF_TOKEN)
        print(f"Downloaded {filename} -> {local_path}")
        lines.append(f"{env_var}={local_path}")

    # Write to a file that the container's start command sources before
    # launching uvicorn (see Dockerfile's CMD).
    with open("/tmp/checkpoint_env.sh", "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"export {line}\n")


if __name__ == "__main__":
    main()
