"""Shared pipeline state tracking for auto-resume across training stages.

Each training script marks its stage complete on success. Re-running a stage that's
already complete is a no-op unless --reset-step is passed, which clears the stage's
state entry and deletes its output directory so it starts fresh.

State lives in .dental_ai_state.json at the repo root, kept intentionally tiny
(stage name -> completion metadata only; actual model artifacts stay under runs/).
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

STATE_PATH = Path(__file__).resolve().parents[3] / ".dental_ai_state.json"


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    with open(STATE_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def is_step_complete(step_name: str) -> bool:
    return load_state().get(step_name, {}).get("complete", False)


def mark_step_complete(step_name: str, checkpoint_path: str, metric_name: str, metric_value: float) -> None:
    state = load_state()
    state[step_name] = {
        "complete": True,
        "checkpoint": checkpoint_path,
        metric_name: metric_value,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    save_state(state)


def reset_step(step_name: str, output_dir: Path) -> None:
    """Clear a stage's completion state and delete its output directory, after
    archiving any existing best.pt/classes.json first -- otherwise a retrain
    that turns out worse (or crashes) permanently loses the previous best
    checkpoint with no way to recover it."""
    state = load_state()
    prev_metric = {k: v for k, v in state.get(step_name, {}).items() if k not in ("complete", "checkpoint", "completed_at")}
    if step_name in state:
        del state[step_name]
        save_state(state)

    best_path = output_dir / "best.pt"
    if best_path.exists():
        archive_dir = output_dir.parent / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        metric_suffix = "_".join(f"{k}{v}" for k, v in prev_metric.items()) or "unknown"
        shutil.copy2(best_path, archive_dir / f"{step_name}_best_{metric_suffix}_{timestamp}.pt")
        classes_path = output_dir / "classes.json"
        if classes_path.exists():
            shutil.copy2(classes_path, archive_dir / f"{step_name}_classes_{metric_suffix}_{timestamp}.json")
        print(f"Archived previous {step_name} best.pt ({metric_suffix}) to {archive_dir} before reset.")

    if output_dir.exists():
        shutil.rmtree(output_dir)
