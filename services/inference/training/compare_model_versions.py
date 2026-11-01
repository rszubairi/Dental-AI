"""Compare model versions by how much dentists had to correct their charts.

Reads the dentist corrections log (numbering_corrections.py), where each saved
chart records the model_version that produced it (model_version.py), and reports
per version:

- charts reviewed, and the share accepted with no change at all,
- renumber rate: of the model's boxes, the share the dentist gave another FDI number,
- false-box rate: of the model's boxes, the share marked "not a tooth".

Lower rates are better. Versions reviewed on different X-rays aren't strictly
comparable (a harder batch of cases inflates the rates), so --baseline and
--candidate also compare two versions on only the X-rays reviewed under both.

Charts saved before versioning was introduced are grouped as "unversioned".

Usage (from services/inference):
    python training/compare_model_versions.py
    python training/compare_model_versions.py --baseline 2026.10.0+1a2b3c4d --candidate 2026.11.0+5e6f7a8b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

INFERENCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INFERENCE_DIR))

from numbering_corrections import CORRECTIONS_LOG, NOT_A_TOOTH  # noqa: E402

UNVERSIONED = "unversioned"


def load_latest_per_version(log_path: Path) -> dict[str, dict[str, dict]]:
    """version id -> image -> that image's most recent chart under that version.
    Unlike load_latest_charts, a re-review under a newer version doesn't supersede
    the older one -- both are needed to compare the versions."""
    charts: dict[str, dict[str, dict]] = {}
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                version = record.get("model_version", UNVERSIONED)
                charts.setdefault(version, {})[record["image"]] = record
    return charts


def chart_stats(record: dict) -> dict[str, int]:
    teeth = record["teeth"]
    false_boxes = sum(t["fdi"] == NOT_A_TOOTH for t in teeth)
    renumbered = sum(t["fdi"] != NOT_A_TOOTH and t["fdi"] != t["model_fdi"] for t in teeth)
    return {"boxes": len(teeth), "renumbered": renumbered, "false_boxes": false_boxes}


def summarize(records: list[dict]) -> dict[str, float]:
    stats = [chart_stats(r) for r in records]
    boxes = sum(s["boxes"] for s in stats)
    return {
        "charts": len(stats),
        "boxes": boxes,
        "accepted_as_is": sum(s["renumbered"] + s["false_boxes"] == 0 for s in stats) / len(stats) if stats else 0.0,
        "renumber_rate": sum(s["renumbered"] for s in stats) / boxes if boxes else 0.0,
        "false_box_rate": sum(s["false_boxes"] for s in stats) / boxes if boxes else 0.0,
    }


def _print_table(rows: list[tuple[str, dict]]) -> None:
    width = max([len("version")] + [len(name) for name, _ in rows])
    print(f"{'version':<{width}}  charts  boxes  accepted  renumbered  false boxes")
    for name, s in rows:
        print(
            f"{name:<{width}}  {s['charts']:>6}  {s['boxes']:>5}  {s['accepted_as_is']:>8.1%}"
            f"  {s['renumber_rate']:>10.1%}  {s['false_box_rate']:>11.1%}"
        )


def compare_pair(charts: dict[str, dict[str, dict]], baseline: str, candidate: str) -> None:
    for version in (baseline, candidate):
        if version not in charts:
            sys.exit(f"No charts recorded for version {version}. Known: {', '.join(sorted(charts))}")
    shared = sorted(charts[baseline].keys() & charts[candidate].keys())
    print(f"\nSame X-rays reviewed under both versions: {len(shared)}")
    if not shared:
        return
    _print_table([(v, summarize([charts[v][img] for img in shared])) for v in (baseline, candidate)])

    def corrections(version: str, image: str) -> int:
        s = chart_stats(charts[version][image])
        return s["renumbered"] + s["false_boxes"]

    better = sum(corrections(candidate, img) < corrections(baseline, img) for img in shared)
    worse = sum(corrections(candidate, img) > corrections(baseline, img) for img in shared)
    print(
        f"Candidate needed fewer corrections on {better}, more on {worse}, "
        f"the same on {len(shared) - better - worse} X-ray(s)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=CORRECTIONS_LOG, help="Corrections log (corrections.jsonl)")
    parser.add_argument("--baseline", help="Version id to compare against")
    parser.add_argument("--candidate", help="Version id expected to improve on --baseline")
    args = parser.parse_args()
    if bool(args.baseline) != bool(args.candidate):
        parser.error("--baseline and --candidate must be given together")

    charts = load_latest_per_version(args.log)
    if not charts:
        sys.exit(f"No dentist corrections found in {args.log}")

    # Oldest version first, by when its first chart was saved.
    order = sorted(charts, key=lambda v: min(r["saved_at"] for r in charts[v].values()))
    _print_table([(v, summarize(list(charts[v].values()))) for v in order])
    if args.baseline:
        compare_pair(charts, args.baseline, args.candidate)


if __name__ == "__main__":
    main()
