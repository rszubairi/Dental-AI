import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from compare_model_versions import UNVERSIONED, load_latest_per_version, summarize
from numbering_corrections import NOT_A_TOOTH


def _record(image: str, version: str | None, teeth: list[tuple[str, str]], saved_at: str = "2026-11-01") -> dict:
    record = {"image": image, "saved_at": saved_at, "teeth": [{"model_fdi": m, "fdi": f} for m, f in teeth]}
    if version:
        record["model_version"] = version
    return record


def test_rates_count_renumbered_and_false_boxes():
    summary = summarize(
        [
            _record("a.png", "v1", [("24", "25"), ("26", "26"), ("28", NOT_A_TOOTH), ("11", "11")]),
            _record("b.png", "v1", [("11", "11"), ("12", "12"), ("13", "13"), ("14", "14")]),
        ]
    )
    assert summary["charts"] == 2
    assert summary["renumber_rate"] == 1 / 8
    assert summary["false_box_rate"] == 1 / 8
    assert summary["accepted_as_is"] == 0.5


def test_rereview_under_new_version_keeps_both(tmp_path):
    log = tmp_path / "corrections.jsonl"
    records = [
        _record("a.png", None, [("24", "25")]),
        _record("a.png", "v1", [("24", "25")]),
        _record("a.png", "v1", [("24", "24")]),  # later save of the same chart wins
        _record("a.png", "v2", [("25", "25")]),
    ]
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    charts = load_latest_per_version(log)
    assert set(charts) == {UNVERSIONED, "v1", "v2"}
    assert charts["v1"]["a.png"]["teeth"] == [{"model_fdi": "24", "fdi": "24"}]
