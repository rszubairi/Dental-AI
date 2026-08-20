"""Convert run_validation.py's output (validation_log.jsonl) into a Label
Studio import JSON, with our pipeline's own detections embedded as
"predictions" -- so the images can be visually reviewed with pathology boxes
already drawn, without running a live ML backend.

Detection bboxes are in the 1024x1024 square canvas the tooth detector runs
on (see preprocessing/image_loader.py's normalize_for_model). Converting to
a percentage of that canvas size (bbox / 1024 * 100) lines up correctly with
the *original* (non-square) image in Label Studio too: resizing each axis
independently to reach 1024 preserves each axis's fraction-of-dimension, so
no aspect-ratio correction is needed here.

Usage:
    python build_label_studio_validation_import.py \
        --log validation_log.jsonl \
        --images-url http://localhost:8898 \
        --out validation_label_studio_import.json
"""

from __future__ import annotations

import argparse
import json
import math

DETECTION_CANVAS_SIZE = 1024

# Points further apart than this (fraction of image size) are treated as
# belonging to different curve segments (e.g. left vs right sinus_floor)
# rather than connected -- otherwise a single nearest-neighbour walk would
# jump across the gap and draw one long line across the whole image.
CURVE_MAX_GAP = 0.12

LABEL_CONFIG = """<View>
  <Image name="image" value="$image" zoom="true"/>
  <RectangleLabels name="pathology" toName="image">
    <Label value="healthy" background="#52C41A"/>
    <Label value="caries" background="#FAAD14"/>
    <Label value="deep_caries" background="#FA541C"/>
    <Label value="periapical_lesion" background="#F5222D"/>
    <Label value="impacted" background="#722ED1"/>
  </RectangleLabels>
  <PolygonLabels name="landmarks" toName="image" opacity="0.9" strokeWidth="2">
    <Label value="bone_crest" background="#FFA39E"/>
    <Label value="sinus_floor" background="#D4380D"/>
    <Label value="nerve_canal" background="#FFC069"/>
  </PolygonLabels>
</View>"""


def order_into_curves(points: list[list[float]], max_gap: float = CURVE_MAX_GAP) -> list[list[list[float]]]:
    """Greedy nearest-neighbour walk that splits into a new curve whenever the
    nearest remaining point is further than max_gap away -- keeps bilateral
    structures (e.g. left/right sinus_floor) as separate segments instead of
    one line jumping across the image."""
    remaining = list(points)
    curves: list[list[list[float]]] = []
    while remaining:
        curve = [remaining.pop(0)]
        while remaining:
            last = curve[-1]
            idx = min(
                range(len(remaining)),
                key=lambda i: math.hypot(last[0] - remaining[i][0], last[1] - remaining[i][1]),
            )
            dist = math.hypot(last[0] - remaining[idx][0], last[1] - remaining[idx][1])
            if dist > max_gap:
                break
            curve.append(remaining.pop(idx))
        curves.append(curve)
    return curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="validation_log.jsonl")
    parser.add_argument("--images-url", required=True, help="Base URL serving the validation images")
    parser.add_argument("--out", default="validation_label_studio_import.json")
    args = parser.parse_args()

    tasks = []
    with open(args.log, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if "error" in entry:
                continue

            image_url = f"{args.images_url.rstrip('/')}/{entry['image']}"
            result = []
            for det in entry.get("detections", []):
                x0, y0, x1, y1 = det["bbox"]
                pathology = det.get("pathology") or "healthy"
                result.append(
                    {
                        "from_name": "pathology",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "score": det.get("pathology_confidence") or det.get("confidence") or 0.0,
                        "value": {
                            "x": x0 / DETECTION_CANVAS_SIZE * 100,
                            "y": y0 / DETECTION_CANVAS_SIZE * 100,
                            "width": (x1 - x0) / DETECTION_CANVAS_SIZE * 100,
                            "height": (y1 - y0) / DETECTION_CANVAS_SIZE * 100,
                            "rotation": 0,
                            "rectanglelabels": [pathology],
                        },
                        "meta": {
                            "text": [
                                f"fdi={det.get('fdi_number')}",
                                f"det_conf={det.get('confidence', 0):.2f}",
                                f"path_conf={det.get('pathology_confidence', 0):.2f}",
                            ]
                        },
                    }
                )

            for structure, points in entry.get("landmarks", {}).items():
                for curve in order_into_curves(points):
                    if len(curve) < 2:
                        continue  # a lone point can't form a line
                    result.append(
                        {
                            "from_name": "landmarks",
                            "to_name": "image",
                            "type": "polygonlabels",
                            "value": {
                                "points": [[x * 100, y * 100] for x, y in curve],
                                "polygonlabels": [structure],
                            },
                        }
                    )

            tasks.append(
                {
                    "data": {
                        "image": image_url,
                        "case_id": entry["image"],
                        "missing_teeth": ", ".join(entry.get("missing_teeth", [])),
                    },
                    "predictions": [
                        {
                            "model_version": "stage1+stage2a-validation",
                            "result": result,
                        }
                    ],
                }
            )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)

    print(f"Wrote {len(tasks)} tasks to {args.out}")
    print("\nLabel Studio project setup:")
    print("1. Create a new project -> Labeling Setup -> Custom template -> paste this config:\n")
    print(LABEL_CONFIG)
    print(f"\n2. Import -> upload {args.out}")
    print(f"3. Keep the image file server running at {args.images_url} while viewing (Label Studio fetches images by URL).")
    print("4. Predictions appear under each task's \"Predictions\" tab for visual review -- not saved as annotations.")


if __name__ == "__main__":
    main()
