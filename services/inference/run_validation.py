"""Run the full inference pipeline against every image in a folder and log
predictions + summary stats to a file.

No ground-truth labels are used/required -- this is a plausibility/health
check against unseen images (e.g. datasets/Dentex/validation_data, DENTEX's
held-out test set, whose labels were never published), not an accuracy score.

Usage:
    python run_validation.py --images-url http://localhost:8898 --images-dir <path> --out validation_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import httpx

INFER_URL = "http://localhost:8000/v1/infer"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, help="Local folder of images (used to enumerate filenames)")
    parser.add_argument("--images-url", required=True, help="Base URL serving that same folder over HTTP")
    parser.add_argument("--out", default="validation_log.jsonl")
    parser.add_argument("--model", default="full_assessment")
    parser.add_argument("--limit", type=int, default=None, help="Only process N images (sorted order)")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N images (sorted order)")
    parser.add_argument("--append", action="store_true", help="Append to --out instead of overwriting it")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    image_files = sorted(images_dir.glob("*.png"))
    if args.offset:
        image_files = image_files[args.offset :]
    if args.limit:
        image_files = image_files[: args.limit]
    print(f"Found {len(image_files)} images under {images_dir} (offset={args.offset})")

    pathology_counts: Counter[str] = Counter()
    detection_counts: list[int] = []
    missing_counts: list[int] = []
    errors: list[tuple[str, str]] = []

    with open(args.out, "a" if args.append else "w", encoding="utf-8") as log:
        for i, img_path in enumerate(image_files, start=1):
            image_url = f"{args.images_url.rstrip('/')}/{img_path.name}"
            payload = {
                "job_id": f"val-{i}",
                "case_id": img_path.stem,
                "image_url": image_url,
                "model": args.model,
            }
            t0 = time.time()
            try:
                resp = httpx.post(INFER_URL, json=payload, timeout=120.0)
                resp.raise_for_status()
                result = resp.json()
                elapsed = time.time() - t0

                detections = result.get("detections", [])
                missing = result.get("missing_teeth", [])
                landmarks = result.get("landmarks") or {}

                detection_counts.append(len(detections))
                missing_counts.append(len(missing))
                for d in detections:
                    if d.get("pathology"):
                        pathology_counts[d["pathology"]] += 1

                log.write(
                    json.dumps(
                        {
                            "image": img_path.name,
                            "elapsed_s": round(elapsed, 2),
                            "n_detections": len(detections),
                            "n_missing_teeth": len(missing),
                            "landmark_point_counts": {k: len(v) for k, v in landmarks.items()},
                            "detections": detections,
                            "missing_teeth": missing,
                            "landmarks": landmarks,
                        }
                    )
                    + "\n"
                )
                print(f"[{i}/{len(image_files)}] {img_path.name}: {len(detections)} teeth, "
                      f"{len(missing)} missing, {elapsed:.1f}s")
            except Exception as exc:  # noqa: BLE001
                errors.append((img_path.name, str(exc)))
                log.write(json.dumps({"image": img_path.name, "error": str(exc)}) + "\n")
                print(f"[{i}/{len(image_files)}] {img_path.name}: ERROR {exc}")

    print("\n--- Summary ---")
    print(f"Images processed: {len(image_files)}  |  errors: {len(errors)}")
    if detection_counts:
        print(f"Detections per image: min={min(detection_counts)} max={max(detection_counts)} "
              f"avg={sum(detection_counts)/len(detection_counts):.1f}")
    if missing_counts:
        print(f"Missing teeth per image: min={min(missing_counts)} max={max(missing_counts)} "
              f"avg={sum(missing_counts)/len(missing_counts):.1f}")
    print("Pathology class distribution:", dict(pathology_counts))
    if errors:
        print("Errors:")
        for name, err in errors:
            print(f"  {name}: {err}")
    print(f"\nFull per-image log written to {args.out}")


if __name__ == "__main__":
    main()
