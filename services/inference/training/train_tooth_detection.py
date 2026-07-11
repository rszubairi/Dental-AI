"""Train the tooth detection + FDI numbering model.

Requires annotated data in datasets/tooth_detection/ (see README.md there for format).
Usage:
    python train_tooth_detection.py --data ../../../datasets/tooth_detection/dataset.yaml --epochs 150
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to YOLO dataset.yaml")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--project", default="../models/tooth_detection/runs")
    args = parser.parse_args()

    model = YOLO(args.base_model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name="tooth_detection",
    )


if __name__ == "__main__":
    main()
