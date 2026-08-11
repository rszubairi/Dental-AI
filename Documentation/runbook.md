End-to-End Execution Runbook
0. Setup (one-time)

cd services/inference
pip install -r requirements.txt
This installs ultralytics, torch, torchvision, opencv-python-headless, etc. Confirm GPU is visible if you have one:


python3 -c "import torch; print(torch.cuda.is_available())"
1. Prepare Stage 1 data (tooth detection, 32 FDI classes)
Converts DENTEX's quadrant_enumeration COCO subset into a CLAHE + letterboxed 640×640 YOLO dataset.


cd datasets/dentex
python3 prepare_stage1_dataset.py --dentex-root . --out yolo
Produces datasets/dentex/yolo/{images,labels}/{train,val,test}/ + dataset.yaml (507/63/64 split).

2. Prepare the quadrant pretraining data (backbone warm-start)
Converts the coarse 4-class quadrant subset (693 images) — used only to pretrain the backbone before Stage 1's real fine-tune.


python3 prepare_quadrant_pretrain_dataset.py --dentex-root . --out yolo_quadrant_pretrain
Produces datasets/dentex/yolo_quadrant_pretrain/dataset.yaml (624/69 split).

3. Pretrain the backbone on quadrant localization

cd ../../services/inference/training
python3 train_quadrant_pretrain.py --data ../../../datasets/dentex/yolo_quadrant_pretrain/dataset.yaml
Outputs runs/pretrain_quadrant/train/weights/best.pt
Auto-skips if already run (checks .dental_ai_state.json); pass --reset-step to force a redo.
4. Train Stage 1: tooth detection + FDI numbering

python3 train_tooth_detection.py \
  --data ../../../datasets/dentex/yolo/dataset.yaml \
  --base-model ../../../services/inference/training/runs/pretrain_quadrant/train/weights/best.pt
(Omit --base-model to fall back to stock yolov8s.pt COCO weights instead of the quadrant-pretrained backbone — still valid, just skips step 3's benefit.)

100 epochs, patience 20, AdamW, cosine LR, FP16, doc-matching augmentation.
Outputs runs/stage1/train/weights/best.pt (+ last.pt, results.csv, confusion_matrix.png from Ultralytics).
Target: mAP@0.5 ≥ 0.70. Check runs/stage1/train/results.csv or console output.
Same auto-skip/--reset-step behavior.
5. Prepare Stage 2A data (pathology classifier crops)

cd ../../../datasets/dentex
python3 prepare_stage2a_dataset.py --dentex-root . --out stage2a
Produces datasets/dentex/stage2a/{train,val}/{healthy,caries,deep_caries,periapical_lesion,impacted}/*.png (2858/698 crops, image-level split).

6. Train Stage 2A: EfficientNet-B3 pathology classifier

cd ../../services/inference/training
python3 train_pathology_classifier.py --data ../../../datasets/dentex/stage2a
50 epochs, patience 15, AdamW, class-weighted CrossEntropyLoss.
Outputs runs/stage2a/best.pt, last.pt, classes.json.
Target: macro AUC-ROC ≥ 0.85.
7. Stage 2B — blocked until you have client data
Collect ≥200 annotated OPGs per datasets/landmarks/README.md's format (images/*.png + annotations/*.json with bone_crest/sinus_floor/nerve_canal normalized point lists).

No auto-labeling shortcut exists for this stage (unlike Stage 1) — no model has ever been trained to predict these landmarks, so there's nothing to run inference with yet. First batch must be annotated manually in Label Studio, either with KeyPointLabels (points) or PolygonLabels (outlines) for "bone_crest", "sinus_floor", "nerve_canal" — both are supported by the converter. Export the project as JSON (native Label Studio export, not YOLO), then convert into this project's layout:

cd datasets/landmarks
python import_label_studio_export.py --export path/to/project-export.json
Writes images/<stem>.png + annotations/<stem>.json. For keypoint annotations, points are used directly. For polygon annotations, each polygon is rasterized and skeletonized (scikit-image) to extract an ordered centerline through the structure — this matters because a polygon traces a structure's boundary/width, not its path, so using raw polygon vertices as landmark points would produce a ring around the structure instead of points tracing through it.
Any other label found in the export (e.g. a pathology taxonomy like "interproximal_mild/moderate/severe" that doesn't match any Stage 2A class) is written to unmatched_labels.json for review rather than silently dropped or merged in.
Set LABEL_STUDIO_URL / LABEL_STUDIO_ACCESS_TOKEN env vars if images are stored in Label Studio itself (local/uploaded storage) rather than external URLs — Label Studio prefixes uploaded filenames with a hash (e.g. "7592fbd2-train_0.png"), which the converter strips to recover the original stem.

Once available:


python3 train_landmark_regression.py --data ../../../datasets/landmarks
Running it today will print a "BLOCKED: found 0 annotated images..." message and exit — that's expected, not a bug.

8. Deploy trained checkpoints for inference
The inference service reads from separate paths (not runs/), controlled by env vars:


export TOOTH_DETECTION_CHECKPOINT=services/inference/training/runs/stage1/train/weights/best.pt
export PATHOLOGY_CLASSIFIER_CHECKPOINT=services/inference/training/runs/stage2a/best.pt
export PATHOLOGY_CLASSIFIER_CLASSES=services/inference/training/runs/stage2a/classes.json
Then start the service:


cd services/inference
uvicorn app:app --reload
Call it:


curl -X POST localhost:8000/v1/infer -H "Content-Type: application/json" -d '{
  "job_id": "j1", "case_id": "c1",
  "image_url": "https://.../opg.png",
  "model": "full_assessment"
}'
"model": "tooth_detection" runs Stage 1 only; "full_assessment" runs Stage 1 → Stage 2A composed, returning per-tooth fdi_number, bbox, pathology, pathology_confidence, plus missing_teeth.

Notes
Steps 3–4 and 5–6 are independent of each other — you can run the 2A pipeline (5–6) in parallel with or before Stage 1 (3–4) if you want.
--reset-step on any training script wipes that stage's runs/... output and its .dental_ai_state.json entry, forcing a clean retrain.
All of steps 1–6 have been verified end-to-end against your real local DENTEX data in this session (small-scale/smoke-tested for training loops due to no full epoch budget here) — full multi-epoch runs are yours to execute when ready, ideally on a GPU box.

git -C apps/web checkout main
git -C apps/web branch --show-current

label-studio

9. Auto-labeling in Label Studio via the Stage 1 model
Runs a trained checkpoint as a Label Studio ML backend so new images get pre-annotated boxes to review/correct instead of labeling from scratch.

cd services/inference
TOOTH_DETECTION_CHECKPOINT=../../runs/detect/runs/stage1/train/weights/best.pt \
    uvicorn label_studio_ml_backend:app --host 0.0.0.0 --port 9090
In Label Studio: Settings -> Model -> Connect Model -> URL http://localhost:9090 (or the LAN/host address if Label Studio runs elsewhere, e.g. Docker).
Label Studio's rectangle labels in your labeling config must be named as plain FDI tooth numbers ("11".."48") to match the model's class names.
If Label Studio stores images itself (local/uploaded storage, not external URLs), set LABEL_STUDIO_URL and LABEL_STUDIO_ACCESS_TOKEN env vars so the backend can fetch them with auth.
Tune sensitivity with LS_CONFIDENCE_THRESHOLD (default 0.25) — lower surfaces more (noisier) boxes to correct, higher surfaces fewer but more confident ones.
Tested locally against a real dataset image (datasets/Dentex/yolo/images/test/train_101.png) via curl — predictions returned correct percentage-based boxes and FDI labels.
After labeling more data with this assist, re-run datasets/Dentex/import_label_studio_export.py to fold the new labels into the training set.