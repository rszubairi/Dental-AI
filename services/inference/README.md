---
title: Dental AI Inference
emoji: 🦷
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
license: cc-by-nc-sa-4.0
---

# Dental AI Inference Service

FastAPI service serving three models trained on the [DENTEX dataset](https://huggingface.co/datasets/ibrahimhamamci/DENTEX)
(CC BY-NC-SA 4.0) plus client-provided landmark annotations:

- **Stage 1** — YOLOv8 tooth detection + FDI numbering
- **Stage 2A** — EfficientNet-B3 per-tooth pathology classification (healthy / caries / deep_caries / periapical_lesion / impacted)
- **Stage 2B** — U-Net heatmap regression for bone_crest / sinus_floor / nerve_canal landmark curves

## Endpoints

- `GET /health`
- `POST /v1/infer` — `{"job_id", "case_id", "image_url", "model"}` where `model` is
  `tooth_detection`, `landmarks`, or `full_assessment`.
- `POST /v1/infer-upload` — same, but takes a multipart file upload (`file`, `model`)
  instead of a hosted `image_url`.

## Configuration

Set `HF_CHECKPOINTS_REPO` (Space secret) to the HF model repo id holding the trained
checkpoints (see `download_checkpoints.py`); they're downloaded once at container
startup rather than baked into this image.

## Model versioning

Every response carries `model_version` (e.g. `2026.11.0+3f9a1c2b`) and its
`model_version_components`: a content hash of each stage's weights, of the
inference/post-processing code, and the thresholds in effect (see `model_version.py`).
Same id means same weights, code and settings. `GET /v1/model-version` shows what is
currently deployed.

To release: tag the HF checkpoints repo (e.g. `2026.11.0`) and set the Space variable
`HF_CHECKPOINTS_REVISION` to that tag. It becomes the release label (override with
`MODEL_RELEASE`). Store `model_version` with every report and with dentist feedback
on it, then compare versions with:

    python training/compare_model_versions.py --baseline <old id> --candidate <new id>

## Inference run log (Streamlit)

Every run in `streamlit_app.py` is logged to Convex (`inference_runs.py` ->
`apps/web/convex/inferenceRuns.ts`): the X-ray (stored once, by sha256), model
version and components, every checkpoint path and threshold, duration, and the
outputs or error. The app's **Run log** view lists runs by model version.

Setup: pick a long random secret, then

    cd apps/web && npx convex env set INFERENCE_LOG_SECRET <secret>

and give the viewer `CONVEX_SITE_URL` (the deployment's `https://<name>.convex.site`
URL) and `INFERENCE_LOG_SECRET`, as env vars or in `.streamlit/secrets.toml`.
Without them, inference still works but runs aren't logged.

## License

Non-commercial (CC BY-NC-SA 4.0), inherited from the DENTEX dataset's license terms.
