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

## License

Non-commercial (CC BY-NC-SA 4.0), inherited from the DENTEX dataset's license terms.
