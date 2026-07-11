# Tooth Detection + FDI Numbering — Dataset Requirements

**🚩 ANNOTATED DATA NEEDED — this is the blocker for training the first model.**

Nothing here trains without this data. Everything else in the repo (inference
service, training script, FDI mapping) is already scaffolded and waiting on it.

## What to collect

- **Image type:** Panoramic (OPG) X-rays preferred for whole-arch tooth
  detection. Periapical/bitewing films can be added later as a second dataset.
- **Format:** JPEG/PNG or DICOM (`.dcm`). Both are supported by the loader.
- **Volume:** Realistic starting point is 300–500 images minimum for a usable
  first model; more (1000+) is strongly preferred for production-grade
  accuracy. Fewer than ~150 will likely produce an unreliable model but is
  enough to validate the pipeline end-to-end.
- **De-identification:** Strip patient names/DOB/MRN from DICOM headers and
  filenames before sharing. This is a legal/compliance requirement, not
  optional — do this before the folder is handed over.

## Annotation format (YOLO)

For every image, one bounding box per visible tooth, labeled with its FDI
number (11–48, adult dentition only for v1 — see
`services/inference/postprocessing/fdi_mapping.py`).

```
datasets/tooth_detection/
    raw_images/
        case_0001.jpg
        case_0002.jpg
        ...
    annotations/
        case_0001.txt      # YOLO format: "<class_index> <x_center> <y_center> <width> <height>", normalized 0-1
        case_0002.txt
        ...
    splits/
        train.txt           # list of image filenames
        val.txt
        test.txt
    dataset.yaml            # generated once folders are populated — see below
```

`class_index` maps to FDI number via the fixed order in
`fdi_mapping.FDI_TOOTH_NUMBERS` (index 0 = "11", index 1 = "12", ... index 31 = "48").

If your annotation tool exports COCO JSON, Label Studio JSON, or Pascal VOC
XML instead of raw YOLO txt, that's fine — say which tool you used and I'll
write a converter rather than asking you to re-annotate.

## Recommended annotation tools

- [Label Studio](https://labelstud.io/) — free, self-hostable, exports to
  YOLO/COCO directly.
- [CVAT](https://www.cvat.ai/) — free, good for bounding boxes at volume.
- Roboflow — free tier, handles export format conversion for you.

## What I need from you

1. Drop de-identified X-ray images into `raw_images/`.
2. Either annotate them yourself in one of the tools above and export to this
   folder, or hand me the raw images + a note on which teeth are missing/
   present per image and I can advise on getting annotation help (this
   project cannot auto-generate ground-truth annotations — a model can't
   supervise itself on data it hasn't seen).
3. Confirm rough dataset size so I can set realistic train/val/test splits
   (default plan: 80/10/10).

Once ~150+ annotated images are in place, run:

```
python services/inference/training/train_tooth_detection.py --data datasets/tooth_detection/dataset.yaml
```
