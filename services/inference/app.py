"""
Isolated AI inference microservice.

Never exposed to end users directly. Convex Actions are the only caller:
Next.js -> Convex Mutation -> Inference Queue -> this service -> Convex -> Next.js
"""

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from inference import LandmarkRegressionPipeline, PathologyClassificationPipeline, ToothDetectionPipeline

app = FastAPI(title="Dental AI Inference Service", version="0.1.0")

_tooth_pipeline = ToothDetectionPipeline()
_pathology_pipeline = PathologyClassificationPipeline()
_landmark_pipeline = LandmarkRegressionPipeline()

SUPPORTED_MODELS = {"tooth_detection", "full_assessment", "landmarks"}


class InferenceRequest(BaseModel):
    job_id: str
    case_id: str
    image_url: str
    model: str = "tooth_detection"


class DetectionResult(BaseModel):
    fdi_number: str
    bbox: list[float]
    confidence: float
    pathology: str | None = None
    pathology_confidence: float | None = None


class InferenceResponse(BaseModel):
    job_id: str
    case_id: str
    model: str
    detections: list[DetectionResult]
    missing_teeth: list[str]
    landmarks: dict[str, list[list[float]]] | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _run_model_on_url(model: str, image_url: str) -> tuple[list[dict], list[str], dict | None]:
    landmarks = None
    if model == "tooth_detection":
        result = _tooth_pipeline.run(image_url)
        detections = result["detections"]
        missing_teeth = result["missing_teeth"]
    elif model == "landmarks":
        detections = []
        missing_teeth = []
        landmarks = _landmark_pipeline.run(image_url)
    else:  # full_assessment: tooth detection + pathology classification + landmarks
        raw_detections, missing_teeth, image = _tooth_pipeline.run_with_image(image_url)
        detections = _pathology_pipeline.classify_crops(image, raw_detections)
        landmarks = _landmark_pipeline.run(image_url)
    return detections, missing_teeth, landmarks


def _run_model_on_bytes(model: str, raw_bytes: bytes, filename: str) -> tuple[list[dict], list[str], dict | None]:
    landmarks = None
    if model == "tooth_detection":
        detections, missing_teeth, _image = _tooth_pipeline.run_with_image_bytes(raw_bytes, filename)
    elif model == "landmarks":
        detections = []
        missing_teeth = []
        landmarks = _landmark_pipeline.run_with_image_bytes(raw_bytes, filename)
    else:  # full_assessment
        raw_detections, missing_teeth, image = _tooth_pipeline.run_with_image_bytes(raw_bytes, filename)
        detections = _pathology_pipeline.classify_crops(image, raw_detections)
        landmarks = _landmark_pipeline.run_with_image_bytes(raw_bytes, filename)
    return detections, missing_teeth, landmarks


@app.post("/v1/infer", response_model=InferenceResponse)
def infer(req: InferenceRequest) -> InferenceResponse:
    if req.model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")

    detections, missing_teeth, landmarks = _run_model_on_url(req.model, req.image_url)

    return InferenceResponse(
        job_id=req.job_id,
        case_id=req.case_id,
        model=req.model,
        detections=[DetectionResult(**d) for d in detections],
        missing_teeth=missing_teeth,
        landmarks=landmarks,
    )


@app.post("/v1/infer-upload", response_model=InferenceResponse)
async def infer_upload(
    file: UploadFile = File(...),
    model: str = Form("full_assessment"),
    job_id: str = Form("upload"),
    case_id: str = Form("upload"),
) -> InferenceResponse:
    """Same as /v1/infer, but takes a raw file upload instead of an image_url --
    for standalone testing (e.g. the web app's upload-and-analyze page) where
    there's no hosted URL for the image."""
    if model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    raw_bytes = await file.read()
    detections, missing_teeth, landmarks = _run_model_on_bytes(model, raw_bytes, file.filename or "upload.png")

    return InferenceResponse(
        job_id=job_id,
        case_id=case_id,
        model=model,
        detections=[DetectionResult(**d) for d in detections],
        missing_teeth=missing_teeth,
        landmarks=landmarks,
    )
