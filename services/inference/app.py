"""
Isolated AI inference microservice.

Never exposed to end users directly. Convex Actions are the only caller:
Next.js -> Convex Mutation -> Inference Queue -> this service -> Convex -> Next.js
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference import PathologyClassificationPipeline, ToothDetectionPipeline

app = FastAPI(title="Dental AI Inference Service", version="0.1.0")

_tooth_pipeline = ToothDetectionPipeline()
_pathology_pipeline = PathologyClassificationPipeline()

SUPPORTED_MODELS = {"tooth_detection", "full_assessment"}


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/infer", response_model=InferenceResponse)
def infer(req: InferenceRequest) -> InferenceResponse:
    if req.model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")

    if req.model == "tooth_detection":
        result = _tooth_pipeline.run(req.image_url)
        detections = result["detections"]
        missing_teeth = result["missing_teeth"]
    else:  # full_assessment: tooth detection + per-tooth pathology classification
        raw_detections, missing_teeth, image = _tooth_pipeline.run_with_image(req.image_url)
        detections = _pathology_pipeline.classify_crops(image, raw_detections)

    return InferenceResponse(
        job_id=req.job_id,
        case_id=req.case_id,
        model=req.model,
        detections=[DetectionResult(**d) for d in detections],
        missing_teeth=missing_teeth,
    )
