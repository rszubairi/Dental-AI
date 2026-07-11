"""
Isolated AI inference microservice.

Never exposed to end users directly. Convex Actions are the only caller:
Next.js -> Convex Mutation -> Inference Queue -> this service -> Convex -> Next.js
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference import ToothDetectionPipeline

app = FastAPI(title="Dental AI Inference Service", version="0.1.0")

_pipeline = ToothDetectionPipeline()


class InferenceRequest(BaseModel):
    job_id: str
    case_id: str
    image_url: str
    model: str = "tooth_detection"


class DetectionResult(BaseModel):
    fdi_number: str
    bbox: list[float]
    confidence: float


class InferenceResponse(BaseModel):
    job_id: str
    case_id: str
    model: str
    detections: list[DetectionResult]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/infer", response_model=InferenceResponse)
def infer(req: InferenceRequest) -> InferenceResponse:
    if req.model != "tooth_detection":
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")

    detections = _pipeline.run(req.image_url)
    return InferenceResponse(
        job_id=req.job_id,
        case_id=req.case_id,
        model=req.model,
        detections=[DetectionResult(**d) for d in detections],
    )
