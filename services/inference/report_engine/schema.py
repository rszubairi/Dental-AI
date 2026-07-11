"""Shared response schema returned to Convex after an inference job completes."""

from pydantic import BaseModel


class ToothFinding(BaseModel):
    fdi_number: str
    bbox: list[float]
    confidence: float


class InferenceReport(BaseModel):
    job_id: str
    case_id: str
    model: str
    model_version: str
    findings: list[ToothFinding]
