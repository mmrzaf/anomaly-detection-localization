from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import get_inference_service
from app.services.inference import InferenceService

router = APIRouter()


class ModelEntry(BaseModel):
    method: Literal["padim", "student"]
    category: str
    artifact_version: str | None = None
    schema_version: str
    backbone: str
    layers: list[str]
    target_hw: list[int]
    threshold_raw: float
    threshold_quantile: float
    created_at: str | None = None


class ModelsResponse(BaseModel):
    count: int
    categories: list[str]
    models: list[ModelEntry]


@router.get("/models", response_model=ModelsResponse)
async def list_models(
    service: InferenceService = Depends(get_inference_service),
) -> ModelsResponse:
    models = service.list_models()
    categories = sorted({m["category"] for m in models})
    return ModelsResponse(count=len(models), categories=categories, models=models)  # type: ignore[arg-type]
