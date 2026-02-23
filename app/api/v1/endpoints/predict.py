from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_current_settings, get_inference_service
from app.core.config import Settings
from app.services.inference import InferenceService

router = APIRouter()


class PredictResponse(BaseModel):
    method: Literal["padim", "student"]
    category: str
    filename: str
    image_width: int
    image_height: int
    score_map_width: int
    score_map_height: int

    raw_image_score: float = Field(..., ge=0.0)
    score_z: float
    calibrated_score: float = Field(..., ge=0.0, le=1.0)

    threshold_raw: float = Field(..., ge=0.0)
    threshold_calibrated: float = Field(..., ge=0.0, le=1.0)
    threshold_quantile: float = Field(..., ge=0.0, le=1.0)
    is_anomalous: bool

    model_meta: dict[str, Any]
    timings_ms: dict[str, float]

    visual_content_type: str | None = None
    original_image_base64: str | None = None
    heatmap_image_base64: str | None = None
    overlay_image_base64: str | None = None

    note: str | None = None


@router.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    method: Literal["padim", "student"] = Form("padim"),
    category: str = Form(...),
    return_visuals: bool | None = Form(None),
    service: InferenceService = Depends(get_inference_service),
    settings: Settings = Depends(get_current_settings),
) -> PredictResponse:
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise ValueError("Uploaded file must be an image")

    raw = await image.read()
    if not raw:
        raise ValueError("Uploaded image is empty")

    if len(raw) > settings.request_max_bytes:
        raise ValueError(f"Uploaded image exceeds max size ({settings.request_max_bytes} bytes)")

    effective_return_visuals = (
        settings.default_return_visuals if return_visuals is None else bool(return_visuals)
    )

    result = await run_in_threadpool(
        service.predict,
        raw,
        method,
        category,
        effective_return_visuals,
    )

    return PredictResponse(
        **result,
        filename=image.filename or "upload",
        note=None
        if effective_return_visuals
        else "Set return_visuals=true to include heatmap/overlay images.",
    )
