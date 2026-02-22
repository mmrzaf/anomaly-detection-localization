import base64
from typing import Literal

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field

router = APIRouter()


class PredictResponse(BaseModel):
    method: str
    filename: str
    content_type: str
    image_score: float = Field(..., ge=0.0)
    is_anomalous: bool
    threshold: float = 0.5
    original_image_base64: str
    heatmap_image_base64: str
    overlay_image_base64: str
    note: str


@router.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    method: Literal["padim", "student"] = Form("padim"),
) -> PredictResponse:
    """
    Minimal demo inference endpoint.

    This is a stub that echoes the uploaded image into all preview slots.
    Replace the score + generated heatmap/overlay later with real model output.
    """
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise ValueError("Uploaded file must be an image")

    raw = await image.read()
    if not raw:
        raise ValueError("Uploaded image is empty")

    img_b64 = base64.b64encode(raw).decode("utf-8")

    # Stub score (replace with real inference)
    image_score = 0.12 if method == "padim" else 0.09
    threshold = 0.50

    return PredictResponse(
        method=method,
        filename=image.filename or "upload",
        content_type=image.content_type,
        image_score=image_score,
        is_anomalous=image_score >= threshold,
        threshold=threshold,
        original_image_base64=img_b64,
        heatmap_image_base64=img_b64,  # placeholder
        overlay_image_base64=img_b64,  # placeholder
        note="Stub response: heatmap/overlay currently mirror the uploaded image.",
    )
