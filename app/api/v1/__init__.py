from fastapi import APIRouter

from app.api.v1.endpoints.models import router as models_router
from app.api.v1.endpoints.predict import router as predict_router

router = APIRouter()
router.include_router(models_router, tags=["Model Registry"])
router.include_router(predict_router, tags=["Inference"])
