from fastapi import APIRouter

from app.api.v1.endpoints.predict import router as predict_router

router = APIRouter()
router.include_router(predict_router, tags=["Inference"])
