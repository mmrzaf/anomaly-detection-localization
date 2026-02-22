from fastapi import APIRouter

from .exception_handlers import exception_handlers
from .health import router as health_router
from .metrics import router as metrics_router
from .v1 import router as v1_router

router = APIRouter()

router.include_router(health_router, prefix="/health", tags=["Health & Monitoring"])

router.include_router(metrics_router, prefix="/metrics", tags=["Metrics & Stats"])
router.include_router(v1_router, prefix="/v1")

__all__ = ["router", "exception_handlers"]
