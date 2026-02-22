from fastapi import APIRouter
from fastapi.responses import Response

from app.core.metrics import get_metrics

router = APIRouter()


@router.get("/")
async def prometheus_metrics() -> Response:
    """
    Prometheus metrics endpoint.

    Returns application metrics in Prometheus text format.
    """
    metrics_data, media_type = get_metrics()
    return Response(content=metrics_data, media_type=media_type)
