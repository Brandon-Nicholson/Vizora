"""
Health check endpoint for Vizora Web API.
"""

from fastapi import APIRouter

from vizora.web.models.responses import HealthResponse


router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns basic health status for load balancers and monitoring.
    """
    return HealthResponse(status="healthy", version="1.0.0")
