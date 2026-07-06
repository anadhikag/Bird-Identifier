"""Health check endpoint for uptime monitoring and Azure App Service probes."""

from __future__ import annotations

from fastapi import APIRouter

from backend.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    """Return a simple liveness status.

    This endpoint intentionally does not depend on the classifier or
    chat services, so it keeps responding even if one of those
    optional components failed to initialize — Azure App Service (and
    any load balancer) can use it purely as a process-liveness check.

    Returns:
        HealthResponse with a fixed "ok" status.
    """
    return HealthResponse(status="ok")
