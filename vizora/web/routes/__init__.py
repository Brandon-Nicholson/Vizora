# Routes package
from .health import router as health_router
from .analysis import router as analysis_router
from .auth import router as auth_router
from .billing import router as billing_router
from .schedules import router as schedules_router
from .google import router as google_router

__all__ = [
    "health_router",
    "analysis_router",
    "auth_router",
    "billing_router",
    "schedules_router",
    "google_router",
]
