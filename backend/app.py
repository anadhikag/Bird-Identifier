"""FastAPI application entry point for the Bird Identification Assistant backend.

This module is responsible for ONE thing: application wiring — creating
the FastAPI app, loading singleton ML services once at startup,
registering routers, CORS, and centralized exception handling. It
contains no ML logic and no route-level business logic; those live in
`backend/services/*` and `backend/routes/*` respectively.

Run locally:
    uvicorn backend.app:app --reload

Run on Azure App Service (Linux, via a startup command such as):
    gunicorn -k uvicorn.workers.UvicornWorker -w 4 --bind 0.0.0.0:$PORT backend.app:app
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.routes import chat as chat_routes
from backend.routes import health as health_routes
from backend.routes import predict as predict_routes
from backend.routes import species as species_routes
from backend.schemas import ErrorResponse
from backend.services.classifier_service import ClassifierService
from backend.services.rag_service import RAGService
from backend.services.knowledge_service import KnowledgeService

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CHECKPOINT_PATH = _PROJECT_ROOT / "models" / "bird_classifier_best.pt"

# All paths and settings are environment-configurable rather than
# hardcoded, so the same image runs unmodified locally and on Azure
# App Service (where these would be set as Application Settings).
CHECKPOINT_PATH = Path(os.getenv("MODEL_CHECKPOINT_PATH", str(_DEFAULT_CHECKPOINT_PATH)))
MODEL_DEVICE = os.getenv("MODEL_DEVICE") or None
ALLOWED_ORIGINS = [
    origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load singleton ML services once at startup; release them at shutdown.

    `ClassifierService` (BirdInference + GradCAMGenerator) and
    `RAGService` (BirdChat) are expensive to construct — they load a
    model checkpoint and an embedding model respectively. Loading them
    here, once per process, and storing them on `app.state` means every
    request reuses the same instances instead of reloading models.

    A failure to initialize either service is logged but does not
    crash the application: the corresponding `app.state` attribute is
    left as None, and `backend.dependencies` turns that into a clean
    503 response only for the endpoints that actually need it. This
    keeps a misconfigured optional component (e.g. a missing
    GROQ_API_KEY) from taking down the entire API, including /health.
    """
    app.state.classifier_service = None
    app.state.rag_service = None
    app.state.knowledge_service = None

    try:
        app.state.knowledge_service = KnowledgeService()
    except Exception:
        logger.exception(
            "Failed to initialize KnowledgeService; /species will return 503 "
            "until this is resolved."
        )

    try:
        app.state.classifier_service = ClassifierService(
            checkpoint_path=CHECKPOINT_PATH, device=MODEL_DEVICE
        )
    except Exception:
        logger.exception(
            "Failed to initialize ClassifierService; /predict will return 503 "
            "until this is resolved."
        )

    try:
        app.state.rag_service = RAGService()
    except Exception:
        logger.exception(
            "Failed to initialize RAGService; /chat will return 503 until this "
            "is resolved."
        )

    yield

    classifier_service: ClassifierService | None = app.state.classifier_service
    if classifier_service is not None:
        classifier_service.close()

    logger.info("Application shutdown complete.")


def _register_exception_handlers(app: FastAPI) -> None:
    """Attach centralized exception handlers so routes stay free of try/except.

    Route handlers and services raise plain Python exceptions
    (ValueError, RuntimeError, FileNotFoundError); this is the single
    place that maps them to HTTP responses. FastAPI's built-in
    `HTTPException` handling (used by `backend.dependencies` for 503s)
    takes precedence over the catch-all handler below, since it is
    registered for a more specific exception type.

    Args:
        app: The FastAPI application instance to attach handlers to.
    """

    @app.exception_handler(ValueError)
    async def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
        logger.warning("Bad request on %s: %s", request.url.path, exc)
        return JSONResponse(status_code=400, content=ErrorResponse(detail=str(exc)).model_dump())

    @app.exception_handler(RuntimeError)
    async def handle_runtime_error(request: Request, exc: RuntimeError) -> JSONResponse:
        logger.error("Upstream failure on %s: %s", request.url.path, exc)
        return JSONResponse(status_code=502, content=ErrorResponse(detail=str(exc)).model_dump())

    @app.exception_handler(FileNotFoundError)
    async def handle_file_not_found(request: Request, exc: FileNotFoundError) -> JSONResponse:
        logger.error("Missing resource on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(detail="A required server resource is missing.").model_dump(),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(detail="Internal server error.").model_dump(),
        )


def create_app() -> FastAPI:
    """Application factory for the Bird Identification Assistant backend.

    Returns:
        A fully configured FastAPI application instance.
    """
    app = FastAPI(
        title="Bird Identification Assistant API",
        description="Species classification, Grad-CAM explanations, and RAG-grounded chat.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_routes.router, tags=["health"])
    app.include_router(predict_routes.router, tags=["predict"])
    app.include_router(chat_routes.router, tags=["chat"])
    app.include_router(species_routes.router, tags=["species"])

    _register_exception_handlers(app)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
