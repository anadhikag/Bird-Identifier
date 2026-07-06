"""Dependency-injection providers for FastAPI routes.

This module is responsible for ONE thing: exposing FastAPI dependency
functions that retrieve already-constructed singleton services from
application state. Services are loaded exactly once, at startup, by
`backend.app`'s lifespan handler — these functions never construct a
service themselves, they only look one up (and fail loudly with a
clean HTTP error if it isn't available).
"""

from __future__ import annotations

from typing import Optional

from fastapi import HTTPException, Request

from backend.services.classifier_service import ClassifierService
from backend.services.rag_service import RAGService
from backend.services.knowledge_service import KnowledgeService


def get_classifier_service(request: Request) -> ClassifierService:
    """Retrieve the singleton ClassifierService from application state.

    Args:
        request: Incoming request, used to access `request.app.state`.

    Returns:
        The application-wide ClassifierService instance.

    Raises:
        HTTPException: 503 if the classifier service failed to
            initialize at startup.
    """
    service: Optional[ClassifierService] = getattr(request.app.state, "classifier_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Classifier service is not available.")
    return service


def get_rag_service(request: Request) -> RAGService:
    """Retrieve the singleton RAGService from application state.

    Args:
        request: Incoming request, used to access `request.app.state`.

    Returns:
        The application-wide RAGService instance.

    Raises:
        HTTPException: 503 if the chat service failed to initialize at
            startup (e.g. missing GROQ_API_KEY or a vector store that
            has not been built yet).
    """
    service: Optional[RAGService] = getattr(request.app.state, "rag_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Chat service is not available.")
    return service


def get_knowledge_service(request: Request) -> KnowledgeService:
    """Retrieve the singleton KnowledgeService from application state.

    Args:
        request: Incoming request, used to access `request.app.state`.

    Returns:
        The application-wide KnowledgeService instance.

    Raises:
        HTTPException: 503 if the knowledge service failed to initialize at
            startup.
    """
    service: Optional[KnowledgeService] = getattr(request.app.state, "knowledge_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Knowledge service is not available.")
    return service

