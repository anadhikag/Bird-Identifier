"""Pydantic v2 request and response schemas for the backend API.

This module is responsible for ONE thing: defining the API's wire
format. It contains no business logic, no ML logic, and no I/O — only
data shape and validation, shared by the route handlers.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    """A single ranked species prediction."""

    species_id: str = Field(..., description="Canonical species ID (folder name).")
    common_name: str = Field(..., description="Species common name.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Softmax confidence score in [0, 1]."
    )


class PredictResponse(BaseModel):
    """Response body for POST /predict."""

    species_id: str = Field(..., description="Canonical species ID of top prediction.")
    common_name: str = Field(..., description="Common name of top prediction.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Top-1 confidence score.")
    top_predictions: list[PredictionItem] = Field(
        ..., description="Top-k predictions, ordered by descending confidence."
    )
    gradcam_image: str = Field(
        ..., description="Grad-CAM overlay image, PNG-encoded and base64-serialized."
    )


class IdentificationInfo(BaseModel):
    length: str = Field(..., description="Average length.")
    wingspan: str = Field(..., description="Average wingspan.")
    weight: str = Field(..., description="Average weight.")
    differences: str = Field(..., description="Male vs Female differences.")
    features: str = Field(..., description="Distinctive features.")


class MigrationInfo(BaseModel):
    status: str = Field(..., description="Migratory status.")
    pattern: str = Field(..., description="Migration pattern.")
    raw: str = Field(..., description="Raw markdown content.")


class ConservationInfo(BaseModel):
    status: str = Field(..., description="IUCN Conservation status.")
    threats: str = Field(..., description="Threats.")
    raw: str = Field(..., description="Raw markdown content.")


class SpeciesResponse(BaseModel):
    """Response body for GET /species/{species_id}."""

    common_name: str = Field(..., description="Common name.")
    scientific_name: str = Field(..., description="Scientific name.")
    family: str = Field(..., description="Family taxonomy.")
    order: str = Field(..., description="Order taxonomy.")
    identification: IdentificationInfo
    habitat: str = Field(..., description="Habitat description.")
    geographic_distribution: str = Field(..., description="Geographic distribution.")
    migration: MigrationInfo
    diet: str = Field(..., description="Diet information.")
    behaviour: str = Field(..., description="Behavioral traits.")
    vocalization: str = Field(..., description="Vocalizations details.")
    breeding: str = Field(..., description="Breeding details.")
    conservation: ConservationInfo
    ecological_importance: str = Field(..., description="Ecological importance details.")
    interesting_facts: list[str] = Field(..., description="List of interesting facts.")



class ChatRequest(BaseModel):
    """Request body for POST /chat.

    ``species_id`` must be the canonical knowledge-base folder identifier
    returned by POST /predict (e.g. ``"017.Cardinal"``), NOT the human-readable
    display name.  The RAG retriever matches against folder names stored in the
    FAISS index metadata, so passing the folder name is the only way to ensure
    correct context retrieval for every species.
    """

    species_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Canonical species folder ID, e.g. '017.Cardinal'. "
            "Must match the species_id returned by POST /predict."
        ),
    )
    question: str = Field(..., min_length=1, description="The user's natural-language question.")


class ChatResponse(BaseModel):
    """Response body for POST /chat."""

    answer: str = Field(..., description="Answer grounded in the retrieved knowledge base context.")


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = Field(..., description="Liveness status, e.g. 'ok'.")


class ErrorResponse(BaseModel):
    """Standard error response body used by centralized exception handlers."""

    detail: str = Field(..., description="Human-readable error description.")
