from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException

from backend.dependencies import get_knowledge_service
from backend.schemas import SpeciesResponse
from backend.services.knowledge_service import KnowledgeService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/species/{species_id}",
    response_model=SpeciesResponse,
    summary="Get structured bird species details from the field guide",
)
async def get_species(
    species_id: str,
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
) -> SpeciesResponse:
    """Retrieve structured field-guide information for a given bird species.

    Args:
        species_id: The canonical species folder name, e.g. '017.Cardinal'.
        knowledge_service: Injected singleton knowledge database service.

    Returns:
        Structured species data including taxonomy, habitat, behaviour, and interesting facts.

    Raises:
        HTTPException: 404 if the species ID is invalid or cannot be found.
    """
    logger.info("Received species request: species_id=%s", species_id)
    species_data = knowledge_service.get_species(species_id)
    if species_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Species with ID '{species_id}' not found in the field guide database.",
        )
    return species_data
