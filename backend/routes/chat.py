"""Chat endpoint: RAG-grounded question answering for an identified species."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from backend.dependencies import get_rag_service
from backend.schemas import ChatRequest, ChatResponse
from backend.services.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/chat", response_model=ChatResponse, summary="Ask a question about an identified species"
)
async def chat(
    payload: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    """Answer a question about an already-identified bird species.

    Args:
        payload: Request body containing the identified species and
            the user's question.
        rag_service: Injected singleton RAG chat service.

    Returns:
        ChatResponse containing the grounded answer text.

    Raises:
        HTTPException: 503 if the chat service is unavailable (raised
            by `get_rag_service`).
    """
    logger.info("Received chat request: species_id=%s", payload.species_id)
    # Pass the canonical folder ID to the RAG service so the retriever can
    # match it against the FAISS metadata (which also stores folder names).
    # The retriever's _normalize_species_name() strips the numeric prefix
    # ("017.") and converts underscores to spaces, so "017.Cardinal" will
    # correctly match all Cardinal chunks regardless of display name.
    answer = rag_service.ask(species=payload.species_id, question=payload.question)
    return ChatResponse(answer=answer)
