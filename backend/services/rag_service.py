"""RAG chat service: application-layer facade over the existing BirdChat class.

This module is responsible for ONE thing: adapting `BirdChat` (complete
and unmodified) into a call the API layer can depend on. It contains NO
retrieval or LLM logic of its own — it only delegates to `BirdChat`.

Intended to be constructed exactly once per process (singleton) and
reused across all requests; see backend/app.py's lifespan handler.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.rag.chat import BirdChat

logger = logging.getLogger(__name__)


class RAGService:
    """Application-layer facade over BirdChat.

    Exists as a thin, separately-testable layer between the API routes
    and BirdChat, so route handlers depend on this service rather than
    importing `src.rag` directly.
    """

    def __init__(self, chat: Optional[BirdChat] = None) -> None:
        """Load the RAG chat pipeline.

        Args:
            chat: An already-constructed BirdChat instance, primarily
                for dependency injection in tests. If None, a new
                BirdChat is constructed using its default configuration.

        Raises:
            FileNotFoundError: If the vector store has not been built
                yet.
            ValueError: If GROQ_API_KEY is not set in the environment.
        """
        logger.info("Loading RAGService.")
        self._chat = chat or BirdChat()
        logger.info("RAGService ready.")

    def ask(self, species: str, question: str) -> str:
        """Answer a question about an already-identified bird species.

        Args:
            species: Common name of the species identified upstream by
                the CNN classifier.
            question: The user's natural-language question.

        Returns:
            The grounded, plain-text answer.

        Raises:
            ValueError: If species or question is empty or blank.
            RuntimeError: If the underlying Groq API call fails.
        """
        return self._chat.ask(species=species, question=question)
