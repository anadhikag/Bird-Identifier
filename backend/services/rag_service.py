"""RAG chat service: application-layer facade over the existing BirdChat class.

This module is responsible for ONE thing: adapting `BirdChat` (complete
and unmodified) into a call the API layer can depend on. It contains NO
retrieval or LLM logic of its own — it only delegates to `BirdChat`.

Intended to be constructed exactly once per process (singleton) and
reused across all requests; see backend/app.py's lifespan handler.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.rag.chat import BirdChat

logger = logging.getLogger(__name__)


class RAGService:
    """Application-layer facade over BirdChat.

    Exists as a thin, separately-testable layer between the API routes
    and BirdChat, so route handlers depend on this service rather than
    importing `src.rag` directly.
    """

    def __init__(self, chat: Optional[BirdChat] = None) -> None:
        """Store initial chat instance or setup lazy initialization parameters.

        Args:
            chat: An already-constructed BirdChat instance, primarily
                for dependency injection in tests. If None, a new
                BirdChat is constructed using its default configuration.
        """
        self._chat = chat
        self._lock = threading.Lock()

    def _ensure_initialized(self) -> None:
        """Initialize BirdChat thread-safely."""
        if self._chat is not None:
            return

        with self._lock:
            if self._chat is not None:
                return

            logger.info("Initializing RAGService...")
            # Delayed import to avoid loading sentence_transformers/faiss on startup
            from src.rag.chat import BirdChat as _BirdChat
            self._chat = _BirdChat()
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
        self._ensure_initialized()
        assert self._chat is not None
        return self._chat.ask(species=species, question=question)

