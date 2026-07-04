"""End-to-end RAG chat orchestration for the Bird Identification Assistant.

This module is responsible for ONE thing: wiring retrieval
(retriever.py), context formatting (prompts.py), and the Groq LLM
(src.llm.groq_client) together into the single public entry point the
rest of the application uses to ask a grounded question about an
already-identified bird species.

Public API:
    chat = BirdChat()
    answer = chat.ask(species="Blue Jay", question="Why does this bird migrate?")
    print(answer)
"""

from __future__ import annotations

import logging
from typing import Optional

from src.llm.groq_client import GroqClient
from src.rag.embeddings import VectorStoreConfig
from src.rag.prompts import NO_CONTEXT_ANSWER, format_retrieved_context
from src.rag.retriever import DEFAULT_TOP_K, Retriever

logger = logging.getLogger(__name__)


class BirdChat:
    """High-level RAG chat interface for answering species questions.

    Composes a `Retriever` and a `GroqClient` so callers (Streamlit
    today, FastAPI/React later) only need to construct one object and
    call `ask()`. Neither retrieval nor prompt-formatting logic is
    duplicated here — this class only orchestrates calls to
    retriever.py, prompts.py, and src.llm.groq_client.
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        groq_client: Optional[GroqClient] = None,
        vector_store_config: Optional[VectorStoreConfig] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        """Initialize the chat pipeline.

        Args:
            retriever: Optional pre-constructed Retriever, primarily for
                dependency injection in tests. If None, one is built
                using `vector_store_config`.
            groq_client: Optional pre-constructed GroqClient, primarily
                for dependency injection in tests. If None, a new one is
                constructed using its default configuration (reads
                GROQ_API_KEY from the environment).
            vector_store_config: Vector store configuration used to
                build the default Retriever. Ignored if `retriever` is
                provided.
            top_k: Default number of chunks to retrieve per question.

        Raises:
            FileNotFoundError: If no `retriever` is given and the
                vector store has not been built yet.
            ValueError: If no `groq_client` is given and GROQ_API_KEY is
                not set in the environment.
        """
        self._retriever = retriever or Retriever(config=vector_store_config)
        self._groq_client = groq_client or GroqClient()
        self.top_k = top_k
        logger.info("BirdChat initialized.")

    def ask(self, species: str, question: str, top_k: Optional[int] = None) -> str:
        """Answer a question about an already-identified bird species.

        Retrieves relevant knowledge base context scoped to `species`,
        formats it into a prompt, and delegates the final answer
        generation to `GroqClient.answer_question`.

        Args:
            species: Common name of the species identified by the CNN
                classifier, e.g. "Blue Jay".
            question: The user's natural-language question.
            top_k: Number of chunks to retrieve for this question.
                Defaults to `self.top_k` if not provided.

        Returns:
            The grounded, plain-text answer produced by the LLM. If no
            relevant context is found for the species, a fixed
            "I do not know." response is returned without calling the
            LLM.

        Raises:
            ValueError: If species or question is empty or blank.
            RuntimeError: If the underlying Groq API call fails.
        """
        if not species or not species.strip():
            raise ValueError("species must be a non-empty string.")
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string.")

        resolved_top_k = top_k or self.top_k
        logger.info("Answering question for species '%s': %s", species, question)

        chunks = self._retriever.retrieve(
            query=question, top_k=resolved_top_k, species=species
        )

        if not chunks:
            logger.warning(
                "No retrieved context for species '%s'; returning default answer.", species
            )
            return NO_CONTEXT_ANSWER

        context = format_retrieved_context(chunks)

        answer = self._groq_client.answer_question(
            species=species,
            retrieved_context=context,
            question=question,
        )

        logger.info("Answer generated for species '%s'.", species)
        return answer
