"""Reusable prompt templates for the RAG pipeline.

This module is responsible for ONE thing: defining the text used to
instruct the LLM and to format retrieved context into prompts. It
contains no retrieval logic (retriever.py) and makes NO LLM calls itself
(chat.py) — it only produces strings.
"""

from __future__ import annotations

from src.rag.retriever import RetrievedChunk

RAG_SYSTEM_PROMPT: str = """\
You are an ornithologist.
The bird species has already been identified by a CNN.
Answer ONLY using the retrieved context.
If the answer is not contained in the context, say "I do not know."
Never invent facts.
Never use outside knowledge."""

NO_CONTEXT_ANSWER: str = "I do not know."


def format_retrieved_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a single context block for the LLM.

    Args:
        chunks: Retrieved chunks, ordered by descending relevance.

    Returns:
        A single string with each chunk clearly separated and labeled
        by source, suitable for use as the `retrieved_context` argument
        to `GroqClient.answer_question`. Returns an empty string if no
        chunks were provided.
    """
    if not chunks:
        return ""

    sections = [
        f"[Source {index} - {chunk.species}]\n{chunk.text}"
        for index, chunk in enumerate(chunks, start=1)
    ]
    return "\n\n---\n\n".join(sections)


def build_rag_user_prompt(species: str, context: str, question: str) -> str:
    """Build the full user-role message for a RAG-grounded question.

    This mirrors the message structure `GroqClient.answer_question`
    constructs internally. It is exposed here as a reusable template so
    other LLM integrations (beyond Groq) can be wired into this RAG
    pipeline later without duplicating prompt-formatting logic.

    Args:
        species: Common name of the already-identified bird species.
        context: Formatted retrieved context, typically the output of
            `format_retrieved_context`.
        question: The user's natural-language question.

    Returns:
        Combined user-role prompt string.
    """
    return (
        f"Identified species: {species}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Question: {question}"
    )
