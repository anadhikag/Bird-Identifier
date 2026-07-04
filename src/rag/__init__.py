"""RAG (Retrieval-Augmented Generation) package for the Bird Identification Assistant.

Exposes the public entry point (`BirdChat`) plus the core building
blocks of the pipeline for advanced or manual usage:

    from src.rag import BirdChat

    chat = BirdChat()
    answer = chat.ask(species="Blue Jay", question="Why does this bird migrate?")
    print(answer)
"""

from src.rag.chat import BirdChat
from src.rag.embeddings import EmbeddingModel, VectorStoreConfig, build_or_load_vector_store
from src.rag.ingest import Document, IngestConfig, load_document_chunks
from src.rag.prompts import RAG_SYSTEM_PROMPT, format_retrieved_context
from src.rag.retriever import Retriever, RetrievedChunk

__all__ = [
    "BirdChat",
    "EmbeddingModel",
    "VectorStoreConfig",
    "build_or_load_vector_store",
    "Document",
    "IngestConfig",
    "load_document_chunks",
    "RAG_SYSTEM_PROMPT",
    "format_retrieved_context",
    "Retriever",
    "RetrievedChunk",
]
