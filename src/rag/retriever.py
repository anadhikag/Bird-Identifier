"""Vector similarity retrieval for the RAG pipeline.

This module is responsible for ONE thing: given a natural-language
query, returning the most relevant knowledge base chunks from the FAISS
vector store. It performs no markdown parsing (ingest.py), no index
building (embeddings.py), and makes NO LLM calls of any kind — retrieval
is purely a similarity search step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.rag.embeddings import EmbeddingModel, VectorStoreConfig, load_vector_store

logger = logging.getLogger(__name__)

DEFAULT_TOP_K: int = 5
DEFAULT_OVER_FETCH_FACTOR: int = 20


@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieved knowledge base chunk with its similarity score.

    Attributes:
        text: The chunk's markdown text content.
        species: Species folder name this chunk belongs to.
        source_path: Absolute path to the source species.md file.
        chunk_index: 0-based position of this chunk within its source
            document.
        score: Cosine similarity score (higher is more relevant).
    """

    text: str
    species: str
    source_path: str
    chunk_index: int
    score: float


def _normalize_species_name(value: str) -> str:
    """Normalize a species name for loose, separator-insensitive comparison.

    Args:
        value: A species name, e.g. "Blue_Jay" or "Blue Jay".

    Returns:
        Lowercased, underscore-free, whitespace-trimmed representation.
    """
    return value.strip().lower().replace("_", " ")


def _species_matches(stored_species: str, requested_species: str) -> bool:
    """Compare two species names loosely, ignoring case and separators.

    Args:
        stored_species: Species name as stored in chunk metadata (e.g.
            folder-derived, may use underscores).
        requested_species: Species name as provided by the caller (e.g.
            "Blue Jay").

    Returns:
        True if the two names refer to the same species after
        normalization.
    """
    return _normalize_species_name(stored_species) == _normalize_species_name(requested_species)


class Retriever:
    """Retrieves the most relevant knowledge base chunks for a query.

    Loads the FAISS index and metadata once at construction time and
    reuses them, along with a single embedding model instance, across
    all subsequent `retrieve` calls.
    """

    def __init__(
        self,
        config: Optional[VectorStoreConfig] = None,
        embedding_model: Optional[EmbeddingModel] = None,
    ) -> None:
        """Initialize the retriever by loading the vector store.

        Args:
            config: Vector store configuration specifying index and
                metadata paths, and the embedding model name. Defaults
                to `VectorStoreConfig()` if not provided.
            embedding_model: Optional pre-constructed EmbeddingModel,
                reused for query embedding. If None, a new one is
                created using `config.model_name`. Passing the same
                instance used at index-build time guarantees query
                vectors live in the same embedding space as the index.

        Raises:
            FileNotFoundError: If the vector store has not been built
                yet (see `embeddings.build_or_load_vector_store`).
        """
        self.config = config or VectorStoreConfig()
        self._index, self._metadata = load_vector_store(self.config)
        self._embedding_model = embedding_model or EmbeddingModel(model_name=self.config.model_name)

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        species: Optional[str] = None,
        over_fetch_factor: int = DEFAULT_OVER_FETCH_FACTOR,
    ) -> list[RetrievedChunk]:
        """Retrieve the top-k most relevant chunks for a query.

        Args:
            query: Natural-language question or search text.
            top_k: Number of chunks to return.
            species: If provided, restrict results to chunks whose
                metadata species matches this value (case-insensitive,
                underscore/space-insensitive). Used to scope retrieval
                to the species already identified by the CNN
                classifier.
            over_fetch_factor: Multiplier applied to `top_k` when
                `species` filtering is requested, so FAISS returns
                enough candidates for post-filtering to still yield
                `top_k` results.

        Returns:
            List of RetrievedChunk objects ordered by descending
            similarity score. May contain fewer than `top_k` items if
            the vector store does not have enough matching chunks.

        Raises:
            ValueError: If query is empty/blank or top_k is not
                positive.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        # Retrieve more candidates when filtering by species
        fetch_count = top_k * over_fetch_factor if species else top_k
        fetch_count = max(min(fetch_count, self._index.ntotal), 1)

        # Include the species name in the semantic search query.
        # The CNN has already identified the bird, so this helps
        # the embedding model retrieve chunks from the correct species.
        search_query = query

        if species:
            search_query = f"Species: {species}\nQuestion: {query}"

        query_embedding = self._embedding_model.encode([search_query])
        scores, indices = self._index.search(query_embedding, fetch_count)

        candidates: list[RetrievedChunk] = []
        for score, position in zip(scores[0], indices[0]):
            if position < 0:
                continue

            record = self._metadata[position]
            if species and not _species_matches(record["species"], species):
                continue

            candidates.append(
                RetrievedChunk(
                    text=record["text"],
                    species=record["species"],
                    source_path=record["source_path"],
                    chunk_index=record["chunk_index"],
                    score=float(score),
                )
            )

            if len(candidates) >= top_k:
                break

        if species and not candidates:
            logger.warning("No chunks found for species filter: %s", species)

        return candidates
