"""Vector similarity retrieval for the RAG pipeline.

This module is responsible for ONE thing: given a natural-language
query, returning the most relevant knowledge base chunks from the FAISS
vector store. It performs no markdown parsing (ingest.py), no index
building (embeddings.py), and makes NO LLM calls of any kind — retrieval
is purely a similarity search step.
"""

from __future__ import annotations
import re
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

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
    """Normalize species names for comparison."""
    value = value.strip().lower()

    # Remove folder prefix like "017."
    value = re.sub(r"^\d+\.", "", value)

    # Convert underscores to spaces
    value = value.replace("_", " ")

    # Collapse multiple spaces
    value = " ".join(value.split())

    return value


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

    Species-aware retrieval
    -----------------------
    When ``species`` is supplied to :meth:`retrieve`, we *first* restrict
    the candidate pool to only those FAISS positions that belong to that
    species, then rank those positions by cosine similarity to the query.
    This guarantees that relevant chunks are always found, regardless of
    how semantically similar the query is to other species' content.

    The position map ``_species_positions`` is built once at construction
    time in a single O(n) pass over the metadata list (n ≈ 772).  At
    query time, looking up the positions for a species is O(1), and
    reconstructing 3-5 vectors from a flat FAISS index is O(k × dim) —
    faster than a global 772-vector search.
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
                to ``VectorStoreConfig()`` if not provided.
            embedding_model: Optional pre-constructed EmbeddingModel,
                reused for query embedding. If None, a new one is
                created using ``config.model_name``. Passing the same
                instance used at index-build time guarantees query
                vectors live in the same embedding space as the index.

        Raises:
            FileNotFoundError: If the vector store has not been built
                yet (see ``embeddings.build_or_load_vector_store``).
        """
        self.config = config or VectorStoreConfig()
        self._index, self._metadata = load_vector_store(self.config)
        self._embedding_model = embedding_model or EmbeddingModel(model_name=self.config.model_name)

        # Build an O(1) lookup from species folder name → list of FAISS
        # positions.  This single O(n) pass at startup is cheaper than
        # scanning metadata on every retrieve() call.
        self._species_positions: dict[str, list[int]] = {}
        for position, record in enumerate(self._metadata):
            key = record["species"]
            self._species_positions.setdefault(key, []).append(position)

        logger.info(
            "Retriever ready: %d vectors, %d species",
            self._index.ntotal,
            len(self._species_positions),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve_for_species(
        self,
        query_vector: np.ndarray,
        species: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rank only the chunks that belong to *species* by similarity.

        Uses ``IndexFlatIP.reconstruct()`` to fetch the stored vectors
        for the target species, then computes cosine similarity via a
        simple dot product (valid because all vectors are L2-normalised
        at index-build time).

        This is guaranteed to find relevant chunks because it *never*
        discards any candidate before scoring — every chunk for the
        species is considered.

        Args:
            query_vector: Shape ``(dim,)`` float32, L2-normalised.
            species: Canonical species folder name, e.g. ``"017.Cardinal"``.
            top_k: Maximum number of chunks to return.

        Returns:
            List of :class:`RetrievedChunk` ordered by descending score,
            at most ``top_k`` items.
        """
        # Resolve positions using exact folder name first; fall back to
        # normalized matching in case the caller passes a slightly
        # different form (e.g. without the numeric prefix).
        positions = self._species_positions.get(species)

        if positions is None:
            # Try normalized matching as a fallback.
            norm_requested = _normalize_species_name(species)
            for stored_key, pos_list in self._species_positions.items():
                if _normalize_species_name(stored_key) == norm_requested:
                    positions = pos_list
                    break

        if not positions:
            logger.warning(
                "No vectors found in index for species: %s", species
            )
            return []

        # Reconstruct the stored float32 vectors for this species.
        # IndexFlatIP.reconstruct(i) is O(dim) and always correct for
        # flat indexes — it returns the exact vector that was added.
        species_vectors = np.stack(
            [self._index.reconstruct(pos) for pos in positions],
            axis=0,
        )  # shape: (n_chunks, dim)

        # Cosine similarity = dot product for L2-normalised vectors.
        scores = species_vectors @ query_vector  # shape: (n_chunks,)

        # Sort descending and take the top_k.
        order = np.argsort(scores)[::-1][:top_k]

        chunks: list[RetrievedChunk] = []
        for idx in order:
            pos = positions[idx]
            record = self._metadata[pos]
            chunks.append(
                RetrievedChunk(
                    text=record["text"],
                    species=record["species"],
                    source_path=record["source_path"],
                    chunk_index=record["chunk_index"],
                    score=float(scores[idx]),
                )
            )

        return chunks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        species: Optional[str] = None,
        over_fetch_factor: int = DEFAULT_OVER_FETCH_FACTOR,
    ) -> list[RetrievedChunk]:
        """Retrieve the top-k most relevant chunks for a query.

        When ``species`` is supplied the search is restricted to that
        species' chunks *before* any similarity comparison, so results
        are always grounded in the correct species' knowledge.

        When ``species`` is ``None`` the existing global FAISS search is
        used unchanged.

        Args:
            query: Natural-language question or search text.
            top_k: Number of chunks to return.
            species: If provided, restrict results to chunks whose
                metadata species matches this value.  Must be the
                canonical folder name returned by ``POST /predict``,
                e.g. ``"017.Cardinal"``.
            over_fetch_factor: Retained for API compatibility.  Has no
                effect when ``species`` is supplied, because the
                species-aware path considers every chunk for that species
                rather than relying on FAISS over-fetching.

        Returns:
            List of :class:`RetrievedChunk` objects ordered by
            descending similarity score. May contain fewer than
            ``top_k`` items if the knowledge base has fewer chunks for
            the species than requested.

        Raises:
            ValueError: If ``query`` is empty/blank or ``top_k`` is not
                positive.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        # Embed the query once regardless of the retrieval path.
        query_embedding = self._embedding_model.encode([query])
        query_vector = query_embedding[0]  # shape: (dim,)

        # ------------------------------------------------------------------
        # Species-aware path: filter-first, then rank.
        # ------------------------------------------------------------------
        if species:
            chunks = self._retrieve_for_species(query_vector, species, top_k)
            if not chunks:
                logger.warning("No chunks found for species filter: %s", species)
            return chunks

        # ------------------------------------------------------------------
        # Global path (no species filter): existing FAISS search unchanged.
        # ------------------------------------------------------------------
        scores, indices = self._index.search(query_embedding, top_k)

        candidates: list[RetrievedChunk] = []
        for score, position in zip(scores[0], indices[0]):
            if position < 0:
                continue
            record = self._metadata[position]
            candidates.append(
                RetrievedChunk(
                    text=record["text"],
                    species=record["species"],
                    source_path=record["source_path"],
                    chunk_index=record["chunk_index"],
                    score=float(score),
                )
            )

        return candidates
