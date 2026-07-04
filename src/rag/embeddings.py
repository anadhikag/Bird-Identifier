"""Embedding model and FAISS vector store management for the RAG pipeline.

This module is responsible for ONE thing: turning chunked Document
objects (from ingest.py) into vector embeddings, and building, saving,
and loading the FAISS index that stores them.

It has no knowledge of markdown parsing (ingest.py) or of how retrieved
chunks are used downstream (retriever.py, chat.py). The `EmbeddingModel`
class defined here is shared by retriever.py so that query-time and
index-time embeddings are always produced by the exact same model and
configuration.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.rag.ingest import Document

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
DEFAULT_INDEX_PATH: Path = Path("vector_db/faiss.index")
DEFAULT_METADATA_PATH: Path = Path("vector_db/metadata.pkl")
DEFAULT_ENCODE_BATCH_SIZE: int = 32


@dataclass(frozen=True)
class VectorStoreConfig:
    """Filesystem and model configuration for the FAISS vector store.

    Attributes:
        index_path: Path to the on-disk FAISS index file.
        metadata_path: Path to the on-disk pickled metadata list, kept
            in the same order as vectors in the FAISS index.
        model_name: Name of the sentence-transformers embedding model
            used to build and query this vector store.
    """

    index_path: Path = DEFAULT_INDEX_PATH
    metadata_path: Path = DEFAULT_METADATA_PATH
    model_name: str = DEFAULT_MODEL_NAME


class EmbeddingModel:
    """Loads BAAI/bge-small-en-v1.5 once and embeds text into vectors.

    Shared by both index building (this module) and query embedding
    (retriever.py), so the embedding logic — and the resulting vector
    space — is guaranteed to be identical at index time and query time.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model: Optional[SentenceTransformer] = None,
    ) -> None:
        """Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformers model to load.
                Ignored if `model` is provided.
            model: An already-constructed SentenceTransformer instance.
                Primarily intended for dependency injection in tests;
                production code should leave this as None so the model
                is loaded from `model_name`.
        """
        self.model_name = model_name
        self._model = model or SentenceTransformer(model_name)
        logger.info("Embedding model ready: %s", model_name)

    @property
    def embedding_dimension(self) -> int:
        """Dimensionality of vectors produced by this model."""
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts into L2-normalized vectors.

        Vectors are L2-normalized so that FAISS inner-product search
        (`IndexFlatIP`) is mathematically equivalent to cosine
        similarity search.

        Args:
            texts: List of text strings to embed.

        Returns:
            Array of shape (len(texts), embedding_dimension), dtype
            float32, L2-normalized.
        """
        embeddings = self._model.encode(
            texts,
            batch_size=DEFAULT_ENCODE_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a flat inner-product FAISS index from embedding vectors.

    Args:
        embeddings: Array of shape (n, dim) of L2-normalized float32
            vectors.

    Returns:
        A populated `faiss.IndexFlatIP` instance.

    Raises:
        ValueError: If embeddings is empty.
    """
    if embeddings.size == 0:
        raise ValueError("Cannot build a FAISS index from empty embeddings.")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def document_to_metadata(document: Document) -> dict:
    """Convert a Document into a plain-dict metadata record for storage.

    Storing plain dicts (rather than pickling Document instances
    directly) keeps the on-disk metadata format decoupled from the
    Document dataclass definition in ingest.py.

    Args:
        document: Source Document instance.

    Returns:
        Dictionary with the fields needed to reconstruct retrieval
        results at query time.
    """
    return {
        "doc_id": document.doc_id,
        "species": document.species,
        "source_path": document.source_path,
        "chunk_index": document.chunk_index,
        "text": document.text,
    }


def save_vector_store(index: faiss.Index, metadata: list[dict], config: VectorStoreConfig) -> None:
    """Persist a FAISS index and its parallel metadata list to disk.

    Args:
        index: FAISS index to save.
        metadata: List of metadata dicts, one per vector, in the same
            order they were added to the index.
        config: Vector store configuration specifying output paths.
    """
    config.index_path.parent.mkdir(parents=True, exist_ok=True)
    config.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(config.index_path))
    with config.metadata_path.open("wb") as handle:
        pickle.dump(metadata, handle)

    logger.info("Saved FAISS index to %s", config.index_path)
    logger.info("Saved metadata to %s", config.metadata_path)


def load_vector_store(config: VectorStoreConfig) -> tuple[faiss.Index, list[dict]]:
    """Load a previously saved FAISS index and its metadata from disk.

    Args:
        config: Vector store configuration specifying input paths.

    Returns:
        Tuple of (faiss_index, metadata_list).

    Raises:
        FileNotFoundError: If either the index or metadata file is
            missing.
    """
    if not config.index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at: {config.index_path}")
    if not config.metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {config.metadata_path}")

    index = faiss.read_index(str(config.index_path))
    with config.metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)

    logger.info("Loaded FAISS index from %s (%d vectors)", config.index_path, index.ntotal)
    return index, metadata


def vector_store_exists(config: VectorStoreConfig) -> bool:
    """Check whether a FAISS index and metadata file already exist.

    Args:
        config: Vector store configuration specifying file paths.

    Returns:
        True if both the index and metadata files exist on disk.
    """
    return config.index_path.exists() and config.metadata_path.exists()


def build_or_load_vector_store(
    documents: list[Document],
    config: Optional[VectorStoreConfig] = None,
    embedding_model: Optional[EmbeddingModel] = None,
    force_rebuild: bool = False,
) -> tuple[faiss.Index, list[dict]]:
    """Build the FAISS index if needed, or load it if it already exists.

    This is the main entry point for populating the vector store. It
    avoids redundant, expensive re-embedding of all 200 species by
    reusing an existing on-disk index unless `force_rebuild` is True.

    Args:
        documents: Chunked Document objects to embed and index. Only
            used when a rebuild is actually performed.
        config: Vector store configuration specifying model and paths.
            Defaults to `VectorStoreConfig()` if not provided.
        embedding_model: Optional pre-constructed EmbeddingModel. If
            None, one is created using `config.model_name`.
        force_rebuild: If True, rebuild the index even if one already
            exists on disk.

    Returns:
        Tuple of (faiss_index, metadata_list) ready for use by
        retriever.py.

    Raises:
        ValueError: If a rebuild is required but `documents` is empty.
    """
    resolved_config = config or VectorStoreConfig()

    if not force_rebuild and vector_store_exists(resolved_config):
        logger.info("Existing vector store found, skipping rebuild.")
        return load_vector_store(resolved_config)

    if not documents:
        raise ValueError("Cannot build a vector store from an empty document list.")

    logger.info("Building vector store from %d document chunks.", len(documents))
    model = embedding_model or EmbeddingModel(model_name=resolved_config.model_name)

    texts = [document.text for document in documents]
    embeddings = model.encode(texts)

    index = build_faiss_index(embeddings)
    metadata = [document_to_metadata(document) for document in documents]

    save_vector_store(index, metadata, resolved_config)
    return index, metadata
