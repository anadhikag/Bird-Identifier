"""Build (or load) the FAISS vector database for the RAG pipeline.

This script is the single command-line entry point for turning the
on-disk knowledge base (`knowledge/<species>/species.md`) into the
persisted FAISS vector store consumed by `src.rag.retriever.Retriever`.

It performs no ingestion, embedding, or indexing logic itself — all of
that lives in `src.rag.ingest` and `src.rag.embeddings`, so this script
is purely an orchestration and reporting layer.

Usage:
    python -m scripts.build_vector_db
    python -m scripts.build_vector_db --force-rebuild
    python -m scripts.build_vector_db --chunk-size 1000 --chunk-overlap 150
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from src.rag.embeddings import VectorStoreConfig, build_or_load_vector_store
from src.rag.ingest import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, IngestConfig, load_document_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildConfig:
    """Resolved paths and parameters for a single vector database build.

    Attributes:
        knowledge_dir: Root directory containing one folder per species.
        vector_db_dir: Root directory where the FAISS index and metadata
            are saved.
        chunk_size: Target maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between
            consecutive chunks.
        force_rebuild: If True, rebuild the vector store even if one
            already exists on disk.
    """

    knowledge_dir: Path
    vector_db_dir: Path
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    force_rebuild: bool = False


def _format_bytes(num_bytes: int) -> str:
    """Format a byte count as a human-readable string.

    Args:
        num_bytes: Size in bytes.

    Returns:
        Human-readable size string, e.g. "482.3 KB" or "1.2 MB".
    """
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def run_build(config: BuildConfig) -> None:
    """Load the knowledge base, build/load the FAISS index, and report stats.

    Args:
        config: Resolved build configuration.
    """
    ingest_config = IngestConfig(
        knowledge_dir=config.knowledge_dir,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    documents = load_document_chunks(ingest_config)

    num_species = len({document.species for document in documents})
    num_chunks = len(documents)

    vector_store_config = VectorStoreConfig(
        index_path=config.vector_db_dir / "faiss.index",
        metadata_path=config.vector_db_dir / "metadata.pkl",
    )

    index, metadata = build_or_load_vector_store(
        documents=documents,
        config=vector_store_config,
        force_rebuild=config.force_rebuild,
    )

    embedding_dimension = index.d
    index_size = index.ntotal
    index_file_size = (
        _format_bytes(vector_store_config.index_path.stat().st_size)
        if vector_store_config.index_path.exists()
        else "unknown"
    )

    print("=" * 60)
    print("Vector Database Build Report")
    print("=" * 60)
    print(f"Species:              {num_species}")
    print(f"Chunks:                {num_chunks}")
    print(f"Embedding dimension:   {embedding_dimension}")
    print(f"Index size (vectors):  {index_size}")
    print(f"Metadata records:      {len(metadata)}")
    print(f"Index file size:       {index_file_size}")
    print(f"Index saved to:        {vector_store_config.index_path}")
    print(f"Metadata saved to:     {vector_store_config.metadata_path}")
    print("=" * 60)


def parse_args() -> BuildConfig:
    """Parse command-line arguments into a BuildConfig.

    Returns:
        Resolved BuildConfig using project-relative default paths.
    """
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Build (or load) the FAISS vector database for the RAG pipeline."
    )
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        default=project_root / "knowledge",
        help="Root directory containing one folder per species.",
    )
    parser.add_argument(
        "--vector-db-dir",
        type=Path,
        default=project_root / "vector_db",
        help="Directory to save/load the FAISS index and metadata.",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild the vector store even if one already exists on disk.",
    )
    args = parser.parse_args()

    return BuildConfig(
        knowledge_dir=args.knowledge_dir,
        vector_db_dir=args.vector_db_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        force_rebuild=args.force_rebuild,
    )


def main() -> None:
    """Entry point: build or load the vector database and report stats."""
    config = parse_args()
    run_build(config)


if __name__ == "__main__":
    main()
