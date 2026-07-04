"""Markdown knowledge base loader and chunker for the RAG pipeline.

This module is responsible for ONE thing: turning the on-disk knowledge
base (`knowledge/<species>/species.md`) into a flat list of chunked
`Document` objects, each carrying enough metadata to be embedded,
indexed, retrieved, and traced back to its source file.

It has no knowledge of embeddings, vector stores, or LLMs — those
responsibilities live in embeddings.py, retriever.py, and chat.py.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE: int = 800
DEFAULT_CHUNK_OVERLAP: int = 100
SPECIES_MARKDOWN_FILENAME: str = "species.md"


@dataclass(frozen=True)
class Document:
    """A single retrievable chunk of a species knowledge document.

    Attributes:
        doc_id: Stable, unique identifier for this chunk, of the form
            "<species>::<chunk_index>".
        species: Species folder name (e.g. "Black_footed_Albatross"),
            preserved exactly as it appears on disk.
        source_path: Absolute path to the source species.md file.
        chunk_index: 0-based position of this chunk within its source
            document.
        text: The chunk's raw markdown text content.
    """

    doc_id: str
    species: str
    source_path: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class IngestConfig:
    """Configuration controlling knowledge base loading and chunking.

    Attributes:
        knowledge_dir: Root directory containing one folder per species.
        chunk_size: Target maximum number of characters per chunk.
        chunk_overlap: Number of characters carried over between
            consecutive chunks to preserve context across boundaries.
    """

    knowledge_dir: Path
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP


def find_species_markdown_files(knowledge_dir: Path) -> list[Path]:
    """Recursively find every species.md file under the knowledge base.

    Args:
        knowledge_dir: Root directory containing one folder per species.

    Returns:
        Sorted list of paths to species.md files.

    Raises:
        FileNotFoundError: If knowledge_dir does not exist.
    """
    if not knowledge_dir.exists():
        raise FileNotFoundError(f"Knowledge directory not found: {knowledge_dir}")

    return sorted(knowledge_dir.rglob(SPECIES_MARKDOWN_FILENAME))


def _split_into_paragraphs(text: str) -> list[str]:
    """Split markdown text into non-empty paragraph blocks.

    Paragraphs are separated by one or more blank lines, matching the
    section-separated structure of the species.md template.

    Args:
        text: Raw markdown text.

    Returns:
        List of non-empty, stripped paragraph strings.
    """
    raw_paragraphs = re.split(r"\n\s*\n", text)
    return [paragraph.strip() for paragraph in raw_paragraphs if paragraph.strip()]


def _split_oversized_paragraph(paragraph: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split a single paragraph that exceeds chunk_size into slices.

    Args:
        paragraph: Paragraph text longer than chunk_size.
        chunk_size: Maximum number of characters per slice.
        chunk_overlap: Number of overlapping characters between
            consecutive slices, to preserve local context.

    Returns:
        List of text slices, each at most chunk_size characters.
    """
    slices: list[str] = []
    start = 0
    text_length = len(paragraph)
    step = max(chunk_size - chunk_overlap, 1)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        slices.append(paragraph[start:end].strip())
        if end == text_length:
            break
        start += step

    return [text_slice for text_slice in slices if text_slice]


def chunk_markdown(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split markdown text into overlapping, size-bounded chunks.

    Chunks are built by greedily accumulating whole paragraphs (as
    delimited by blank lines) up to `chunk_size` characters. When a
    single paragraph exceeds `chunk_size` on its own, it is further
    split into fixed-size overlapping slices.

    Args:
        text: Raw markdown text of a single species.md document.
        chunk_size: Target maximum number of characters per chunk.
        chunk_overlap: Number of characters carried over from the end of
            one chunk into the start of the next, to preserve context
            across chunk boundaries.

    Returns:
        Ordered list of markdown chunk strings.

    Raises:
        ValueError: If chunk_size is not positive, or chunk_overlap is
            negative or greater than or equal to chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")

    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            chunks.extend(_split_oversized_paragraph(paragraph, chunk_size, chunk_overlap))
            continue

        candidate = f"{current_chunk}\n\n{paragraph}" if current_chunk else paragraph

        if len(candidate) <= chunk_size:
            current_chunk = candidate
            continue

        chunks.append(current_chunk)
        overlap_tail = current_chunk[-chunk_overlap:] if chunk_overlap else ""
        current_chunk = f"{overlap_tail}\n\n{paragraph}".strip() if overlap_tail else paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def load_document_chunks(config: IngestConfig) -> list[Document]:
    """Load, chunk, and wrap every species.md file into Document objects.

    Args:
        config: Ingestion configuration specifying the knowledge base
            location and chunking parameters.

    Returns:
        Flat list of Document objects across all species, ready to be
        embedded and indexed.
    """
    markdown_paths = find_species_markdown_files(config.knowledge_dir)
    logger.info("Found %d species.md files under %s", len(markdown_paths), config.knowledge_dir)

    documents: list[Document] = []

    for markdown_path in markdown_paths:
        species = markdown_path.parent.name
        text = markdown_path.read_text(encoding="utf-8")
        chunks = chunk_markdown(
            text, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )

        for chunk_index, chunk_text in enumerate(chunks):
            documents.append(
                Document(
                    doc_id=f"{species}::{chunk_index}",
                    species=species,
                    source_path=str(markdown_path.resolve()),
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )

        logger.info("Chunked %s into %d chunk(s).", species, len(chunks))

    logger.info("Loaded %d total chunks across %d species.", len(documents), len(markdown_paths))
    return documents
