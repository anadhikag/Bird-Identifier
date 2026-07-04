"""Knowledge base population driver for the Bird Identification Assistant.

This module is responsible for ONE thing: walking every species folder
created by `knowledge_builder.py`, and — for any species not yet
generated — invoking a single content-generation hook and persisting its
result.

The actual content generation (LLM call, web scraping, or any other data
source) is intentionally NOT implemented here. `generate_species_markdown`
is a stable interface that a future implementation will plug into without
requiring any change to the orchestration logic in this file.
"""

from __future__ import annotations

from src.llm.groq_client import GroqClient
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

groq_client = GroqClient()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_species_markdown(species_name: str) -> str:
    """
    Generate the markdown for a bird species using Groq.
    """
    return groq_client.generate_species_markdown(species_name)


@dataclass
class PopulationStats:
    """Tracks progress while populating the knowledge base."""

    total: int = 0
    skipped: int = 0
    succeeded: int = 0
    failed: int = 0
    failed_species: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Build a human-readable summary of the population run."""
        return (
            f"Total species: {self.total} | "
            f"Skipped (already generated): {self.skipped} | "
            f"Succeeded: {self.succeeded} | "
            f"Failed: {self.failed}"
        )


def load_metadata(metadata_path: Path) -> dict:
    """Load a species' metadata.json file.

    Args:
        metadata_path: Path to the metadata.json file.

    Returns:
        Parsed metadata dictionary.

    Raises:
        FileNotFoundError: If metadata_path does not exist.
        json.JSONDecodeError: If metadata_path contains invalid JSON.
    """
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_metadata(metadata_path: Path, metadata: dict) -> None:
    """Persist a species' metadata.json file.

    Args:
        metadata_path: Path to write the metadata.json file to.
        metadata: Metadata dictionary to serialize.
    """
    metadata_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")


def determine_species_name(metadata: dict, species_dir: Path) -> str:
    """Determine the species' human-readable name for content generation.

    Prefers the `common_name` field from metadata.json, falling back to
    the folder name (with underscores replaced by spaces) if metadata is
    missing that field.

    Args:
        metadata: Parsed metadata.json contents.
        species_dir: Path to the species' folder, used as a fallback.

    Returns:
        Human-readable species common name.
    """
    common_name = metadata.get("common_name")
    if common_name:
        return str(common_name)
    return species_dir.name.replace("_", " ")


def process_species_folder(species_dir: Path) -> bool:
    """Process a single species folder: generate and persist content.

    Args:
        species_dir: Path to the species folder (must contain
            species.md and metadata.json).

    Returns:
        True if content was generated and saved successfully, False if
        generation was skipped because it was already done.

    Raises:
        FileNotFoundError: If required files are missing.
        NotImplementedError: Propagated from `generate_species_markdown`
            until that function is implemented.
        Exception: Any other error raised during generation is
            propagated to the caller, which is responsible for
            catching it and continuing to the next species.
    """
    metadata_path = species_dir / "metadata.json"
    species_md_path = species_dir / "species.md"

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in: {species_dir}")
    if not species_md_path.exists():
        raise FileNotFoundError(f"species.md not found in: {species_dir}")

    metadata = load_metadata(metadata_path)

    if metadata.get("generated") is True:
        return False

    species_name = determine_species_name(metadata, species_dir)

    # Existing species.md content is available here if a future
    # implementation wants to use it as additional context, but the
    # current interface only requires the species name.
    _ = species_md_path.read_text(encoding="utf-8")

    generated_markdown = generate_species_markdown(species_name)

    species_md_path.write_text(generated_markdown, encoding="utf-8")

    metadata["generated"] = True
    save_metadata(metadata_path, metadata)

    return True


def iter_species_folders(knowledge_dir: Path) -> list[Path]:
    """List all species folders within the knowledge base directory.

    Args:
        knowledge_dir: Root directory containing one folder per species.

    Returns:
        Sorted list of species folder paths.

    Raises:
        FileNotFoundError: If knowledge_dir does not exist.
    """
    if not knowledge_dir.exists():
        raise FileNotFoundError(f"Knowledge directory not found: {knowledge_dir}")

    return sorted(path for path in knowledge_dir.iterdir() if path.is_dir())


def populate_knowledge_base(knowledge_dir: Path) -> PopulationStats:
    """Populate every not-yet-generated species folder in the knowledge base.

    Iterates every species folder, skips those already marked as
    generated, and calls `generate_species_markdown` for the rest. Any
    failure for a single species is logged and does not stop processing
    of the remaining species.

    Args:
        knowledge_dir: Root directory containing one folder per species.

    Returns:
        PopulationStats summarizing the outcome of the run.
    """
    species_dirs = iter_species_folders(knowledge_dir)
    stats = PopulationStats(total=len(species_dirs))

    for index, species_dir in enumerate(species_dirs, start=1):
        logger.info("[%d/%d] Processing: %s", index, stats.total, species_dir.name)

        try:
            was_generated = process_species_folder(species_dir)
        except Exception as error:  # noqa: BLE001 - intentional catch-all to continue the batch
            stats.failed += 1
            stats.failed_species.append(species_dir.name)
            logger.error("Failed to generate content for %s: %s", species_dir.name, error)
            continue

        if was_generated:
            stats.succeeded += 1
            logger.info("Generated and saved species.md for: %s", species_dir.name)

        else:
            stats.skipped += 1
            logger.info("Already generated, skipping: %s", species_dir.name)

    return stats


def main() -> None:
    """Entry point: populate the knowledge base using default project paths."""
    project_root = Path(__file__).resolve().parents[2]
    knowledge_dir = project_root / "knowledge"

    stats = populate_knowledge_base(knowledge_dir)

    logger.info(stats.summary())
    if stats.failed_species:
        logger.info("Failed species: %s", ", ".join(stats.failed_species))


if __name__ == "__main__":
    main()
