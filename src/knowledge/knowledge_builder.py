"""Knowledge base scaffolding for the Bird Identification Assistant.

This module is responsible for ONE thing: creating the initial, empty
knowledge base directory structure — one folder per CUB-200-2011 species,
each containing a template `species.md` and a `metadata.json` file.

It does NOT generate any species content (no LLM calls, no web scraping).
That responsibility belongs to `populate_knowledge.py`. This script only
ever creates the skeleton and is safe to re-run at any time: existing
files are never overwritten.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeciesEntry:
    """A single row parsed from CUB-200-2011's classes.txt."""

    cub_class_id: int
    raw_name: str

    @property
    def common_name(self) -> str:
        """Human-readable common name, e.g. 'Black footed Albatross'."""
        return self.raw_name.replace("_", " ")


@dataclass(frozen=True)
class BuilderConfig:
    """Filesystem paths used by the knowledge base builder."""

    classes_txt_path: Path
    knowledge_dir: Path


SPECIES_MD_TEMPLATE = """# {common_name}

## Basic Information
Scientific Name:
Family:
Order:

---

## Identification
Average Length:
Average Wingspan:
Average Weight:
Male vs Female Differences:
Distinctive Features:

---

## Habitat

---

## Geographic Distribution

---

## Migration
Migratory Status:
Migration Pattern:

---

## Diet

---

## Behaviour

---

## Vocalization

---

## Breeding

---

## Conservation
IUCN Status:
Threats:

---

## Ecological Importance

---

## Interesting Facts
- Fact 1
- Fact 2
- Fact 3

---

## References
- Wikipedia
- Cornell Lab of Ornithology
- BirdLife International
"""


def parse_classes_file(classes_txt_path: Path) -> list[SpeciesEntry]:
    """Parse CUB-200-2011's classes.txt into structured species entries.

    Args:
        classes_txt_path: Path to the classes.txt file, where each line
            has the format "<class_id> <Raw_Species_Name>".

    Returns:
        List of SpeciesEntry objects, one per line in the file.

    Raises:
        FileNotFoundError: If classes_txt_path does not exist.
    """
    if not classes_txt_path.exists():
        raise FileNotFoundError(f"classes.txt not found at: {classes_txt_path}")

    entries: list[SpeciesEntry] = []
    with classes_txt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            class_id_str, raw_name = line.split(" ", 1)
            entries.append(SpeciesEntry(cub_class_id=int(class_id_str), raw_name=raw_name))

    return entries


def render_species_markdown(common_name: str) -> str:
    """Render the empty species.md template for a given species.

    Args:
        common_name: Human-readable species name to fill into the
            top-level heading.

    Returns:
        Markdown text following the fixed species.md template, with all
        fields left blank for later population.
    """
    return SPECIES_MD_TEMPLATE.format(common_name=common_name)


def build_metadata(entry: SpeciesEntry) -> dict:
    """Build the initial metadata dictionary for a species.

    Args:
        entry: Parsed species entry from classes.txt.

    Returns:
        Dictionary matching the required metadata.json schema, with
        `generated` set to False since no content has been created yet.
    """
    return {
        "cub_class_id": entry.cub_class_id,
        "common_name": entry.common_name,
        "scientific_name": "",
        "generated": False,
    }


def create_species_folder(entry: SpeciesEntry, knowledge_dir: Path) -> None:
    """Create a single species folder with species.md and metadata.json.

    Existing files are never overwritten, making this function — and the
    script as a whole — safe to run multiple times.

    Args:
        entry: Parsed species entry from classes.txt.
        knowledge_dir: Root directory under which species folders live.
    """
    species_dir = knowledge_dir / entry.raw_name
    species_dir.mkdir(parents=True, exist_ok=True)

    species_md_path = species_dir / "species.md"
    metadata_path = species_dir / "metadata.json"

    if species_md_path.exists():
        logger.info("Skipping species.md (already exists): %s", species_md_path)
    else:
        species_md_path.write_text(render_species_markdown(entry.common_name), encoding="utf-8")
        logger.info("Created species.md: %s", species_md_path)

    if metadata_path.exists():
        logger.info("Skipping metadata.json (already exists): %s", metadata_path)
    else:
        metadata = build_metadata(entry)
        metadata_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")
        logger.info("Created metadata.json: %s", metadata_path)


def build_knowledge_base(config: BuilderConfig) -> None:
    """Build (or resume building) the full knowledge base skeleton.

    Args:
        config: Resolved filesystem paths for the builder.
    """
    logger.info("Reading species list from: %s", config.classes_txt_path)
    entries = parse_classes_file(config.classes_txt_path)
    logger.info("Found %d species.", len(entries))

    config.knowledge_dir.mkdir(parents=True, exist_ok=True)

    for index, entry in enumerate(entries, start=1):
        logger.info("[%d/%d] Processing: %s", index, len(entries), entry.common_name)
        create_species_folder(entry, config.knowledge_dir)

    logger.info("Knowledge base build complete. Location: %s", config.knowledge_dir)


def main() -> None:
    """Entry point: build the knowledge base using default project paths."""
    project_root = Path(__file__).resolve().parents[2]
    config = BuilderConfig(
        classes_txt_path=project_root / "data" / "CUB_200_2011" / "classes.txt",
        knowledge_dir=project_root / "knowledge",
    )
    build_knowledge_base(config)


if __name__ == "__main__":
    main()
