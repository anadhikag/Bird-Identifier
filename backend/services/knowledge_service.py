from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def parse_species_markdown(content: str) -> Dict[str, Any]:
    """Parse species markdown document into structured sections."""
    sections = {}
    current_section = None
    current_content = []
    
    for line in content.split('\n'):
        header_match = re.match(r'^##\s+(.+)$', line)
        if header_match:
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = header_match.group(1).strip()
            current_content = []
        elif line.strip() == '---':
            continue
        elif current_section:
            current_content.append(line)
            
    if current_section:
        sections[current_section] = '\n'.join(current_content).strip()
        
    # Get common name from first H1
    h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    common_name = h1_match.group(1).strip() if h1_match else ""
    
    # Parse sub-fields for Basic Information and Identification if they exist
    basic_info = {}
    if "Basic Information" in sections:
        for line in sections["Basic Information"].split('\n'):
            parts = line.split(':', 1)
            if len(parts) == 2:
                basic_info[parts[0].strip()] = parts[1].strip()
                
    identification = {}
    if "Identification" in sections:
        for line in sections["Identification"].split('\n'):
            parts = line.split(':', 1)
            if len(parts) == 2:
                identification[parts[0].strip()] = parts[1].strip()
                
    migration = {}
    if "Migration" in sections:
        for line in sections["Migration"].split('\n'):
            parts = line.split(':', 1)
            if len(parts) == 2:
                migration[parts[0].strip()] = parts[1].strip()

    conservation = {}
    if "Conservation" in sections:
        for line in sections["Conservation"].split('\n'):
            parts = line.split(':', 1)
            if len(parts) == 2:
                conservation[parts[0].strip()] = parts[1].strip()
                
    # Parse interesting facts into a list
    facts = []
    if "Interesting Facts" in sections:
        for line in sections["Interesting Facts"].split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('*'):
                facts.append(line[1:].strip())
            elif line:
                facts.append(line)
                
    return {
        "common_name": common_name,
        "scientific_name": basic_info.get("Scientific Name", ""),
        "family": basic_info.get("Family", ""),
        "order": basic_info.get("Order", ""),
        "identification": {
            "length": identification.get("Average Length", ""),
            "wingspan": identification.get("Average Wingspan", ""),
            "weight": identification.get("Average Weight", ""),
            "differences": identification.get("Male vs Female Differences", ""),
            "features": identification.get("Distinctive Features", "")
        },
        "habitat": sections.get("Habitat", ""),
        "geographic_distribution": sections.get("Geographic Distribution", ""),
        "migration": {
            "status": migration.get("Migratory Status", ""),
            "pattern": migration.get("Migration Pattern", ""),
            "raw": sections.get("Migration", "")
        },
        "diet": sections.get("Diet", ""),
        "behaviour": sections.get("Behaviour", ""),
        "vocalization": sections.get("Vocalization", ""),
        "breeding": sections.get("Breeding", ""),
        "conservation": {
            "status": conservation.get("IUCN Status", ""),
            "threats": conservation.get("Threats", ""),
            "raw": sections.get("Conservation", "")
        },
        "ecological_importance": sections.get("Ecological Importance", ""),
        "interesting_facts": facts
    }


class KnowledgeService:
    """Manages species information parsing, caching, and lookups."""
    
    def __init__(self, knowledge_dir: Optional[Path] = None) -> None:
        if knowledge_dir is None:
            project_root = Path(__file__).resolve().parents[2]
            knowledge_dir = project_root / "knowledge"
            
        self.knowledge_dir = knowledge_dir
        self.cache: Dict[str, Dict[str, Any]] = {}
        # Maps index (0-199) -> species_id (folder name)
        self.index_to_species_id: Dict[int, str] = {}
        
        self.load_and_cache_all()
        
    def load_and_cache_all(self) -> None:
        """Scan the knowledge directory, parse all species files, and cache them."""
        logger.info("Initializing KnowledgeService caching from %s", self.knowledge_dir)
        if not self.knowledge_dir.exists():
            raise FileNotFoundError(f"Knowledge directory not found at {self.knowledge_dir}")
            
        for folder in self.knowledge_dir.iterdir():
            if folder.is_dir() and "." in folder.name:
                # Find species.md
                md_file = folder / "species.md"
                if md_file.exists():
                    try:
                        content = md_file.read_text(encoding="utf-8")
                        parsed_data = parse_species_markdown(content)
                        species_id = folder.name
                        self.cache[species_id] = parsed_data
                        
                        # Extract folder index prefix to map index
                        prefix_str = folder.name.split(".", 1)[0]
                        try:
                            idx = int(prefix_str) - 1 # 0-indexed
                            self.index_to_species_id[idx] = species_id
                        except ValueError:
                            pass
                    except Exception as e:
                        logger.error("Failed to parse knowledge file for %s: %s", folder.name, e)
                        
        logger.info("KnowledgeService successfully cached %d species profiles.", len(self.cache))
        
    def get_species(self, species_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve structured species data by canonical ID (folder name)."""
        return self.cache.get(species_id)
        
    def get_species_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """Retrieve structured species data by model classification index (0-199)."""
        species_id = self.index_to_species_id.get(index)
        if species_id:
            return self.get_species(species_id)
        return None
        
    def get_species_id_by_index(self, index: int) -> Optional[str]:
        """Retrieve the canonical species ID by model classification index (0-199)."""
        return self.index_to_species_id.get(index)
