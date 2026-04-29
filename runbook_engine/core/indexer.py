"""Build and manage the vector index for runbook definitions.

Index is stored as routing_index.json next to the YAML definitions.
Rebuilt automatically when any YAML is newer than the index file.
"""

import json
import logging
from pathlib import Path

from .embedder import Embedder
from .models import Skill

logger = logging.getLogger(__name__)


def _index_text(skill: Skill) -> str:
    tags = " ".join(skill.tags)
    return f"{skill.description} {tags}".strip()


def _needs_rebuild(index_path: Path, definitions_dirs: list[str]) -> bool:
    if not index_path.exists():
        return True
    index_mtime = index_path.stat().st_mtime
    for dir_str in definitions_dirs:
        d = Path(dir_str).expanduser()
        if any(p.stat().st_mtime > index_mtime for p in d.glob("*.yaml")):
            return True
    return False


async def build(skills: list[Skill], embedder: Embedder, index_path: Path) -> None:
    """Embed all skills and write index JSON."""
    index = []
    for skill in skills:
        vector = await embedder.embed(_index_text(skill))
        index.append({"name": skill.name, "domain": skill.domain, "vector": vector})
        logger.debug("indexed skill: %s (domain=%s)", skill.name, skill.domain)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    logger.info("runbook index built: %d skills → %s", len(index), index_path)


async def ensure_index(
    skills: list[Skill],
    embedder: Embedder,
    index_path: Path,
    definitions_dirs: list[str],
) -> None:
    """Rebuild index only if any YAML definition is newer than the cached index."""
    if _needs_rebuild(index_path, definitions_dirs):
        logger.info("runbook index outdated or missing, rebuilding...")
        await build(skills, embedder, index_path)
