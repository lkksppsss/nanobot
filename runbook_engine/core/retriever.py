"""Embed user query and return top-N matching skills by cosine similarity.

Pure Python math — no numpy dependency, works with any Embedder implementation.
"""

import json
import logging
import math
from pathlib import Path

from .embedder import Embedder
from .models import SYSTEM_DOMAIN, Skill

logger = logging.getLogger(__name__)

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def retrieve(
    user_message: str,
    skills: list[Skill],
    embedder: Embedder,
    index_path: Path,
    top_n: int = 3,
    threshold: float = 0.40,
    domain_filter: str | None = None,
) -> list[Skill]:
    """Return top-N skills whose index vectors are closest to user_message.

    domain_filter: 若指定，只搜尋該 domain 的 skills。
    """
    index = json.loads(index_path.read_text(encoding="utf-8"))

    filtered_skills = (
        skills if not domain_filter
        else [s for s in skills if s.domain in (domain_filter, SYSTEM_DOMAIN)]
    )
    skill_map = {s.name: s for s in filtered_skills}

    query_vec = await embedder.embed(user_message)

    scores: list[tuple[float, Skill]] = []
    for entry in index:
        name = entry["name"]
        if name not in skill_map:
            continue
        score = _cosine(query_vec, entry["vector"])
        scores.append((score, skill_map[name]))

    scores.sort(key=lambda x: x[0], reverse=True)

    results = [skill for score, skill in scores[:top_n] if score >= threshold]
    logger.debug(
        "retrieve top-%d (domain=%s, threshold=%.2f): %s",
        top_n, domain_filter or "all", threshold, [s.name for s in results],
    )
    return results
