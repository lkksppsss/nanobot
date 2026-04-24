from pathlib import Path

import yaml

from .models import Skill, Step

_DEFINITIONS_DIR = Path(__file__).parent / "definitions"


def load_all() -> list[Skill]:
    skills = []
    for path in _DEFINITIONS_DIR.glob("*.yaml"):
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        skills.append(_parse(data))
    return skills


def _parse(data: dict) -> Skill:
    steps = [Step(tool=s["tool"], params=s.get("params", {})) for s in data.get("steps", [])]
    return Skill(
        name=data["name"],
        description=data["description"],
        triggers=data.get("triggers", []),
        params_schema=data.get("params_schema", {}),
        steps=steps,
    )
