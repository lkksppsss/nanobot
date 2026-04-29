from pathlib import Path

import yaml

from .models import Skill, Step


def load_all(definitions_dirs: list[str], domain_map: dict[str, str] | None = None) -> list[Skill]:
    """從多個目錄載入所有 runbook YAML，每個 Skill 帶上所屬 domain key。

    domain_map: {definitions_dir → domain_key}，由 EngineConfig 提供。
    """
    domain_map = domain_map or {}
    skills = []
    for dir_str in definitions_dirs:
        path = Path(dir_str).expanduser()
        domain_key = domain_map.get(dir_str, "")
        for yaml_path in path.glob("*.yaml"):
            with yaml_path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)
            skill = _parse(data)
            skill.domain = domain_key
            skills.append(skill)
    return skills


def _parse(data: dict) -> Skill:
    steps = [_parse_step(s) for s in data.get("steps", [])]
    return Skill(
        name=data["name"],
        description=data["description"],
        steps=steps,
        category=data.get("category", ""),
        tags=data.get("tags", []),
        params_schema=data.get("params_schema", {}),
        summary_hint=data.get("summary_hint", ""),
    )


def _parse_step(s: dict) -> Step:
    if "llm" in s:
        return Step(
            kind="llm",
            prompt=s.get("prompt", ""),
            input=s.get("input", ""),
            save_as=s.get("save_as", ""),
        )
    if "builtin" in s:
        return Step(
            kind="builtin",
            tool=s["builtin"],
            params=s.get("params", {}),
        )
    return Step(
        kind="tool",
        tool=s["tool"],
        params=s.get("params", {}),
        save_as=s.get("save_as", ""),
    )
