"""RunbookEngine configuration — loaded from runbook_engine.config.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DomainConfig:
    display_name: str
    description: str
    definitions_dir: str


@dataclass
class EngineConfig:
    domains: dict[str, DomainConfig] = field(default_factory=dict)
    threshold: float = 0.40
    top_n: int = 3
    index_cache: str = "~/.cache/runbook_index.json"

    @classmethod
    def load(cls, path: str | Path) -> "EngineConfig":
        path = Path(path).expanduser()
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        engine_cfg = data.get("engine", {})
        domains = {
            key: DomainConfig(
                display_name=val["display_name"],
                description=val["description"],
                definitions_dir=str(Path(path.parent / val["definitions_dir"])),
            )
            for key, val in data.get("domains", {}).items()
        }
        return cls(
            domains=domains,
            threshold=engine_cfg.get("threshold", 0.40),
            top_n=engine_cfg.get("top_n", 3),
            index_cache=engine_cfg.get("index_cache", "~/.cache/runbook_index.json"),
        )

    def all_definitions_dirs(self) -> list[str]:
        return [d.definitions_dir for d in self.domains.values()]

    def system_description(self) -> str:
        """給 gatekeeper 用的整體系統描述。"""
        if not self.domains:
            return "維運助理"
        parts = [f"{d.display_name}（{d.description}）" for d in self.domains.values()]
        return "；".join(parts)
