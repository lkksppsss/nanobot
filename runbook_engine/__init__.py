"""RunbookEngine — generic runbook routing framework for AI agents."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from .config import EngineConfig
from .core import executor, gatekeeper, indexer, loader, retriever, selector, summarizer
from .core.embedder import Embedder, OllamaEmbedder
from .core.models import SYSTEM_DOMAIN, RouteMatch, Skill, Step

_BUILTINS_DIR = str(Path(__file__).parent / "builtins")


class RunbookEngine:
    """Domain-aware runbook routing engine.

    Usage:
        config = EngineConfig.load("runbook_engine.config.yaml")
        engine = RunbookEngine(provider, model, config, embedder=OllamaEmbedder())
        await engine.warm_up()  # build vector index at startup

        # In agent loop:
        result = await engine.route(user_message, tools, session_key="discord:user123")

    embedder=None:
        route() (vector search) is skipped; flow falls through directly to force_route().
        Useful when Ollama or a compatible embedding service is unavailable.
    """

    def __init__(
        self,
        provider,
        model: str,
        config: EngineConfig,
        embedder: Embedder | None = None,
    ):
        self._provider = provider
        self._model = model
        self._config = config
        self._embedder = embedder  # None = no vector search, route() becomes a no-op
        self._index_path = Path(config.index_cache).expanduser()

        domain_map = {
            d.definitions_dir: key
            for key, d in config.domains.items()
        }
        domain_map[_BUILTINS_DIR] = SYSTEM_DOMAIN
        all_dirs = [_BUILTINS_DIR] + config.all_definitions_dirs()
        self._skills = loader.load_all(all_dirs, domain_map)
        self._all_dirs = all_dirs
        self._index_ready = False
        self._session_domain: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_domain(self, session_key: str, domain: str) -> None:
        """明確指定 session 的 active domain。"""
        self._session_domain[session_key] = domain

    def get_domain(self, session_key: str) -> str | None:
        return self._session_domain.get(session_key)

    async def warm_up(self) -> None:
        """在啟動時預先建立 vector index，避免第一個請求承擔延遲。"""
        await self._ensure_index()

    async def should_use_tools(self, user_message: str) -> bool:
        return await gatekeeper.should_use_tools(
            user_message,
            self._provider,
            self._model,
            system_description=self._config.system_description(),
        )

    async def route(self, user_message: str, tools, session_key: str = "") -> str | None:
        """向量搜尋路由。無 embedder 時直接返回 None，讓 flow 繼續到 force_route。"""
        if self._embedder is None:
            return None

        await self._ensure_index()
        domain_filter = self._session_domain.get(session_key)

        candidates = await retriever.retrieve(
            user_message,
            self._skills,
            self._embedder,
            index_path=self._index_path,
            top_n=self._config.top_n,
            threshold=self._config.threshold,
            domain_filter=domain_filter,
        )
        if not candidates:
            return None

        if not await gatekeeper.is_relevant(user_message, candidates, self._provider, self._model):
            return None

        result = await selector.select(user_message, candidates, self._provider, self._model)
        return await self._execute(user_message, result, tools, session_key)

    async def force_route(self, user_message: str, tools, session_key: str = "") -> str | None:
        """全 skill 給 selector，跳過 threshold。
        不呼叫 _ensure_index()：force_route 不使用向量 index，直接用已載入的 skill list。
        """
        domain_filter = self._session_domain.get(session_key)
        candidates = (
            [s for s in self._skills if s.domain in (domain_filter, SYSTEM_DOMAIN)]
            if domain_filter
            else self._skills
        )
        result = await selector.select(
            user_message, candidates, self._provider, self._model, forced=True
        )
        return await self._execute(user_message, result, tools, session_key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_index(self) -> None:
        if self._embedder is None or self._index_ready:
            return
        await indexer.ensure_index(
            self._skills,
            self._embedder,
            self._index_path,
            self._all_dirs,
        )
        self._index_ready = True

    async def _execute(
        self, user_message: str, result: selector.SelectResult, tools, session_key: str = ""
    ) -> str | None:
        if result.ask_user:
            return result.ask_user
        if result.match is None:
            return None

        if any(s.kind == "builtin" for s in result.match.skill.steps):
            return self._run_builtins(result.match.skill, result.match.params, session_key)

        logger.warning("[runbook] skill=%s params=%s", result.match.skill.name, result.match.params)

        if msg := _validate_params(result.match.skill, result.match.params):
            return msg

        raw = await executor.execute(
            result.match.skill, result.match.params, tools,
            provider=self._provider, model=self._model,
        )
        if result.match.skill.has_llm_steps:
            return raw
        return await summarizer.summarize(
            user_message,
            [(result.match.skill.name, raw)],
            self._provider,
            self._model,
            hint=result.match.skill.summary_hint,
        )

    def _run_builtins(self, skill: Skill, params: dict, session_key: str) -> str:
        for step in skill.steps:
            if step.kind != "builtin":
                continue
            if step.tool == "set_domain":
                domain = params.get("domain", "")
                self.set_domain(session_key, domain)
                cfg = self._config.domains.get(domain)
                display = cfg.display_name if cfg else domain
                return f"已切換到 {display}"
        return ""


def _validate_params(skill: Skill, params: dict) -> str | None:
    for k, v in skill.params_schema.items():
        val = params.get(k)
        if v.get("required") and not val:
            return f"請提供以下資訊才能執行：{k}（{v.get('description', '')}）"
        enum = v.get("enum")
        if enum and val and val not in enum:
            return f"「{val}」不是有效的 {k}，有效選項：{', '.join(enum)}"
    return None


__all__ = [
    "RunbookEngine",
    "EngineConfig",
    "Skill", "Step", "RouteMatch",
    "Embedder", "OllamaEmbedder",
    "SYSTEM_DOMAIN",
]
