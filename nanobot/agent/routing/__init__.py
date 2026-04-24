from . import executor, intent, loader
from .models import RouteMatch, Skill, Step


class IntentRouter:
    def __init__(self, provider, model: str):
        self.skills = loader.load_all()
        self._provider = provider
        self._model = model

    async def route(self, user_message: str, tools) -> str | None:
        match = await intent.match(user_message, self.skills, self._provider, self._model)
        if match is None:
            return None
        return await executor.execute(match.skill, match.params, tools)


__all__ = ["IntentRouter", "Skill", "Step", "RouteMatch"]
