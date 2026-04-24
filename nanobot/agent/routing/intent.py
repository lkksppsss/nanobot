import json
import logging

from .models import RouteMatch, Skill

logger = logging.getLogger(__name__)

_PROMPT = """\
你是指令分發器。根據用戶訊息判斷要執行哪個 skill。

可用 skill：
{skills_list}

用戶說：「{user_message}」

若明確符合，回傳 JSON：
{{"skill": "skill_name", "params": {{"param_name": "value"}}}}

若無法確定或是一般對話，回傳：
{{"skill": null}}

只回 JSON，不要其他文字。"""


async def match(user_message: str, skills: list[Skill], provider, model: str) -> RouteMatch | None:
    skills_list = "\n".join(
        f"- {s.name}：{s.description}（觸發詞：{', '.join(s.triggers)}）"
        for s in skills
    )
    prompt = _PROMPT.format(skills_list=skills_list, user_message=user_message)

    try:
        response = await provider.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=256,
            temperature=0.0,
        )
        raw = (response.content or "").strip()
        data = json.loads(raw)
    except Exception as e:
        logger.debug("intent match failed: %s", e)
        return None

    skill_name = data.get("skill")
    if not skill_name:
        return None

    skill = next((s for s in skills if s.name == skill_name), None)
    if skill is None:
        logger.debug("intent matched unknown skill: %s", skill_name)
        return None

    return RouteMatch(skill=skill, params=data.get("params", {}))
