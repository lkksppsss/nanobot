"""LLM-based selector: picks one runbook from top-N candidates, or lists options for the user."""

import json
import logging
import re
from dataclasses import dataclass

from .models import RouteMatch, Skill

logger = logging.getLogger(__name__)


def _extract_json(raw: str) -> dict | None:
    """從模型輸出中提取 JSON，處理小模型常見的格式問題。"""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # markdown code block：```json { ... } ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 前後有多餘文字：取第一個 { ... }
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None

_PROMPT = """\
用戶說：「{user_message}」

以下是可能相關的維運流程：
{candidates_text}

請判斷用戶是否明確要執行其中一個流程，並提取需要的參數。
若用戶的問題是閒聊、問天氣、問時間、或與以上流程完全無關，必須回傳 null。

若確定（含只有一個選項），直接回傳：{{"selected": "skill_name", "params": {{"param_name": "value"}}, "confident": true}}
若兩個以上都同樣可能，回傳：{{"selected": null, "candidates": ["skill_a", "skill_b"]}}
若完全無關，回傳：{{"selected": null, "candidates": []}}

只回 JSON，不要其他文字。"""

# forced 模式：向量搜尋未命中，跳過 threshold 直接給 selector
# 比普通模式更短更簡單：三選一 → 二選一，降低小模型負擔
_FORCED_PROMPT = """\
用戶說：「{user_message}」

以下是所有可用操作：
{candidates_text}

用戶是否【明確】要執行以上其中一個操作？若有任何不確定，直接回傳 null。

若確定，回傳：{{"selected": "skill_name", "params": {{"param_name": "value"}}, "confident": true}}
若不確定，回傳：{{"selected": null, "candidates": []}}

只回 JSON，不要其他文字。"""


@dataclass
class SelectResult:
    match: RouteMatch | None      # 確定選定的 skill + params
    ask_user: str | None = None   # 不確定時，回覆給用戶的選項訊息


async def select(
    user_message: str,
    candidates: list[Skill],
    provider,
    model: str,
    *,
    forced: bool = False,
) -> SelectResult:
    if not candidates:
        return SelectResult(match=None)

    lines = []
    for i, skill in enumerate(candidates, 1):
        tags_str = "、".join(skill.tags) if skill.tags else ""
        params_str = ""
        if skill.params_schema:
            parts = [
                f"{k}（{v.get('description', '')}）{'*必填' if v.get('required') else ''}"
                for k, v in skill.params_schema.items()
            ]
            params_str = f"，需要參數：{', '.join(parts)}"
        lines.append(f"{i}. {skill.name}：{skill.description}（tags: {tags_str}）{params_str}")

    template = _FORCED_PROMPT if forced else _PROMPT
    prompt = template.format(
        user_message=user_message,
        candidates_text="\n".join(lines),
    )

    try:
        response = await provider.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=256,
            temperature=0.0,
        )
        raw = (response.content or "").strip()
        data = _extract_json(raw)
        if data is None:
            logger.debug("selector: cannot extract JSON from: %r", raw[:100])
            return SelectResult(match=None)
    except Exception as e:
        logger.debug("selector failed: %s", e)
        return SelectResult(match=None)

    selected_name = data.get("selected")

    if selected_name:
        skill = next((s for s in candidates if s.name == selected_name), None)
        if skill is None:
            logger.debug("selector returned unknown skill: %s", selected_name)
            return SelectResult(match=None)
        return SelectResult(match=RouteMatch(skill=skill, params=data.get("params", {})))

    # 不確定，列選項給用戶
    chosen_names = data.get("candidates", [])
    if not chosen_names:
        return SelectResult(match=None)

    chosen = [s for s in candidates if s.name in chosen_names]
    options = "\n".join(f"{i}. {s.description}（`{s.name}`）" for i, s in enumerate(chosen, 1))
    return SelectResult(match=None, ask_user=f"找到幾個可能相關的流程，請確認要執行哪一個：\n{options}")
