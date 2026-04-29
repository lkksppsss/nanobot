import logging

logger = logging.getLogger(__name__)

_RELEVANCE_PROMPT = """\
用戶說：「{user_message}」

以下是可能相關的維運流程：
{skills_text}

用戶是否要【立即執行】以上任一操作？詢問定義、解釋概念、一般問題 → NO。只回 YES 或 NO。"""

_TOOLS_PROMPT = """\
用戶說：「{user_message}」

這是一個維運助理，支援以下系統：
{system_description}

判斷標準：
- 用戶明確要執行以上系統的操作 → YES
- 用戶只是閒聊、問天氣、問時間、或與以上系統完全無關 → NO

只回 YES 或 NO。"""


async def should_use_tools(
    user_message: str,
    provider,
    model: str,
    system_description: str = "",
) -> bool:
    prompt = _TOOLS_PROMPT.format(
        user_message=user_message,
        system_description=system_description or "維運助理",
    )
    try:
        response = await provider.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=5,
            temperature=0.0,
        )
        answer = (response.content or "").strip().upper()
        logger.debug("tool_gatekeeper: %r → %s", user_message[:40], answer)
        return answer.startswith("Y")
    except Exception as e:
        logger.debug("tool_gatekeeper failed: %s", e)
        return True


async def is_relevant(user_message: str, candidates, provider, model: str) -> bool:
    skills_text = "\n".join(f"- {s.name}：{s.description}" for s in candidates)
    prompt = _RELEVANCE_PROMPT.format(user_message=user_message, skills_text=skills_text)
    try:
        response = await provider.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=5,
            temperature=0.0,
        )
        answer = (response.content or "").strip().upper()
        logger.debug("gatekeeper: %r → %s", user_message[:40], answer)
        return answer.startswith("Y")
    except Exception as e:
        logger.debug("gatekeeper failed: %s", e)
        return True
