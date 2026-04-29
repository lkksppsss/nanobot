import logging

logger = logging.getLogger(__name__)

_PROMPT = """\
用戶問：「{user_message}」

以下是執行結果：
{results_text}
{hint_section}
請用繁體中文、自然語言整理以上結果，回答用戶的問題。簡潔清楚，不需重複完整 JSON 原始資料，摘要重點即可。
重要：若結果為空或明確說明找不到，直接告知用戶查無結果，絕對不可自行捏造資料。"""


async def summarize(
    user_message: str,
    results: list[tuple[str, str]],
    provider,
    model: str,
    hint: str = "",
) -> str:
    """整理多個 skill 的 raw results，回傳自然語言摘要。LLM 失敗時 fallback 回 raw results。"""
    results_text = "\n\n".join(
        f"[{skill_name} 結果]\n{raw}" for skill_name, raw in results
    )
    hint_section = f"\n整理重點：\n{hint}" if hint else ""
    prompt = _PROMPT.format(user_message=user_message, results_text=results_text, hint_section=hint_section)

    try:
        response = await provider.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=1024,
            temperature=0.3,
        )
        return (response.content or "").strip()
    except Exception as e:
        logger.debug("summarize failed: %s", e)
        return results_text
