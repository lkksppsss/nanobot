import json
import logging
import re
from .models import Skill, Step

logger = logging.getLogger(__name__)


async def execute(skill: Skill, params: dict, tools, provider=None, model: str = "") -> str:
    context = {}  # 存 save_as 結果，供後續步驟以 {key} 引用
    tool_results: list[str] = []
    last_result = ""

    for step in _expand_steps(skill, params):
        if step.kind == "llm":
            logger.debug("llm step save_as=%r input=%r", step.save_as, _resolve_str(step.input, params, context)[:200])
            result = await _run_llm_step(step, params, context, provider, model)
            logger.debug("llm step result: %r", result[:300])
        else:
            resolved = _resolve_params(step.params, params, context)
            logger.debug("tool step %s params=%s save_as=%r", step.tool, resolved, step.save_as)
            raw = await tools.execute(step.tool, resolved)
            result = json.dumps(raw, ensure_ascii=False) if isinstance(raw, (dict, list)) else str(raw)
            logger.debug("tool step result: %r", result[:200])
            tool_results.append(result)

        if step.save_as:
            context[step.save_as] = result
        last_result = result

    # map-expanded multi-tool steps (no LLM in the mix): combine so summarizer sees everything
    # skills with LLM steps handle their own aggregation; return last_result (the LLM summary)
    if len(tool_results) > 1 and not skill.has_llm_steps:
        return "\n".join(tool_results)
    return last_result


async def _run_llm_step(step: Step, params: dict, context: dict, provider, model: str) -> str:
    input_text = _resolve_str(step.input, params, context)
    content = f"{step.prompt}\n\n{input_text}" if input_text else step.prompt
    try:
        response = await provider.chat(
            messages=[{"role": "user", "content": content}],
            model=model,
            max_tokens=1024,
            temperature=0.0,
        )
        return _strip_code_fence((response.content or "").strip())
    except Exception as e:
        logger.warning("llm step '%s' failed: %s", step.save_as, e)
        return f"[error: {e}]"


def _strip_code_fence(text: str) -> str:
    """移除 LLM 常見的 ```json ... ``` 包裝。"""
    if text.startswith("```"):
        lines = text.splitlines()
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        return "\n".join(lines[1:end]).strip()
    return text


def _expand_steps(skill: Skill, params: dict) -> list[Step]:
    expanded = []
    for step in skill.steps:
        if step.kind == "llm":
            expanded.append(step)
        else:
            expanded.extend(_try_expand(step, skill.params_schema, params))
    return expanded


def _try_expand(step: Step, params_schema: dict, params: dict) -> list[Step]:
    for param_name, schema in params_schema.items():
        if "map" not in schema:
            continue
        placeholder = f"{{{param_name}}}"
        if not any(v == placeholder for v in step.params.values()):
            continue
        value = params.get(param_name, "")
        mapped_values = schema["map"].get(value, [])
        if not mapped_values:
            break
        return [
            Step(
                kind="tool",
                tool=step.tool,
                params={k: mv if v == placeholder else v for k, v in step.params.items()},
            )
            for mv in mapped_values
        ]
    return [step]


def _resolve_params(template: dict, params: dict, context: dict) -> dict:
    return {k: _resolve_str(v, params, context) for k, v in template.items()}


def _resolve_str(value, params: dict, context: dict):
    if not isinstance(value, str):
        return value
    # 整個字串是單一 {key}
    if value.startswith("{") and value.endswith("}") and value.count("{") == 1:
        key = value[1:-1]
        return context.get(key) or params.get(key, "")
    # 字串內嵌多個 {key}，逐一替換
    return re.sub(
        r"\{(\w+)\}",
        lambda m: str(context.get(m.group(1)) or params.get(m.group(1)) or ""),
        value,
    )
