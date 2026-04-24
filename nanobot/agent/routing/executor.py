from .models import Skill


async def execute(skill: Skill, params: dict, tools) -> str:
    results = []
    for step in skill.steps:
        resolved = _resolve_params(step.params, params)
        result = await tools.execute(step.tool, resolved)
        results.append(str(result))
    return "\n\n".join(results)


def _resolve_params(template: dict, dynamic: dict) -> dict:
    out = {}
    for k, v in template.items():
        if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
            out[k] = dynamic.get(v[1:-1], "")
        else:
            out[k] = v
    return out
