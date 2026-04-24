from dataclasses import dataclass, field


@dataclass
class Step:
    """一個執行步驟：呼叫哪個 tool、帶什麼參數。"""
    tool: str    # nanobot 內的完整 tool 名稱，e.g. "mcp_intellinews_search_news_tool"
    params: dict # 固定值或 {placeholder}，placeholder 由 intent router 提取後填入


@dataclass
class Skill:
    """一個 skill 定義：觸發條件 + 執行步驟。"""
    name: str
    description: str
    triggers: list[str]                        # 自然語言觸發提示，給 LLM router 參考
    steps: list[Step]
    params_schema: dict = field(default_factory=dict)  # dynamic params 說明，給 router 提取用


@dataclass
class RouteMatch:
    """Intent router 匹配結果。"""
    skill: Skill
    params: dict   # 從用戶訊息提取的 dynamic params
