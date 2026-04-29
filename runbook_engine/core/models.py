from dataclasses import dataclass, field

SYSTEM_DOMAIN = "__system__"


@dataclass
class Step:
    kind: str = "tool"                              # "tool" | "llm"
    tool: str = ""                                  # kind=tool：nanobot tool 名稱
    params: dict = field(default_factory=dict)      # kind=tool：固定值或 {placeholder}
    prompt: str = ""                                # kind=llm：給 LLM 的指令
    input: str = ""                                 # kind=llm：輸入內容，可引用 {save_as} 變數
    save_as: str = ""                               # 把結果存進 context，供後續步驟引用


@dataclass
class Skill:
    name: str
    description: str
    steps: list[Step]
    category: str = ""
    tags: list[str] = field(default_factory=list)
    params_schema: dict = field(default_factory=dict)
    summary_hint: str = ""                          # 無 llm steps 的 skill 用，給外層 summarizer
    domain: str = ""                                # 所屬 domain key（來自 EngineConfig.domains）

    @property
    def has_llm_steps(self) -> bool:
        return any(s.kind == "llm" for s in self.steps)


@dataclass
class RouteMatch:
    skill: Skill
    params: dict
