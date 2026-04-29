# RunbookEngine

RunbookEngine is a domain-aware skill routing framework built into the nanobot agent loop.
It solves a practical problem: **local 7–8B LLMs are unreliable at tool calling** (wrong
parameter formats, hallucinated values), but are capable enough at intent classification.

RunbookEngine separates the two concerns:
- LLM handles **"which skill does this map to?"**
- Deterministic code handles **"execute the steps"**

---

## How It Fits Into the Agent Loop

Every inbound message passes through a four-layer decision tree before reaching the
full agent loop:

```
User message
    │
    ▼  1. route()            — vector search (cosine ≥ threshold=0.40)
    │                          + gatekeeper relevance check
    │                          + LLM selector → execute skill
    │  (no match ↓)
    ▼  2. should_use_tools() — LLM asks: "does this need tools at all?"
    │  (NO ↓)
    ▼  base LLM              — free-form answer, no tools
    │  (YES ↓)
    ▼  3. force_route()      — all skills → LLM selector (no threshold)
    │  (no match ↓)
    ▼  4. full agent loop    — normal nanobot ReAct agent
```

Most queries are handled by layer 1 or 3. Layer 4 is the final fallback.

---

## Components

| Module | Role |
|--------|------|
| `loader` | Reads all `*.yaml` files from configured `definitions_dir` paths into `Skill` objects |
| `indexer` | Embeds each skill's `description + tags` text using the configured embedder; caches result as JSON; rebuilds only when YAML files are newer |
| `embedder` | Protocol + two implementations: `OllamaEmbedder` (calls `/api/embeddings`) and `SentenceTransformerEmbedder` |
| `retriever` | Embeds the user query; pure-Python cosine similarity against the index; returns top-N skills above threshold |
| `gatekeeper` | Two LLM calls: `is_relevant()` (are the top-N candidates actually relevant?) and `should_use_tools()` (does this message need tools at all?) |
| `selector` | LLM chooses one skill from candidates and extracts parameters; supports `forced=True` mode which skips exclusion heuristics |
| `executor` | Runs skill steps sequentially; resolves `{placeholder}` substitutions from params and prior step results (`save_as`) |
| `summarizer` | Wraps raw tool output into natural language; used when a skill has no `llm` steps of its own |
| `models` | `Skill`, `Step`, `RouteMatch` dataclasses |

---

## Skill YAML Format

```yaml
name: search_news            # unique skill identifier
category: 新聞查詢            # display grouping
description: >               # used for vector indexing and LLM prompts
  搜尋 IntelliNews 新聞資料庫，支援關鍵字或語意搜尋
tags: [新聞, 搜尋, 查詢, 關鍵字]   # also embedded for vector search
summary_hint: ""             # optional hint for the summarizer LLM

params_schema:
  query:
    description: 搜尋關鍵字或問題
    required: true
  mode:
    description: 搜尋模式
    required: false
    enum: [keyword, semantic]   # validated before execution; invalid value → user error message

steps:
  - tool: mcp_intellinews_search_news_tool   # calls a registered nanobot tool
    params:
      query: "{query}"          # {placeholder} resolved from params or prior save_as
      mode: "{mode}"
    save_as: search_result      # stores result in context for later steps

  - llm: extract               # inline LLM call
    input: "{search_result}"
    prompt: |
      請從以下 JSON 中提取標題和時間...
    save_as: extracted

  - llm: summarize             # another LLM call; last result is returned
    input: "{extracted}"
    prompt: 請用繁體中文整理...
```

### Step types

| Type | Key | Description |
|------|-----|-------------|
| `tool` | `tool:` | Calls a named nanobot tool. Result is JSON-serialised if dict/list, else string. |
| `llm` | `llm:` | Calls the LLM with `prompt + input`. Strips ` ```json ` fences automatically. |
| `builtin` | `builtin:` | Engine-internal operation. Currently only `set_domain` (switches active domain for a session). |

### `map` expansion

A param can define a `map` that expands one step into multiple parallel tool calls:

```yaml
params_schema:
  source:
    map:
      all: [pts, cna, udn]   # "all" → three separate tool calls

steps:
  - tool: mcp_crawl_tool
    params:
      spider: "{source}"     # expanded to pts / cna / udn when source=all
```

---

## Configuration (`runbook_engine.config.yaml`)

```yaml
engine:
  threshold: 0.40            # minimum cosine similarity for route() to consider a skill
  top_n: 3                   # max candidates passed to selector
  index_cache: ~/.cache/runbook_engine_index.json   # persisted vector index

domains:
  intellinews:
    display_name: IntelliNews 新聞系統
    description: 新聞搜尋、爬蟲管理、系統 log 查詢、觸發爬蟲
    definitions_dir: ./definitions/intellinews   # relative to the config file
```

Multiple domains can be defined. A session can be locked to a specific domain via
`engine.set_domain(session_key, "intellinews")`, which limits both vector search and
`force_route()` to that domain's skills (plus built-ins).

---

## Adding a New Skill

1. Create `definitions/<domain>/<skill_name>.yaml` following the schema above.
2. The vector index rebuilds automatically at next startup (or next `warm_up()` call)
   if the YAML is newer than the cache file.
3. No code changes required.

---

## Design Decisions

**Why not just use the LLM's native tool calling?**
Local 7–8B models produce garbled parameter values when given a list of tools to call
(observed with `llama3.1:8b` and `qwen2.5:7b`). Separating intent classification
(what skill?) from execution (run steps) makes the system reliable at this model size.

**Why two gatekeeper checks?**
- `is_relevant()` runs *after* vector search to catch false positives from embedding
  similarity (e.g. "weather news" matching a news-search skill).
- `should_use_tools()` runs *before* `force_route()` to avoid sending every casual
  message through an expensive all-skills selector.

**Why `forced=True` in `force_route()`?**
The standard selector prompt includes exclusion heuristics ("if asking about weather,
return null") which can misfire on queries like "weather-related news". `forced=True`
uses a simpler prompt: no exclusions, just yes/no.
