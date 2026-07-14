"""Sub-agent system — fork-return pattern with built-in + custom agent types.
Mirrors Claude Code's AgentTool: explore (read-only), plan (structured), general (full tools),
plus user-defined agents via .claude/agents/*.md."""

# =============================================================================
# 子代理系统模块 (Sub-agent System)
#
# 本模块实现了 Mini Claude Code 的子代理系统，采用"分叉-返回"(fork-return) 模式。
# 它镜像了 Claude Code 的 AgentTool 功能，支持以下内置代理类型：
#   - explore（探索型）：只读模式，用于快速搜索和浏览代码库
#   - plan（计划型）：只读模式，用于分析代码并生成结构化的实现计划
#   - general（通用型）：具备完整工具集，可执行独立任务
#
# 此外，用户可以通过在 .claude/agents/ 目录下创建 Markdown 文件来自定义代理类型，
# 每个 .md 文件定义一个自定义代理，包含名称、描述、允许使用的工具和系统提示词。
# =============================================================================

from __future__ import annotations

from pathlib import Path

from .frontmatter import parse_frontmatter
from .tools import tool_definitions, ToolDef

# ─── Read-only tools (for explore and plan agents) ──────────
# 只读工具集合：供 explore 和 plan 代理使用，这些代理只能读取和搜索，不能修改文件
READ_ONLY_TOOLS = {"read_file", "list_files", "grep_search"}

# EXPLORE_PROMPT：探索型代理的系统提示词
# 该代理专门用于快速搜索和浏览代码库，严格限制为只读模式，
# 禁止任何文件创建、修改或删除操作。
EXPLORE_PROMPT = """You are a file search specialist for Mini Claude Code. You excel at thoroughly navigating and exploring codebases.

=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===
This is a READ-ONLY exploration task. You are STRICTLY PROHIBITED from:
- Creating new files (no write_file, touch, or file creation of any kind)
- Modifying existing files (no edit_file operations)
- Deleting files (no rm or deletion)
- Running ANY commands that change system state

Your role is EXCLUSIVELY to search and analyze existing code.

Your strengths:
- Rapidly finding files using glob patterns
- Searching code and text with powerful regex patterns
- Reading and analyzing file contents

Guidelines:
- Use list_files for broad file pattern matching
- Use grep_search for searching file contents with regex
- Use read_file when you know the specific file path you need to read
- Adapt your search approach based on the thoroughness level specified by the caller

NOTE: You are meant to be a fast agent that returns output as quickly as possible. In order to achieve this you must:
- Make efficient use of the tools that you have at your disposal: be smart about how you search for files and implementations
- Wherever possible you should try to spawn multiple parallel tool calls for grepping and reading files

Complete the user's search request efficiently and report your findings clearly."""

# PLAN_PROMPT：计划型代理的系统提示词
# 该代理专注于分析代码库架构并设计结构化的实现计划，
# 同样为只读模式，不会修改任何文件。
PLAN_PROMPT = """You are a Plan agent — a READ-ONLY sub-agent specialized for designing implementation plans.

IMPORTANT CONSTRAINTS:
- You are READ-ONLY. You only have access to read_file, list_files, and grep_search.
- Do NOT attempt to modify any files.

Your job:
- Analyze the codebase to understand the current architecture
- Design a step-by-step implementation plan
- Identify critical files that need modification
- Consider architectural trade-offs

Return a structured plan with:
1. Summary of current state
2. Step-by-step implementation steps
3. Critical files for implementation
4. Potential risks or considerations"""

# GENERAL_PROMPT：通用型代理的系统提示词
# 该代理具备完整的工具集（除 agent 工具本身外），可执行独立任务，
# 包括文件搜索、多文件分析、复杂问题调查和多步骤研究任务。
GENERAL_PROMPT = """You are an agent for Mini Claude Code. Given the user's message, you should use the tools available to complete the task. Complete the task fully—don't gold-plate, but don't leave it half-done. When you complete the task, respond with a concise report covering what was done and any key findings — the caller will relay this to the user, so it only needs the essentials.

Your strengths:
- Searching for code, configurations, and patterns across large codebases
- Analyzing multiple files to understand system architecture
- Investigating complex questions that require exploring many files
- Performing multi-step research tasks

Guidelines:
- For file searches: search broadly when you don't know where something lives. Use read_file when you know the specific file path.
- For analysis: Start broad and narrow down. Use multiple search strategies if the first doesn't yield results.
- Be thorough: Check multiple locations, consider different naming conventions, look for related files.
- NEVER create files unless they're absolutely necessary for achieving your goal. ALWAYS prefer editing an existing file to creating a new one."""

# ─── Custom agent discovery ─────────────────────────────────
# 自定义代理发现机制相关的缓存和函数

# _cached_custom_agents：自定义代理的缓存字典
# 避免每次调用时都重新扫描文件系统，首次加载后会缓存结果
_cached_custom_agents: dict[str, dict] | None = None


def _discover_custom_agents() -> dict[str, dict]:
    """发现并加载所有自定义代理配置。

    扫描两个位置的 .claude/agents/ 目录来查找自定义代理定义文件（.md 格式）：
    1. 用户级别 (~/.claude/agents/)：较低优先级
    2. 项目级别 (<项目根目录>/.claude/agents/)：较高优先级，会覆盖同名的用户级代理

    返回值：
        dict[str, dict]: 代理名称到代理配置字典的映射。
            每个配置字典包含以下键：
            - name (str): 代理名称
            - description (str): 代理描述
            - allowed_tools (list[str] | None): 允许使用的工具列表，None 表示使用默认工具集
            - system_prompt (str): 代理的系统提示词（Markdown 文件的正文部分）
    """
    global _cached_custom_agents
    # 如果缓存已存在，直接返回缓存结果，避免重复扫描文件系统
    if _cached_custom_agents is not None:
        return _cached_custom_agents

    agents: dict[str, dict] = {}
    # User-level (lower priority)  用户级配置，优先级较低
    _load_agents_from_dir(Path.home() / ".claude" / "agents", agents)
    # Project-level (higher priority, overwrites)  项目级配置，优先级较高，会覆盖同名代理
    _load_agents_from_dir(Path.cwd() / ".claude" / "agents", agents)

    _cached_custom_agents = agents
    return agents


def _load_agents_from_dir(directory: Path, agents: dict[str, dict]) -> None:
    """从指定目录加载自定义代理配置。

    扫描给定目录下所有 .md 文件，解析每个文件的 frontmatter 元数据和正文内容，
    将其转换为代理配置并存入 agents 字典中。

    参数：
        directory (Path): 要扫描的目录路径
        agents (dict[str, dict]): 代理配置字典，加载的配置会写入此字典

    返回值：
        None（直接修改传入的 agents 字典）
    """
    # 目录不存在则直接返回
    if not directory.is_dir():
        return
    for entry in directory.iterdir():
        # 只处理 .md 后缀的文件
        if not entry.suffix == ".md":
            continue
        try:
            raw = entry.read_text()
            # 解析 frontmatter（YAML 格式的元数据头部）和正文
            result = parse_frontmatter(raw)
            meta = result.meta
            # 优先使用 frontmatter 中的 name 字段，否则使用文件名（不含扩展名）作为代理名称
            name = meta.get("name") or entry.stem
            allowed_tools = None
            # 如果 frontmatter 中指定了 allowed-tools，按逗号分隔解析为工具列表
            if "allowed-tools" in meta:
                allowed_tools = [s.strip() for s in meta["allowed-tools"].split(",")]
            # 构建代理配置字典
            agents[name] = {
                "name": name,
                "description": meta.get("description", ""),
                "allowed_tools": allowed_tools,
                "system_prompt": result.body,  # Markdown 正文作为系统提示词
            }
        except Exception:
            # 解析失败时静默跳过，不影响其他代理的加载
            pass


# ─── Main config function ───────────────────────────────────
# 主配置函数：根据代理类型返回对应的系统提示词和工具列表


def get_sub_agent_config(agent_type: str) -> dict:
    """Return {system_prompt, tools} for the given agent type."""
    # 根据代理类型获取对应的配置：系统提示词和可用工具列表。
    #
    # 参数：
    #     agent_type (str): 代理类型名称。可以是：
    #         - "explore"：探索型代理（只读，用于搜索和浏览代码）
    #         - "plan"：计划型代理（只读，用于生成实现计划）
    #         - "general"：通用型代理（完整工具集，可执行独立任务）
    #         - 或任意自定义代理名称（通过 .claude/agents/*.md 定义）
    #
    # 返回值：
    #     dict: 包含以下键的配置字典：
    #         - system_prompt (str): 代理的系统提示词
    #         - tools (list[ToolDef]): 代理可用的工具定义列表

    # 首先检查是否为自定义代理类型
    custom = _discover_custom_agents().get(agent_type)
    if custom:
        # 自定义代理：如果指定了 allowed_tools 则只包含允许的工具，否则排除 agent 工具
        if custom["allowed_tools"]:
            tools = [t for t in tool_definitions if t["name"] in custom["allowed_tools"]]
        else:
            tools = [t for t in tool_definitions if t["name"] != "agent"]
        return {"system_prompt": custom["system_prompt"], "tools": tools}

    # 构建只读工具列表（用于 explore 和 plan 代理）
    read_only = [t for t in tool_definitions if t["name"] in READ_ONLY_TOOLS]

    # 根据内置代理类型返回对应的配置
    if agent_type == "explore":
        return {"system_prompt": EXPLORE_PROMPT, "tools": read_only}
    elif agent_type == "plan":
        return {"system_prompt": PLAN_PROMPT, "tools": read_only}
    else:  # general — 通用代理，使用除 agent 外的所有工具
        return {"system_prompt": GENERAL_PROMPT, "tools": [t for t in tool_definitions if t["name"] != "agent"]}


# ─── Available agent types (for system prompt) ──────────────
# 可用代理类型列表：用于生成系统提示词中的代理说明


def get_available_agent_types() -> list[dict[str, str]]:
    """获取所有可用的代理类型列表。

    返回包含内置代理类型和所有自定义代理类型的列表，
    用于在系统提示词中向用户展示可用的代理选项。

    返回值：
        list[dict[str, str]]: 代理类型信息列表，每个元素包含：
            - name (str): 代理类型名称
            - description (str): 代理类型的描述说明
    """
    # 定义三个内置代理类型
    types = [
        {"name": "explore", "description": "Fast, read-only codebase search and exploration"},
        {"name": "plan", "description": "Read-only analysis with structured implementation plans"},
        {"name": "general", "description": "Full tools for independent tasks"},
    ]
    # 追加所有自定义代理类型
    for name, defn in _discover_custom_agents().items():
        types.append({"name": name, "description": defn["description"]})
    return types


def build_agent_descriptions() -> str:
    """构建自定义代理类型的描述文本。

    如果只有内置的三个代理类型（explore、plan、general），则返回空字符串，
    因为内置类型已在系统提示词中说明。当存在自定义代理时，生成格式化的描述文本。

    返回值：
        str: 自定义代理类型的 Markdown 格式描述文本，无自定义代理时返回空字符串
    """
    types = get_available_agent_types()
    # 如果只有内置类型（3个），不需要额外描述，返回空字符串
    if len(types) <= 3:
        return ""  # Only built-in types, already in system prompt

    # 提取自定义代理类型（跳过前3个内置类型）
    custom = types[3:]
    lines = ["\n# Custom Agent Types", ""]
    # 为每个自定义代理生成 Markdown 格式的列表项
    for t in custom:
        lines.append(f"- **{t['name']}**: {t['description']}")
    return "\n".join(lines)


def reset_agent_cache() -> None:
    """重置自定义代理的缓存。

    清除已缓存的自定义代理配置，下次调用 _discover_custom_agents() 时
    将重新扫描文件系统加载最新的代理定义。
    通常在代理定义文件被修改后调用此函数以刷新缓存。

    返回值：
        None
    """
    global _cached_custom_agents
    _cached_custom_agents = None
