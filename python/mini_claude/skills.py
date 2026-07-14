"""Skills system — discover, parse, and execute .claude/skills/*/SKILL.md
Mirrors Claude Code's skill architecture: frontmatter metadata + prompt templates."""

# 技能系统模块
# 本模块实现了技能（Skill）的发现、解析和执行功能，是对 Claude Code 技能架构的精简复刻。
# 技能定义存放在 .claude/skills/*/SKILL.md 文件中，每个技能由 frontmatter 元数据和提示词模板组成。
# 核心流程：
#   1. discover_skills() — 从用户级和项目级目录中扫描并加载所有技能定义
#   2. _parse_skill_file() — 解析 SKILL.md 文件，提取元数据和提示词模板
#   3. execute_skill() — 根据技能名称查找技能并生成可执行的提示词
#   4. build_skill_descriptions() — 构建系统提示词中关于可用技能的描述文本

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .frontmatter import parse_frontmatter

# ─── Types ──────────────────────────────────────────────────


# 技能定义数据类
# 表示一个完整的技能定义，包含元数据和提示词模板。
# 技能可以由用户手动调用（user_invocable=True），也可以由系统自动触发。
@dataclass
class SkillDefinition:
    name: str                                  # 技能名称，用于标识和调用
    description: str                           # 技能的功能描述
    when_to_use: str | None = None             # 触发条件描述，说明何时应使用该技能
    allowed_tools: list[str] | None = None     # 该技能允许使用的工具列表，None 表示不限制
    user_invocable: bool = True                # 是否允许用户手动调用（通过 /<name> 命令）
    context: str = "inline"                    # 上下文模式："inline"（内联执行）或 "fork"（分叉执行）
    prompt_template: str = ""                  # 提示词模板，支持变量替换（如 $ARGUMENTS）
    source: str = "project"                    # 技能来源："project"（项目级）或 "user"（用户级）
    skill_dir: str = ""                        # 技能所在的目录路径


# ─── Discovery ──────────────────────────────────────────────

# 全局缓存，用于存储已发现的技能列表，避免重复扫描文件系统
_cached_skills: list[SkillDefinition] | None = None


# 发现并加载所有可用技能
# 扫描用户级目录（~/.claude/skills）和项目级目录（<cwd>/.claude/skills），
# 项目级技能优先级更高，会覆盖同名的用户级技能。
# 使用全局缓存，仅在首次调用时扫描文件系统。
# 返回值：所有已发现的技能定义列表
def discover_skills() -> list[SkillDefinition]:
    global _cached_skills
    if _cached_skills is not None:  # 如果缓存存在，直接返回缓存结果
        return _cached_skills

    skills: dict[str, SkillDefinition] = {}

    # User-level skills (lower priority)
    # 加载用户级技能（优先级较低）
    user_dir = Path.home() / ".claude" / "skills"
    _load_skills_from_dir(user_dir, "user", skills)

    # Project-level skills (higher priority, overwrites)
    # 加载项目级技能（优先级较高，会覆盖同名的用户级技能）
    project_dir = Path.cwd() / ".claude" / "skills"
    _load_skills_from_dir(project_dir, "project", skills)

    _cached_skills = list(skills.values())
    return _cached_skills


# 从指定目录加载技能定义
# 遍历 base_dir 下的每个子目录，查找其中的 SKILL.md 文件并解析为技能定义。
# 参数：
#   base_dir — 要扫描的根目录路径
#   source — 技能来源标识（"user" 或 "project"）
#   skills — 用于存储已加载技能的字典（以技能名为键）
def _load_skills_from_dir(
    base_dir: Path, source: str, skills: dict[str, SkillDefinition]
) -> None:
    if not base_dir.is_dir():  # 目录不存在则跳过
        return
    for entry in base_dir.iterdir():
        if not entry.is_dir():  # 只处理子目录
            continue
        skill_file = entry / "SKILL.md"  # 每个子目录中必须包含 SKILL.md 文件
        if not skill_file.exists():
            continue
        skill = _parse_skill_file(skill_file, source, str(entry))
        if skill:
            skills[skill.name] = skill  # 以技能名为键存储，后加载的会覆盖先前的同名技能


# 解析单个技能文件（SKILL.md）
# 读取文件内容，解析 frontmatter 元数据，构建 SkillDefinition 对象。
# frontmatter 支持以下字段：
#   - name: 技能名称
#   - description: 功能描述
#   - when_to_use / when-to-use: 触发条件
#   - allowed-tools: 允许的工具列表（JSON 数组或逗号分隔字符串）
#   - user-invocable: 是否允许用户调用（默认 "true"）
#   - context: 上下文模式（"inline" 或 "fork"）
# 参数：
#   file_path — SKILL.md 文件路径
#   source — 技能来源标识
#   skill_dir — 技能目录路径
# 返回值：解析成功返回 SkillDefinition，失败返回 None
def _parse_skill_file(
    file_path: Path, source: str, skill_dir: str
) -> SkillDefinition | None:
    try:
        raw = file_path.read_text()  # 读取文件原始内容
        result = parse_frontmatter(raw)  # 解析 frontmatter 和正文
        meta = result.meta  # 提取元数据字典

        # 获取技能名称，优先使用 frontmatter 中的 name 字段
        name = meta.get("name") or file_path.parent.name or "unknown"
        # 解析是否允许用户调用，默认为 True
        user_invocable = meta.get("user-invocable", "true") != "false"
        # 解析上下文模式，默认为 "inline"
        context = "fork" if meta.get("context") == "fork" else "inline"

        # 解析允许使用的工具列表
        allowed_tools: list[str] | None = None
        if "allowed-tools" in meta:
            raw_tools = meta["allowed-tools"]
            if raw_tools.startswith("["):
                # JSON 数组格式，尝试用 json.loads 解析
                try:
                    allowed_tools = json.loads(raw_tools)
                except Exception:
                    # JSON 解析失败时，回退到逗号分隔解析
                    allowed_tools = [s.strip() for s in raw_tools.strip("[]").split(",")]
            else:
                # 逗号分隔格式
                allowed_tools = [s.strip() for s in raw_tools.split(",")]

        return SkillDefinition(
            name=name,
            description=meta.get("description", ""),
            when_to_use=meta.get("when_to_use") or meta.get("when-to-use"),
            allowed_tools=allowed_tools,
            user_invocable=user_invocable,
            context=context,
            prompt_template=result.body,  # frontmatter 之后的正文作为提示词模板
            source=source,
            skill_dir=skill_dir,
        )
    except Exception:
        return None  # 解析过程中出现任何异常都返回 None


# ─── Resolution ─────────────────────────────────────────────


# 根据技能名称查找技能定义
# 遍历所有已发现的技能，返回第一个名称匹配的技能。
# 参数：
#   name — 要查找的技能名称
# 返回值：匹配的 SkillDefinition，未找到返回 None
def get_skill_by_name(name: str) -> SkillDefinition | None:
    for s in discover_skills():
        if s.name == name:
            return s
    return None


# 解析技能提示词模板，替换其中的变量占位符
# 支持以下变量替换：
#   - $ARGUMENTS 或 ${ARGUMENTS} → 替换为用户传入的参数
#   - ${CLAUDE_SKILL_DIR} → 替换为技能所在的目录路径
# 参数：
#   skill — 技能定义对象
#   args — 用户传入的参数字符串
# 返回值：替换变量后的最终提示词
def resolve_skill_prompt(skill: SkillDefinition, args: str) -> str:
    import re
    prompt = skill.prompt_template
    prompt = re.sub(r"\$ARGUMENTS|\$\{ARGUMENTS\}", args, prompt)  # 替换参数占位符
    prompt = prompt.replace("${CLAUDE_SKILL_DIR}", skill.skill_dir)  # 替换技能目录占位符
    return prompt


# 执行指定名称的技能
# 查找技能并生成可执行的上下文信息，包括解析后的提示词、允许的工具列表和上下文模式。
# 参数：
#   skill_name — 要执行的技能名称
#   args — 用户传入的参数字符串
# 返回值：包含 prompt、allowed_tools、context 的字典，技能未找到时返回 None
def execute_skill(
    skill_name: str, args: str
) -> dict | None:
    skill = get_skill_by_name(skill_name)
    if not skill:
        return None
    return {
        "prompt": resolve_skill_prompt(skill, args),      # 解析变量后的提示词
        "allowed_tools": skill.allowed_tools,              # 允许使用的工具列表
        "context": skill.context,                          # 上下文模式
    }


# ─── System prompt section ──────────────────────────────────


# 构建系统提示词中的技能描述部分
# 将所有已发现的技能格式化为 Markdown 文本，分为两组：
#   - 用户可调用技能（user_invocable=True）：显示为 /<name> 命令
#   - 自动触发技能（user_invocable=False）：由系统在适当时机自动调用
# 返回值：Markdown 格式的技能描述文本，无技能时返回空字符串
def build_skill_descriptions() -> str:
    skills = discover_skills()
    if not skills:
        return ""

    lines = ["# Available Skills", ""]
    invocable = [s for s in skills if s.user_invocable]   # 用户可调用的技能
    auto_only = [s for s in skills if not s.user_invocable]  # 仅自动触发的技能

    if invocable:
        lines.append("User-invocable skills (user types /<name> to invoke):")
        for s in invocable:
            lines.append(f"- **/{s.name}**: {s.description}")
            if s.when_to_use:
                lines.append(f"  When to use: {s.when_to_use}")
        lines.append("")

    if auto_only:
        lines.append("Auto-invocable skills (use the skill tool when appropriate):")
        for s in auto_only:
            lines.append(f"- **{s.name}**: {s.description}")
            if s.when_to_use:
                lines.append(f"  When to use: {s.when_to_use}")
        lines.append("")

    lines.append("To invoke a skill programmatically, use the `skill` tool with the skill name and optional arguments.")
    return "\n".join(lines)


# 重置技能缓存
# 清除全局缓存，使下一次调用 discover_skills() 时重新扫描文件系统。
# 通常在技能文件发生变更后调用。
def reset_skill_cache() -> None:
    global _cached_skills
    _cached_skills = None
