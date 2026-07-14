"""Memory system — 4-type file-based memory with MEMORY.md index.
Mirrors Claude Code's memory architecture: semantic recall via sideQuery."""

# 记忆系统模块
# 本模块实现了基于文件的持久化记忆系统，是对 Claude Code 记忆架构的精简复刻。
# 支持四种记忆类型：user（用户偏好）、feedback（用户反馈）、project（项目信息）、reference（外部资源引用）。
# 核心功能：
#   1. CRUD 操作 — 通过 Markdown 文件 + YAML frontmatter 存储记忆条目
#   2. MEMORY.md 索引 — 自动维护的记忆索引文件，便于快速浏览
#   3. 语义召回（Semantic Recall）— 通过 sideQuery 调用模型，根据用户查询选择相关记忆
#   4. 预取机制（Prefetch）— 异步预取相关记忆，减少用户等待时间

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .frontmatter import parse_frontmatter, format_frontmatter

# A callable that sends a prompt and returns model text response.
# Signature: async (system: str, user_message: str) -> str
# SideQueryFn 类型定义：一个可调用对象，接收系统提示词和用户消息，返回模型的文本响应。
# 实际上是一个异步函数，返回 Awaitable[str]。
from typing import Callable
SideQueryFn = Callable[[str, str], Any]  # actually Awaitable[str]

# ─── Types ──────────────────────────────────────────────────

VALID_TYPES = {"user", "feedback", "project", "reference"}  # 有效的记忆类型集合
MAX_INDEX_LINES = 200  # 索引文件最大行数，防止索引过大
MAX_INDEX_BYTES = 25000  # 索引文件最大字节数（约 25KB）


# 记忆条目数据类
# 表示一条完整的记忆记录，包含元数据和正文内容。
# 使用 __slots__ 优化内存占用。
class MemoryEntry:
    __slots__ = ("name", "description", "type", "filename", "content")

    def __init__(self, name: str, description: str, type: str, filename: str, content: str):
        self.name = name
        self.description = description
        self.type = type
        self.filename = filename
        self.content = content


# ─── Paths ──────────────────────────────────────────────────


# 计算当前项目的哈希值
# 基于当前工作目录的路径生成 SHA-256 哈希，取前 16 位作为项目唯一标识。
# 用于隔离不同项目的记忆存储目录。
def _project_hash() -> str:
    return hashlib.sha256(str(Path.cwd()).encode()).hexdigest()[:16]


# 获取当前项目的记忆存储目录
# 路径格式：~/.mini-claude/projects/{project_hash}/memory/
# 如果目录不存在则自动创建。
def get_memory_dir() -> Path:
    d = Path.home() / ".mini-claude" / "projects" / _project_hash() / "memory"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_index_path() -> Path:
    return get_memory_dir() / "MEMORY.md"


# ─── Slugify ────────────────────────────────────────────────


# 将文本转换为 URL 友好的 slug 格式
# 规则：转小写、非字母数字字符替换为下划线、去除首尾下划线、截断到 40 字符。
# 用于生成记忆文件名。
def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower())
    s = s.strip("_")
    return s[:40]


# ─── CRUD ───────────────────────────────────────────────────


# 列出所有已保存的记忆条目
# 扫描记忆目录中的所有 .md 文件（排除 MEMORY.md 索引文件），
# 解析每个文件的 frontmatter 元数据，构建 MemoryEntry 对象列表。
# 返回值按文件修改时间倒序排列（最新的在前）。
def list_memories() -> list[MemoryEntry]:
    d = get_memory_dir()
    entries: list[MemoryEntry] = []
    for f in sorted(d.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        try:
            result = parse_frontmatter(f.read_text())
            meta = result.meta
            if not meta.get("name") or not meta.get("type"):
                continue
            t = meta["type"] if meta["type"] in VALID_TYPES else "project"
            entries.append(MemoryEntry(
                name=meta["name"],
                description=meta.get("description", ""),
                type=t,
                filename=f.name,
                content=result.body,
            ))
        except Exception:
            pass
    # Sort by mtime desc
    entries.sort(key=lambda e: (d / e.filename).stat().st_mtime, reverse=True)
    return entries


# 保存一条记忆到文件
# 生成文件名格式：{type}_{slugified_name}.md
# 文件内容包含 YAML frontmatter（name, description, type）和正文。
# 保存后自动更新 MEMORY.md 索引。
# 返回值：保存的文件名
def save_memory(name: str, description: str, type: str, content: str) -> str:
    d = get_memory_dir()
    filename = f"{type}_{_slugify(name)}.md"
    text = format_frontmatter({"name": name, "description": description, "type": type}, content)
    (d / filename).write_text(text)
    _update_memory_index()
    return filename


# 删除指定的记忆文件
# 参数：
#   filename — 要删除的记忆文件名
# 返回值：删除成功返回 True，文件不存在返回 False
# 删除后自动更新 MEMORY.md 索引。
def delete_memory(filename: str) -> bool:
    filepath = get_memory_dir() / filename
    if not filepath.exists():
        return False
    filepath.unlink()
    _update_memory_index()
    return True


# ─── Index ──────────────────────────────────────────────────


# 更新 MEMORY.md 索引文件
# 遍历所有记忆条目，生成 Markdown 格式的索引列表。
# 索引内容包含每个记忆的名称（带链接）、类型和描述。
def _update_memory_index() -> None:
    memories = list_memories()
    lines = ["# Memory Index", ""]
    for m in memories:
        lines.append(f"- **[{m.name}]({m.filename})** ({m.type}) — {m.description}")
    _get_index_path().write_text("\n".join(lines))


# 加载 MEMORY.md 索引内容
# 读取索引文件并返回其内容。如果索引过大（超过 200 行或 25KB），
# 会进行截断处理并添加提示信息。
# 返回值：索引文本内容，索引文件不存在时返回空字符串。
def load_memory_index() -> str:
    index_path = _get_index_path()
    if not index_path.exists():
        return ""
    content = index_path.read_text()
    lines = content.split("\n")
    if len(lines) > MAX_INDEX_LINES:
        content = "\n".join(lines[:MAX_INDEX_LINES]) + "\n\n[... truncated, too many memory entries ...]"
    if len(content.encode()) > MAX_INDEX_BYTES:
        content = content[:MAX_INDEX_BYTES] + "\n\n[... truncated, index too large ...]"
    return content


# ─── Memory Header (lightweight scan) ──────────────────────

# 记忆文件头信息类
# 用于轻量级扫描记忆目录，只读取 frontmatter 元数据（前 30 行），
# 避免读取整个文件内容，提高扫描速度。
class MemoryHeader:
    __slots__ = ("filename", "file_path", "mtime_ms", "description", "type")

    def __init__(self, filename: str, file_path: str, mtime_ms: float,
                 description: str | None, type: str | None):
        self.filename = filename
        self.file_path = file_path
        self.mtime_ms = mtime_ms
        self.description = description
        self.type = type


MAX_MEMORY_FILES = 200  # 最大记忆文件数量
MAX_MEMORY_BYTES_PER_FILE = 4096  # 单个记忆文件最大字节数（4KB）
MAX_SESSION_MEMORY_BYTES = 60 * 1024  # 60KB cumulative per session  # 每个会话累计记忆字节数上限（60KB）


# 扫描记忆目录，获取所有记忆文件的头信息
# 只读取每个文件的前 30 行（frontmatter 部分），提高扫描速度。
# 返回值按修改时间倒序排列，最多返回 200 个。
def scan_memory_headers() -> list[MemoryHeader]:
    """Scan memory directory — read only frontmatter (first 30 lines) for speed."""
    d = get_memory_dir()
    headers: list[MemoryHeader] = []
    for f in d.glob("*.md"):
        if f.name == "MEMORY.md":
            continue
        try:
            stat = f.stat()
            raw = f.read_text()
            first30 = "\n".join(raw.split("\n")[:30])
            result = parse_frontmatter(first30)
            meta = result.meta
            t = meta.get("type")
            headers.append(MemoryHeader(
                filename=f.name,
                file_path=str(f),
                mtime_ms=stat.st_mtime * 1000,
                description=meta.get("description"),
                type=t if t in VALID_TYPES else None,
            ))
        except Exception:
            pass
    headers.sort(key=lambda h: h.mtime_ms, reverse=True)
    return headers[:MAX_MEMORY_FILES]


# 将记忆头信息列表格式化为清单文本
# 每行格式：- [类型] 文件名 (时间戳): 描述
# 用于传递给语义选择器（semantic selector）进行记忆筛选。
def format_memory_manifest(headers: list[MemoryHeader]) -> str:
    """Format manifest for semantic selector: one line per memory."""
    lines = []
    for h in headers:
        tag = f"[{h.type}] " if h.type else ""
        ts = datetime.fromtimestamp(h.mtime_ms / 1000, tz=timezone.utc).isoformat()
        if h.description:
            lines.append(f"- {tag}{h.filename} ({ts}): {h.description}")
        else:
            lines.append(f"- {tag}{h.filename} ({ts})")
    return "\n".join(lines)


# ─── Memory Age / Freshness ────────────────────────────────


# 计算记忆的年龄（相对时间描述）
# 参数：
#   mtime_ms — 记忆文件的最后修改时间（毫秒时间戳）
# 返回值：人类可读的时间描述，如 "today"、"yesterday"、"3 days ago"
def memory_age(mtime_ms: float) -> str:
    days = max(0, int((time.time() * 1000 - mtime_ms) / 86_400_000))
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


# 生成记忆新鲜度警告信息
# 如果记忆超过 1 天，返回警告文本提醒用户记忆可能已过时。
# 返回值：空字符串（1 天内）或警告文本（超过 1 天）。
def memory_freshness_warning(mtime_ms: float) -> str:
    days = max(0, int((time.time() * 1000 - mtime_ms) / 86_400_000))
    if days <= 1:
        return ""
    return (f"This memory is {days} days old. Memories are point-in-time observations, "
            "not live state — claims about code behavior may be outdated. "
            "Verify against current code before asserting as fact.")


# ─── Semantic Recall (sideQuery) ────────────────────────────

# 语义选择记忆的系统提示词
# 用于指导模型从候选记忆中选择与用户查询相关的记忆。
# 模型需要根据记忆的名称和描述判断哪些记忆对当前查询有帮助。
SELECT_MEMORIES_PROMPT = """You are selecting memories that will be useful to an AI coding assistant as it processes a user's query. You will be given the user's query and a list of available memory files with their filenames and descriptions.

Return a JSON object with a "selected_memories" array of filenames for the memories that will clearly be useful (up to 5). Only include memories that you are certain will be helpful based on their name and description.
- If you are unsure if a memory will be useful, do not include it.
- If no memories would clearly be useful, return an empty array."""


# 相关记忆数据类
# 表示被语义选择器选中的相关记忆，包含文件路径、内容、修改时间和头信息。
class RelevantMemory:
    __slots__ = ("path", "content", "mtime_ms", "header")

    def __init__(self, path: str, content: str, mtime_ms: float, header: str):
        self.path = path
        self.content = content
        self.mtime_ms = mtime_ms
        self.header = header


# 异步调用模型进行语义记忆选择
# 流程：
#   1. 扫描记忆目录获取所有记忆头信息
#   2. 过滤掉已经展示过的记忆（already_surfaced）
#   3. 构建记忆清单并调用 side_query 让模型选择相关记忆
#   4. 解析模型返回的 JSON，获取选中的记忆文件名
#   5. 读取选中记忆的完整内容，添加新鲜度警告信息
# 参数：
#   query — 用户查询文本
#   side_query — 用于调用模型的异步函数
#   already_surfaced — 已经展示过的记忆文件路径集合
# 返回值：相关记忆列表，最多 5 条
async def select_relevant_memories(
    query: str,
    side_query: SideQueryFn,
    already_surfaced: set[str],
) -> list[RelevantMemory]:
    """Call the model to semantically select relevant memories."""
    headers = scan_memory_headers()
    if not headers:
        return []

    candidates = [h for h in headers if h.file_path not in already_surfaced]
    if not candidates:
        return []

    manifest = format_memory_manifest(candidates)

    try:
        text = await side_query(
            SELECT_MEMORIES_PROMPT,
            f"Query: {query}\n\nAvailable memories:\n{manifest}",
        )

        # Extract JSON from response
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return []

        parsed = json.loads(match.group(0))
        selected_filenames = set(parsed.get("selected_memories", []))

        selected = [h for h in candidates if h.filename in selected_filenames][:5]

        result: list[RelevantMemory] = []
        for h in selected:
            content = Path(h.file_path).read_text()
            if len(content.encode()) > MAX_MEMORY_BYTES_PER_FILE:
                content = content[:MAX_MEMORY_BYTES_PER_FILE] + "\n\n[... truncated, memory file too large ...]"
            freshness = memory_freshness_warning(h.mtime_ms)
            header_text = (
                f"{freshness}\n\nMemory: {h.file_path}:" if freshness
                else f"Memory (saved {memory_age(h.mtime_ms)}): {h.file_path}:"
            )
            result.append(RelevantMemory(
                path=h.file_path, content=content,
                mtime_ms=h.mtime_ms, header=header_text,
            ))
        return result
    except Exception as e:
        if "cancel" in str(e).lower():
            return []
        print(f"[memory] semantic recall failed: {e}")
        return []


# ─── Prefetch Handle ────────────────────────────────────────

# 记忆预取句柄类
# 封装异步预取任务，用于跟踪预取状态和获取结果。
# consumed 标记是否已被消费，settled 属性检查任务是否完成。
class MemoryPrefetch:
    def __init__(self, task: asyncio.Task):
        self.task = task
        self.consumed = False

    @property
    def settled(self) -> bool:
        return self.task.done()


# 启动异步记忆预取
# 在后台异步执行记忆语义选择，返回预取句柄供后续查询结果。
# 有三个门控条件，任一不满足则返回 None：
#   1. 查询内容必须"有意义"（至少 2 个 CJK 字符或多词）
#   2. 会话记忆字节数未超过上限（60KB）
#   3. 记忆目录中必须存在记忆文件
def start_memory_prefetch(
    query: str,
    side_query: SideQueryFn,
    already_surfaced: set[str],
    session_memory_bytes: int,
) -> MemoryPrefetch | None:
    """Start async memory prefetch. Returns handle to poll for results."""
    # Gate: substantial input only — 2+ CJK chars or multi-word. A pure
    # whitespace test would never trigger for CJK queries like "部署流程"
    # (no spaces). Mirrors the TS isQuerySubstantial() logic.
    stripped = query.strip()
    cjk_count = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", stripped))
    if cjk_count < 2 and not re.search(r"\s", stripped):
        return None

    # Gate: session budget
    if session_memory_bytes >= MAX_SESSION_MEMORY_BYTES:
        return None

    # Gate: memories must exist
    d = get_memory_dir()
    has_memories = any(f.suffix == ".md" and f.name != "MEMORY.md" for f in d.iterdir())
    if not has_memories:
        return None

    task = asyncio.create_task(
        select_relevant_memories(query, side_query, already_surfaced)
    )
    return MemoryPrefetch(task)


# 将召回的记忆格式化为可注入的用户消息内容
# 每条记忆用 <system-reminder> 标签包裹，包含头信息（文件路径、新鲜度）和正文内容。
# 返回值：所有记忆的格式化文本，用空行分隔。
def format_memories_for_injection(memories: list[RelevantMemory]) -> str:
    """Format recalled memories for injection as user message content."""
    parts = []
    for m in memories:
        parts.append(f"<system-reminder>\n{m.header}\n\n{m.content}\n</system-reminder>")
    return "\n\n".join(parts)


# ─── System prompt section ──────────────────────────────────


# 构建系统提示词中的记忆系统部分
# 生成关于记忆系统的说明文本，包括：
#   - 记忆存储路径
#   - 四种记忆类型的说明
#   - 如何保存记忆（write_file + YAML frontmatter）
#   - 不应保存的内容（代码模式、Git 历史等）
#   - 何时召回记忆
#   - 当前记忆索引内容（如果有）
def build_memory_prompt_section() -> str:
    index = load_memory_index()
    memory_dir = str(get_memory_dir())

    return f"""# Memory System

You have a persistent, file-based memory system at `{memory_dir}`.

## Memory Types
- **user**: User's role, preferences, knowledge level
- **feedback**: Corrections and guidance from the user (include Why + How to apply)
- **project**: Ongoing work, goals, deadlines, decisions
- **reference**: Pointers to external resources (URLs, tools, dashboards)

## How to Save Memories
Use the write_file tool to create a memory file with YAML frontmatter:

```markdown
---
name: memory name
description: one-line description
type: user|feedback|project|reference
---
Memory content here.
```

Save to: `{memory_dir}/`
Filename format: `{{type}}_{{slugified_name}}.md`

The MEMORY.md index is auto-updated when you write to the memory directory — do NOT update it manually.

## What NOT to Save
- Code patterns or architecture (read the code instead)
- Git history (use git log)
- Anything already in CLAUDE.md
- Ephemeral task details

## When to Recall
When the user asks you to remember or recall, or when prior context seems relevant.
{chr(10) + "## Current Memory Index" + chr(10) + index if index else chr(10) + "(No memories saved yet.)"}"""
