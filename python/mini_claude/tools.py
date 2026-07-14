"""Tool definitions and execution — 10 tools with 5 permission modes.
Tool system inspired by Claude Code's published design: read_file, write_file, edit_file, list_files,
grep_search, run_shell, skill, enter/exit_plan_mode, agent."""

# =============================================================================
# 工具定义与执行模块 (Tool Definitions and Execution)
#
# 本模块实现了 Mini Claude Code 的核心工具系统，灵感来源于 Claude Code 的公开设计。
# 共定义了 11 个工具（含 tool_search），支持 5 种权限模式：
#
# 工具列表：
#   - read_file：读取文件内容，返回带行号的文本
#   - write_file：写入文件，不存在则创建，存在则覆盖
#   - edit_file：精确字符串替换编辑文件
#   - list_files：按 glob 模式列出匹配的文件
#   - grep_search：在文件中搜索正则表达式模式
#   - run_shell：执行 shell 命令并返回输出
#   - skill：调用已注册的技能模板
#   - web_fetch：获取 URL 内容并返回纯文本
#   - enter_plan_mode / exit_plan_mode：进入/退出计划模式
#   - agent：启动子代理处理独立任务
#   - tool_search：搜索并激活延迟加载的工具
#
# 权限模式：
#   - default：默认模式，读取类工具自动允许，写入/编辑类工具需确认
#   - plan：计划模式，只允许读取和编辑计划文件
#   - acceptEdits：接受编辑模式，编辑类工具自动允许
#   - bypassPermissions：绕过权限模式（--yolo），所有工具自动允许（deny 规则除外）
#   - dontAsk：不询问模式，需要确认的操作自动拒绝
#
# 权限规则通过 .claude/settings.json 文件配置，支持 allow/deny 规则。
# =============================================================================

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from .memory import get_memory_dir
from .frontmatter import parse_frontmatter

# ─── Permission modes ──────────────────────────────────────
# 权限模式定义：定义工具分类和平台检测

PermissionMode = str  # "default" | "plan" | "acceptEdits" | "bypassPermissions" | "dontAsk" | "auto"
# 权限模式类型别名，支持以下模式：
# "default" - 默认模式，读取工具自动允许，写入工具需确认
# "plan" - 计划模式，只允许读取和编辑计划文件
# "acceptEdits" - 接受编辑模式，编辑类工具自动允许
# "bypassPermissions" - 绕过权限模式（--yolo），所有工具自动允许（deny 规则除外）
# "dontAsk" - 不询问模式，需要确认的操作自动拒绝

READ_TOOLS = {"read_file", "list_files", "grep_search", "web_fetch"}
# 只读工具集合：这些工具不会修改文件系统，始终允许执行

EDIT_TOOLS = {"write_file", "edit_file"}
# 编辑工具集合：这些工具会修改文件系统，需要权限检查

# Concurrency-safe tools can run in parallel (read-only, no side effects)
# 并发安全工具：只读且无副作用，可以并行执行
CONCURRENCY_SAFE_TOOLS = {"read_file", "list_files", "grep_search", "web_fetch"}

IS_WIN = sys.platform == "win32"
# 检测是否为 Windows 平台，用于选择 grep 实现（系统 grep 或纯 Python 回退）

# ─── Type alias ──────────────────────────────────────────────

ToolDef = dict  # Anthropic tool schema dict
# 工具定义类型别名：表示 Anthropic 工具 schema 的字典格式

# ─── Tool definitions ───────────────────────────────────────
# 工具定义列表：包含所有可用工具的名称、描述和输入 schema

tool_definitions: list[ToolDef] = [
    {
        "name": "read_file",
        "description": "Read the contents of a file. Returns the file content with line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The path to the file to read"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The path to the file to write"},
                "content": {"type": "string", "description": "The content to write to the file"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Edit a file by replacing an exact string match with new content. The old_string must match exactly (including whitespace and indentation).",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The path to the file to edit"},
                "old_string": {"type": "string", "description": "The exact string to find and replace"},
                "new_string": {"type": "string", "description": "The string to replace it with"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "list_files",
        "description": "List files matching a glob pattern. Returns matching file paths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": 'Glob pattern to match files (e.g., "**/*.ts", "src/**/*")'},
                "path": {"type": "string", "description": "Base directory to search from. Defaults to current directory."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep_search",
        "description": "Search for a pattern in files. Returns matching lines with file paths and line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "The regex pattern to search for"},
                "path": {"type": "string", "description": "Directory or file to search in. Defaults to current directory."},
                "include": {"type": "string", "description": 'File glob pattern to include (e.g., "*.ts", "*.py")'},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "run_shell",
        "description": "Execute a shell command and return its output. Use this for running tests, installing packages, git operations, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"},
                "timeout": {"type": "number", "description": "Timeout in milliseconds (default: 30000)"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "skill",
        "description": "Invoke a registered skill by name. Skills are prompt templates loaded from .claude/skills/. Returns the skill's resolved prompt to follow.",
        "input_schema": {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "The name of the skill to invoke"},
                "args": {"type": "string", "description": "Optional arguments to pass to the skill"},
            },
            "required": ["skill_name"],
        },
    },
    {
        "name": "web_fetch",
        "description": "Fetch a URL and return its content as text. For HTML pages, tags are stripped to return readable text. For JSON/text responses, content is returned directly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch"},
                "max_length": {"type": "number", "description": "Maximum content length in characters (default 50000)"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "enter_plan_mode",
        "description": "Enter plan mode to switch to a read-only planning phase. In plan mode, you can only read files and write to the plan file.",
        "input_schema": {"type": "object", "properties": {}},
        "deferred": True,
    },
    {
        "name": "exit_plan_mode",
        "description": "Exit plan mode after you have finished writing your plan to the plan file.",
        "input_schema": {"type": "object", "properties": {}},
        "deferred": True,
    },
    {
        "name": "agent",
        "description": "Launch a sub-agent to handle a task autonomously. Sub-agents have isolated context and return their result. Types: 'explore' (read-only), 'plan' (read-only, structured planning), 'general' (full tools).",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Short (3-5 word) description of the sub-agent's task"},
                "prompt": {"type": "string", "description": "Detailed task instructions for the sub-agent"},
                "type": {"type": "string", "enum": ["explore", "plan", "general"], "description": "Agent type. Default: general"},
            },
            "required": ["description", "prompt"],
        },
    },
    # ─── Tool search (deferred tool loader) ─────────────────────
    # 工具搜索工具：用于查找并激活延迟加载的工具
    {
        "name": "tool_search",
        "description": "Search for available tools by name or keyword. Returns full schema definitions for matching deferred tools so you can use them.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Tool name or search keywords"},
            },
            "required": ["query"],
        },
    },
]

# ─── Deferred tool activation ───────────────────────────────
# 延迟工具激活机制：某些工具默认不加载，需要时通过 tool_search 激活
# 这样可以减少发送给 API 的工具定义数量，降低 token 消耗

_activated_tools: set[str] = set()
# 已激活的延迟工具集合：记录哪些延迟工具已经被激活


def reset_activated_tools() -> None:
    """重置已激活的延迟工具集合。"""
    _activated_tools.clear()


def get_active_tool_definitions(all_tools: list[ToolDef] | None = None) -> list[ToolDef]:
    """Return tool definitions, excluding deferred tools that haven't been activated.
    Strips the 'deferred' key so it's not sent to the API."""
    # 获取当前活跃的工具定义列表，排除未激活的延迟工具。
    # 同时移除 'deferred' 键，因为该键不需要发送给 API。
    tools = all_tools if all_tools is not None else tool_definitions
    return [
        {k: v for k, v in t.items() if k != "deferred"}
        for t in tools
        if not t.get("deferred") or t["name"] in _activated_tools
    ]


def get_deferred_tool_names(all_tools: list[ToolDef] | None = None) -> list[str]:
    """Return names of deferred tools that haven't been activated yet."""
    # 获取所有尚未激活的延迟工具名称列表
    tools = all_tools if all_tools is not None else tool_definitions
    return [t["name"] for t in tools if t.get("deferred") and t["name"] not in _activated_tools]


# ─── Tool execution ─────────────────────────────────────────
# 工具执行函数：实现各个工具的具体逻辑


def _read_file(inp: dict) -> str:
    """读取文件内容，返回带行号的文本。"""
    try:
        # errors="replace": undecodable bytes become U+FFFD instead of
        # raising — same behavior as Node's readFileSync("utf-8") in the TS
        # version, so both implementations return content for mixed files.
        # 使用 errors="replace" 处理无法解码的字节，避免因编码问题导致读取失败
        content = Path(inp["file_path"]).read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")
        # 为每行添加行号前缀，格式为 "   1 | 内容"
        numbered = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
        return numbered
    except Exception as e:
        return f"Error reading file: {e}"


def _write_file(inp: dict) -> str:
    """写入文件内容，自动创建不存在的目录，返回写入结果和前30行预览。"""
    try:
        path = Path(inp["file_path"])
        # 自动创建父目录（如果不存在）
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(inp["content"])
        # 如果是记忆目录下的 .md 文件，自动更新记忆索引
        _auto_update_memory_index(str(path))
        lines = inp["content"].split("\n")
        line_count = len(lines)
        # 生成前30行的预览
        preview = "\n".join(f"{i+1:4d} | {l}" for i, l in enumerate(lines[:30]))
        trunc = f"\n  ... ({line_count} lines total)" if line_count > 30 else ""
        return f"Successfully wrote to {inp['file_path']} ({line_count} lines)\n\n{preview}{trunc}"
    except Exception as e:
        return f"Error writing file: {e}"


def _auto_update_memory_index(file_path: str) -> None:
    """自动更新记忆目录的索引文件（MEMORY.md）。

    当写入记忆目录下的 .md 文件时，自动扫描所有记忆文件并生成索引。
    索引包含每个记忆文件的名称、类型和描述，方便快速查找。
    """
    try:
        mem_dir = str(get_memory_dir())
        # 仅处理记忆目录下的 .md 文件（排除 MEMORY.md 索引文件本身）
        if file_path.startswith(mem_dir) and file_path.endswith(".md") and not file_path.endswith("MEMORY.md"):
            mem_path = Path(mem_dir)
            lines = ["# Memory Index", ""]
            # 遍历所有记忆文件，提取 frontmatter 元数据
            for f in sorted(mem_path.glob("*.md")):
                if f.name == "MEMORY.md":
                    continue
                try:
                    raw = f.read_text()
                    # 从 frontmatter 中提取 name、type、description 字段
                    name_match = re.search(r"^name:\s*(.+)$", raw, re.MULTILINE)
                    type_match = re.search(r"^type:\s*(.+)$", raw, re.MULTILINE)
                    desc_match = re.search(r"^description:\s*(.+)$", raw, re.MULTILINE)
                    if name_match and type_match:
                        n = name_match.group(1).strip()
                        t = type_match.group(1).strip()
                        d = desc_match.group(1).strip() if desc_match else ""
                        # 生成 Markdown 格式的索引条目
                        lines.append(f"- **[{n}]({f.name})** ({t}) — {d}")
                except Exception:
                    pass
            # 写入索引文件
            (mem_path / "MEMORY.md").write_text("\n".join(lines))
    except Exception:
        pass


# ─── Edit helpers: quote normalization + diff ───────────────
# 编辑辅助函数：引号标准化和差异生成
# 用于 edit_file 工具，支持智能引号匹配和差异输出


def _normalize_quotes(s: str) -> str:
    """将智能引号（弯引号）标准化为直引号。"""
    s = re.sub("[\u2018\u2019\u2032]", "'", s)
    s = re.sub('[\u201c\u201d\u2033]', '"', s)
    return s


def _find_actual_string(file_content: str, search_string: str) -> str | None:
    """在文件内容中查找目标字符串，支持引号标准化匹配。"""
    # 直接匹配
    if search_string in file_content:
        return search_string
    # 引号标准化后匹配
    norm_search = _normalize_quotes(search_string)
    norm_file = _normalize_quotes(file_content)
    idx = norm_file.find(norm_search)
    if idx != -1:
        # 返回原始文件中对应位置的字符串（保留原始引号格式）
        return file_content[idx:idx + len(search_string)]
    return None


def _generate_diff(old_content: str, old_string: str, new_string: str) -> str:
    """生成类似 unified diff 格式的差异输出。"""
    # 计算变更位置的行号
    before_change = old_content.split(old_string)[0]
    line_num = before_change.count("\n") + 1
    old_lines = old_string.split("\n")
    new_lines = new_string.split("\n")

    # 生成 diff 输出
    parts = [f"@@ -{line_num},{len(old_lines)} +{line_num},{len(new_lines)} @@"]
    for l in old_lines:
        parts.append(f"- {l}")
    for l in new_lines:
        parts.append(f"+ {l}")
    return "\n".join(parts)


def _edit_file(inp: dict) -> str:
    """执行文件编辑：精确字符串替换，支持引号标准化匹配。"""
    try:
        path = Path(inp["file_path"])
        content = path.read_text()

        # 查找目标字符串（支持引号标准化匹配）
        actual = _find_actual_string(content, inp["old_string"])
        if not actual:
            return f"Error: old_string not found in {inp['file_path']}"

        # 检查唯一性：old_string 必须在文件中只出现一次
        count = content.count(actual)
        if count > 1:
            return f"Error: old_string found {count} times in {inp['file_path']}. Must be unique."

        # 执行替换（只替换第一次出现）
        new_content = content.replace(actual, inp["new_string"], 1)
        path.write_text(new_content)

        # 生成差异输出
        diff = _generate_diff(content, actual, inp["new_string"])
        # 如果是通过引号标准化匹配的，添加提示信息
        quote_note = " (matched via quote normalization)" if actual != inp["old_string"] else ""
        return f"Successfully edited {inp['file_path']}{quote_note}\n\n{diff}"
    except Exception as e:
        return f"Error editing file: {e}"


def _list_files(inp: dict) -> str:
    """按 glob 模式列出匹配的文件，跳过 node_modules 和隐藏目录。"""
    try:
        base = Path(inp.get("path") or ".")
        pattern = inp["pattern"]
        files = []
        extra = 0
        for p in base.glob(pattern):
            if p.is_file():
                rel = str(p.relative_to(base) if base != Path(".") else p)
                # Skip node_modules / hidden components by exact path part —
                # a substring test would also drop a file merely *named*
                # like "my_node_modules_note.txt". Skipping dotfiles matches
                # the TS glob behavior (dot:false).
                # 跳过 node_modules 和以点开头的隐藏目录/文件
                if any(part == "node_modules" or part.startswith(".") for part in Path(rel).parts):
                    continue
                # Keep at most 200 entries, but keep counting so the model
                # knows how many matches were omitted (matches TS behavior).
                # 最多保留 200 条结果，超出部分计数
                if len(files) < 200:
                    files.append(rel)
                else:
                    extra += 1
        if not files:
            return "No files found matching the pattern."
        result = "\n".join(files)
        if extra:
            result += f"\n... and {extra} more"
        return result
    except Exception as e:
        return f"Error listing files: {e}"


def _grep_search(inp: dict) -> str:
    """在文件中搜索正则表达式模式，返回匹配的行（带文件路径和行号）。"""
    pattern = inp["pattern"]
    path = inp.get("path") or "."
    include = inp.get("include")

    # Try system grep first (Linux/macOS)
    # 优先使用系统 grep 命令（Linux/macOS），性能更好
    if not IS_WIN:
        try:
            args = ["grep", "--line-number", "--color=never", "-r"]
            if include:
                args.append(f"--include={include}")
            args.extend(["--", pattern, path])
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=10
            )
            if result.returncode == 1:
                return "No matches found."
            if result.returncode == 0:
                lines = [l for l in result.stdout.split("\n") if l]
                # 最多显示 100 条匹配结果
                output = "\n".join(lines[:100])
                if len(lines) > 100:
                    output += f"\n... and {len(lines) - 100} more matches"
                return output
            # Non-zero exit (not 1) — fall through to Python fallback
        except Exception:
            pass  # Fall through to Python fallback

    # Pure Python fallback (Windows, or system grep unavailable)
    # 回退到纯 Python 实现（Windows 或系统 grep 不可用时）
    return _grep_python(pattern, path, include)


def _grep_python(pattern: str, directory: str, include: str | None) -> str:
    """纯 Python 实现的 grep 搜索，用于 Windows 或系统 grep 不可用时。"""
    try:
        regex = re.compile(pattern)
    except re.error as e:
        # A model-supplied bad regex must come back as a tool error string,
        # not crash the agent loop (system grep exit 2 also falls through
        # to here with the same broken pattern).
        # 正则表达式编译失败时返回错误信息，而不是崩溃
        return f"Error: invalid regex pattern: {e}"
    include_pattern = include
    matches: list[str] = []
    extra = 0

    def walk(d: str) -> None:
        """递归遍历目录，搜索匹配的文件和行。"""
        nonlocal extra
        try:
            entries = os.listdir(d)
        except Exception:
            return
        for name in entries:
            # 跳过隐藏目录和 node_modules
            if name.startswith(".") or name == "node_modules":
                continue
            full = os.path.join(d, name)
            if os.path.isdir(full):
                walk(full)
                continue
            # 如果指定了文件过滤模式，检查文件名是否匹配
            if include_pattern and not fnmatch.fnmatch(name, include_pattern):
                continue
            try:
                text = Path(full).read_text(errors="replace")
                for i, line in enumerate(text.split("\n")):
                    if regex.search(line):
                        # Show at most 100 matches, but keep counting so the
                        # model knows how many were omitted.
                        # 最多显示 100 条匹配结果
                        if len(matches) < 100:
                            matches.append(f"{full}:{i+1}:{line}")
                        else:
                            extra += 1
            except Exception:
                pass

    walk(directory)
    if not matches:
        return "No matches found."
    output = "\n".join(matches)
    if extra:
        output += f"\n... and {extra} more matches"
    return output


def _run_shell(inp: dict) -> str:
    """执行 shell 命令并返回输出。支持超时控制。"""
    try:
        # 将毫秒超时转换为秒
        timeout_ms = inp.get("timeout", 30000)
        timeout_s = timeout_ms / 1000
        result = subprocess.run(
            inp["command"],
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        output = result.stdout or ""
        if result.returncode != 0:
            # 命令执行失败，返回错误信息和标准输出/错误
            stderr = f"\nStderr: {result.stderr}" if result.stderr else ""
            stdout = f"\nStdout: {result.stdout}" if result.stdout else ""
            return f"Command failed (exit code {result.returncode}){stdout}{stderr}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {inp.get('timeout', 30000)}ms"
    except Exception as e:
        return f"Error: {e}"


def _web_fetch(inp: dict) -> str:
    """获取 URL 内容并返回纯文本。HTML 页面会自动去除标签。"""
    import urllib.request
    import urllib.error

    url = inp.get("url", "")
    max_length = inp.get("max_length", 50000)
    # urllib happily opens file:// and other schemes — that would turn a
    # "web" fetch into local file disclosure. http(s) only (TS fetch already
    # rejects non-http schemes).
    # 安全检查：只允许 http(s) 协议，防止本地文件泄露
    if not url.lower().startswith(("http://", "https://")):
        return "Error: only http(s) URLs are supported"
    req = urllib.request.Request(url, headers={"User-Agent": "mini-claude/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_type = resp.headers.get("Content-Type", "")
            text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return f"HTTP error: {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return f"Error fetching {url}: {e.reason}"
    except Exception as e:
        return f"Error fetching {url}: {e}"

    # HTML 内容处理：去除脚本、样式和标签，提取纯文本
    if "html" in content_type:
        # 移除 <script> 和 <style> 标签及其内容
        text = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
        # 移除所有 HTML 标签
        text = re.sub(r"<[^>]*>", " ", text)
        # 解码 HTML 实体
        text = text.replace("&nbsp;", " ").replace("&amp;", "&")
        text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
        # 压缩多余的空白字符
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

    # 截断超长内容
    if len(text) > max_length:
        text = text[:max_length] + f"\n\n[... truncated at {max_length} characters]"

    return text or "(empty response)"


# ─── Dangerous command patterns ─────────────────────────────
# 危险命令模式：用于检测可能造成系统损害的 shell 命令
# 这些命令在执行前需要用户确认

DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s"),
    re.compile(r"\bgit\s+(push|reset|clean|checkout\s+\.)"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s"),
    re.compile(r">\s*/dev/"),
    re.compile(r"\bkill\b"),
    re.compile(r"\bpkill\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\bdel\s", re.IGNORECASE),
    re.compile(r"\brmdir\s", re.IGNORECASE),
    re.compile(r"\bformat\s", re.IGNORECASE),
    re.compile(r"\btaskkill\s", re.IGNORECASE),
    re.compile(r"\bRemove-Item\s", re.IGNORECASE),
    re.compile(r"\bStop-Process\s", re.IGNORECASE),
]


def is_dangerous(command: str) -> bool:
    """检查 shell 命令是否包含危险模式。"""
    return any(p.search(command) for p in DANGEROUS_PATTERNS)


# ─── Permission rules (.claude/settings.json) ───────────────
# 权限规则配置：从 .claude/settings.json 文件加载 allow/deny 规则
# 规则格式：tool_name 或 tool_name(pattern)


def _parse_rule(rule: str) -> dict:
    """解析权限规则字符串，返回工具名称和匹配模式。"""
    # 格式：tool_name 或 tool_name(pattern)
    m = re.match(r"^([a-z_]+)\((.+)\)$", rule)
    if m:
        return {"tool": m.group(1), "pattern": m.group(2)}
    return {"tool": rule, "pattern": None}


def _load_settings(file_path: Path) -> dict | None:
    """加载 .claude/settings.json 配置文件。"""
    if not file_path.exists():
        return None
    try:
        return json.loads(file_path.read_text())
    except Exception:
        return None


_cached_rules: dict | None = None
# 缓存的权限规则，避免重复加载配置文件


def load_permission_rules() -> dict:
    """加载权限规则配置，合并用户级和项目级设置。

    返回 {"allow": [...], "deny": [...]} 格式的规则字典。
    项目级规则会追加到用户级规则之后（deny 规则优先级更高）。
    """
    global _cached_rules
    if _cached_rules is not None:
        return _cached_rules

    allow: list[dict] = []
    deny: list[dict] = []

    # 加载用户级和项目级配置
    user_settings = _load_settings(Path.home() / ".claude" / "settings.json")
    project_settings = _load_settings(Path.cwd() / ".claude" / "settings.json")

    # 合并两级配置的权限规则
    for settings in [user_settings, project_settings]:
        if not settings or "permissions" not in settings:
            continue
        perms = settings["permissions"]
        for r in perms.get("allow", []):
            allow.append(_parse_rule(r))
        for r in perms.get("deny", []):
            deny.append(_parse_rule(r))

    _cached_rules = {"allow": allow, "deny": deny}
    return _cached_rules


def _matches_rule(rule: dict, tool_name: str, inp: dict) -> bool:
    """检查工具调用是否匹配指定的权限规则。"""
    if rule["tool"] != tool_name:
        return False
    # 无模式的规则匹配所有该工具的调用
    if rule["pattern"] is None:
        return True

    # 根据工具类型获取匹配值
    value = ""
    if tool_name == "run_shell":
        value = inp.get("command", "")
    elif "file_path" in inp:
        value = inp["file_path"]
    else:
        return True

    # 支持通配符匹配（以 * 结尾的模式）
    pattern = rule["pattern"]
    if pattern.endswith("*"):
        return value.startswith(pattern[:-1])
    return value == pattern


def _check_permission_rules(tool_name: str, inp: dict) -> str | None:
    """检查工具调用是否匹配 allow 或 deny 规则。

    返回 "deny"、"allow" 或 None（无匹配规则）。
    deny 规则优先级高于 allow 规则。
    """
    rules = load_permission_rules()
    # deny 规则优先
    for rule in rules["deny"]:
        if _matches_rule(rule, tool_name, inp):
            return "deny"
    for rule in rules["allow"]:
        if _matches_rule(rule, tool_name, inp):
            return "allow"
    return None


def check_permission(
    tool_name: str,
    inp: dict,
    mode: str = "default",
    plan_file_path: str | None = None,
) -> dict:
    """检查工具调用权限，返回操作建议。

    返回格式：{"action": "allow"|"deny"|"confirm", "message": ...}
    - allow：允许执行
    - deny：拒绝执行
    - confirm：需要用户确认
    """
    # Deny rules always win — even bypassPermissions (--yolo) is constrained
    # by deny rules (docs/06-permissions.md), so check them before any mode
    # shortcut.
    # deny 规则始终优先，即使在 bypassPermissions 模式下也受 deny 规则约束
    rule_result = _check_permission_rules(tool_name, inp)
    if rule_result == "deny":
        return {"action": "deny", "message": f"Denied by permission rule for {tool_name}"}

    # Plan mode's read-only contract beats allow rules and bypass: only the
    # plan file itself is writable, and shell stays blocked (docs/10).
    # 计划模式的只读约束优先于 allow 规则和 bypass 模式
    if mode == "plan":
        if tool_name in EDIT_TOOLS:
            file_path = inp.get("file_path") or inp.get("path")
            # 只有计划文件本身可写
            if plan_file_path and file_path == plan_file_path:
                return {"action": "allow"}
            return {"action": "deny", "message": f"Blocked in plan mode: {tool_name}"}
        if tool_name == "run_shell":
            return {"action": "deny", "message": "Shell commands blocked in plan mode"}

    # bypassPermissions 模式：所有操作自动允许
    if mode == "bypassPermissions":
        return {"action": "allow"}

    if rule_result == "allow":
        return {"action": "allow"}

    # 只读工具始终允许
    if tool_name in READ_TOOLS:
        return {"action": "allow"}

    # 计划模式切换工具始终允许
    if tool_name in ("enter_plan_mode", "exit_plan_mode"):
        return {"action": "allow"}

    # acceptEdits 模式：编辑工具自动允许
    if mode == "acceptEdits" and tool_name in EDIT_TOOLS:
        return {"action": "allow"}

    needs_confirm = False
    confirm_message = ""

    # 需要用户确认的情况：
    # 1. 危险的 shell 命令
    # 2. 创建新文件
    # 3. 编辑不存在的文件
    if tool_name == "run_shell" and is_dangerous(inp.get("command", "")):
        needs_confirm = True
        confirm_message = inp.get("command", "")
    elif tool_name == "write_file" and not Path(inp.get("file_path", "")).exists():
        needs_confirm = True
        confirm_message = f"write new file: {inp.get('file_path', '')}"
    elif tool_name == "edit_file" and not Path(inp.get("file_path", "")).exists():
        needs_confirm = True
        confirm_message = f"edit non-existent file: {inp.get('file_path', '')}"

    if needs_confirm:
        # dontAsk 模式：需要确认的操作自动拒绝
        if mode == "dontAsk":
            return {"action": "deny", "message": f"Auto-denied (dontAsk mode): {confirm_message}"}
        return {"action": "confirm", "message": confirm_message}

    return {"action": "allow"}


# ─── Truncate long tool results ─────────────────────────────
# 工具结果截断：防止过长的结果消耗过多 token

MAX_RESULT_CHARS = 50000
# 工具结果最大字符数，超出部分会被截断


def _truncate_result(result: str) -> str:
    """截断过长的工具结果，保留开头和结尾部分。"""
    if len(result) <= MAX_RESULT_CHARS:
        return result
    # 保留开头和结尾各一半，中间截断
    keep_each = (MAX_RESULT_CHARS - 60) // 2
    return (
        result[:keep_each]
        + f"\n\n[... truncated {len(result) - keep_each * 2} chars ...]\n\n"
        + result[-keep_each:]
    )


# ─── Execute a tool call ────────────────────────────────────
# 工具执行入口函数：分发到具体的工具处理函数
# "agent" and "skill" tools are handled in agent.py to avoid circular deps.
# 注意：agent 和 skill 工具在 agent.py 中处理，避免循环依赖


async def execute_tool(
    name: str, inp: dict, read_file_state: dict[str, float] | None = None
) -> str:
    """执行工具调用，返回工具结果字符串。"""
    # ─── read-before-edit + mtime freshness checks ───────────
    # 读取文件后记录修改时间，用于后续编辑前检查文件是否被外部修改
    if name == "read_file":
        result = _read_file(inp)
        if read_file_state is not None and not result.startswith("Error"):
            abs_path = str(Path(inp["file_path"]).resolve())
            try:
                # 记录文件的修改时间，用于检测外部修改
                read_file_state[abs_path] = os.path.getmtime(abs_path)
            except OSError:
                pass
        # Return the full result untruncated: the agent layer persists large
        # results to disk first (persistLargeResult), then truncates as a
        # safety net. Truncating here would destroy data before persistence.
        # 不截断读取结果：agent 层会先持久化大结果到磁盘，然后再截断
        return result

    # 编辑文件前检查：确保已读取文件且文件未被外部修改
    if name in ("write_file", "edit_file") and read_file_state is not None:
        abs_path = str(Path(inp["file_path"]).resolve())
        if os.path.exists(abs_path):
            # 检查是否已读取过该文件
            if abs_path not in read_file_state:
                verb = "writing" if name == "write_file" else "editing"
                return f"Error: You must read this file before {verb}. Use read_file first to see its current contents."
            # 检查文件是否在读取后被外部修改
            if os.path.getmtime(abs_path) != read_file_state[abs_path]:
                verb = "writing" if name == "write_file" else "editing"
                return f"Warning: {inp['file_path']} was modified externally since your last read. Please read_file again before {verb}."

    # tool_search: activate deferred tools and return their schemas
    # 工具搜索：激活延迟工具并返回其 schema
    if name == "tool_search":
        query = (inp.get("query") or "").lower()
        # 在延迟工具中搜索匹配的工具
        deferred = [t for t in tool_definitions if t.get("deferred")]
        matches = [
            t for t in deferred
            if query in t["name"].lower() or query in (t.get("description") or "").lower()
        ]
        if not matches:
            return "No matching deferred tools found."
        # 将匹配的工具添加到已激活集合
        for m in matches:
            _activated_tools.add(m["name"])
        # 返回工具的完整 schema 定义
        return json.dumps(
            [{"name": t["name"], "description": t.get("description", ""), "input_schema": t["input_schema"]} for t in matches],
            indent=2,
        )

    # 工具处理函数映射表
    handlers: dict = {
        "write_file": _write_file,
        "edit_file": _edit_file,
        "list_files": _list_files,
        "grep_search": _grep_search,
        "run_shell": _run_shell,
        "web_fetch": _web_fetch,
    }
    handler = handlers.get(name)
    if not handler:
        return f"Unknown tool: {name}"
    result = handler(inp)

    # Update mtime after successful write/edit
    # 编辑/写入成功后更新文件修改时间记录
    if name in ("write_file", "edit_file") and read_file_state is not None and not result.startswith("Error"):
        abs_path = str(Path(inp["file_path"]).resolve())
        try:
            read_file_state[abs_path] = os.path.getmtime(abs_path)
        except OSError:
            pass

    return result


def reset_permission_cache() -> None:
    """重置权限规则缓存。"""
    global _cached_rules
    _cached_rules = None
