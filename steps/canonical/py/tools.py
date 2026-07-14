"""
工具定义与调度模块

本模块定义了 AI 编程助手中使用的所有工具（Tool），包括工具的元数据定义和实际执行逻辑。
每个工具由三部分组成：名称（name）、描述（description，供模型阅读理解）和执行函数。
这些定义与 API 所期望的格式完全一致。

当前支持的工具：
- read_file: 读取文件内容
- write_file: 写入文件内容
- edit_file: 精确替换文件中的字符串
- list_files: 按 glob 模式列出文件
- grep_search: 使用正则表达式在文件中搜索
- run_shell: 执行 shell 命令
- agent: 委托子代理进行只读调查（从 step >=11 开始可用）
"""

import os
import re
import subprocess

# A tool is three things: a name, a description the model reads, and a function
# that does the work. The definitions below are exactly the shape the API wants.
# 工具定义列表：每个工具包含名称、描述（供模型理解）和输入参数的 JSON Schema。
# 这些定义与 Anthropic API 的 tool 参数格式完全匹配。
tool_definitions = [
    {
        "name": "read_file",
        "description": "Read the contents of a file. Returns the file content with line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string", "description": "The path to the file to read"}},
            "required": ["file_path"],
        },
    },
#step >=2
    # write_file 工具：创建新文件或覆盖已有文件
    {
        "name": "write_file",
        "description": "Write content to a file. Creates it if missing, overwrites if it exists.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The path to the file to write"},
                "content": {"type": "string", "description": "The content to write"},
            },
            "required": ["file_path", "content"],
        },
    },
    # edit_file 工具：精确查找并替换文件中的字符串，要求匹配内容必须唯一
    {
        "name": "edit_file",
        "description": "Replace an exact string in a file with new content. old_string must match exactly and be unique.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The path to the file to edit"},
                "old_string": {"type": "string", "description": "The exact string to find"},
                "new_string": {"type": "string", "description": "The string to replace it with"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    # list_files 工具：按 glob 模式匹配并列出文件，支持递归搜索
    {
        "name": "list_files",
        "description": "List files matching a glob pattern (e.g. \"**/*.py\").",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern to match files"},
                "path": {"type": "string", "description": "Base directory. Defaults to cwd."},
            },
            "required": ["pattern"],
        },
    },
    # grep_search 工具：使用正则表达式在文件中搜索，返回匹配的行及其路径和行号
    {
        "name": "grep_search",
        "description": "Search for a regex pattern in files. Returns matching lines with paths and line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "The regex pattern to search for"},
                "path": {"type": "string", "description": "Directory or file to search. Defaults to cwd."},
            },
            "required": ["pattern"],
        },
    },
    # run_shell 工具：执行 shell 命令并返回输出，适用于运行测试、git 操作、安装包等
    {
        "name": "run_shell",
        "description": "Execute a shell command and return its output. For tests, git, package installs, etc.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The shell command to execute"}},
            "required": ["command"],
        },
    },
#endstep
#step >=11
    # agent 工具：将只读调查任务委托给子代理，子代理自主探索并返回摘要报告
    {
        "name": "agent",
        "description": "Delegate a read-only investigation to a sub-agent. Give it a task; it explores on its own and reports back a summary.",
        "input_schema": {
            "type": "object",
            "properties": {"task": {"type": "string", "description": "The task for the sub-agent to investigate"}},
            "required": ["task"],
        },
    },
#endstep
]


# Dispatch a tool call by name. Unknown names return an error string instead of
# raising, so a hallucinated tool name lets the model self-correct.
#region dispatch
def execute_tool(name: str, inp: dict) -> str:
    """根据工具名称分发执行对应的工具函数。

    这是工具调度的核心函数，通过名称匹配将请求路由到具体的实现函数。
    对于未知的工具名称，返回错误字符串而非抛出异常，这样模型可以自行纠正错误的工具调用。

    Args:
        name (str): 工具名称，如 "read_file"、"write_file" 等
        inp (dict): 工具的输入参数字典，结构与对应工具的 input_schema 匹配

    Returns:
        str: 工具执行的结果文本，或错误提示信息
    """
    if name == "read_file":
        return _read_file(inp)
#step >=2
    if name == "write_file":
        return _write_file(inp)
    if name == "edit_file":
        return _edit_file(inp)
    if name == "list_files":
        return _list_files(inp)
    if name == "grep_search":
        return _grep_search(inp)
    if name == "run_shell":
        return _run_shell(inp)
#endstep
    return f"Unknown tool: {name}"
#endregion


#region read_file
def _read_file(inp: dict) -> str:
    """读取指定文件的内容，并为每行添加行号。

    以 UTF-8 编码读取文件，返回带行号的内容，方便模型定位代码位置。

    Args:
        inp (dict): 输入参数，必须包含 "file_path" 键，值为文件路径字符串

    Returns:
        str: 带行号的文件内容，格式为 "行号 | 内容"；出错时返回错误信息
    """
    try:
        # 按行分割文件内容，便于后续添加行号
        lines = open(inp["file_path"], encoding="utf-8").read().split("\n")
        # 为每行添加从 1 开始的行号，使用 4 位宽度右对齐
        return "\n".join(f"{i + 1:4d} | {line}" for i, line in enumerate(lines))
    except Exception as e:
        return f"Error reading file: {e}"
#endregion


#step >=2
def _write_file(inp: dict) -> str:
    """将内容写入指定文件。如果文件不存在则创建，如果存在则覆盖。

    会自动创建文件所需的目录结构。

    Args:
        inp (dict): 输入参数，必须包含：
            - "file_path" (str): 目标文件路径
            - "content" (str): 要写入的内容

    Returns:
        str: 成功时返回写入确认信息（包含行数）；失败时返回错误信息
    """
    try:
        # 获取文件所在目录路径
        d = os.path.dirname(inp["file_path"])
        # 如果目录不存在，递归创建目录
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        with open(inp["file_path"], "w", encoding="utf-8") as f:
            f.write(inp["content"])
        # 统计写入的行数，用于反馈给模型
        n = len(inp["content"].split("\n"))
        return f"Successfully wrote to {inp['file_path']} ({n} lines)"
    except Exception as e:
        return f"Error writing file: {e}"


# edit_file is the one tool with a real trap: the match must be unique, or you
# edit the wrong place. So we count occurrences and refuse if it isn't unique.
# edit_file 工具有一个关键陷阱：匹配的字符串必须唯一，否则可能替换到错误的位置。
# 因此函数会统计出现次数，如果不唯一则拒绝执行。
#region edit_file
def _edit_file(inp: dict) -> str:
    """在文件中精确查找并替换字符串。

    该函数要求 old_string 在文件中必须存在且唯一，否则会返回错误。
    这是为了防止意外替换到错误位置。

    Args:
        inp (dict): 输入参数，必须包含：
            - "file_path" (str): 目标文件路径
            - "old_string" (str): 要查找的原始字符串（必须精确匹配）
            - "new_string" (str): 替换后的新字符串

    Returns:
        str: 成功时返回编辑确认信息；失败时返回错误信息（未找到、不唯一等）
    """
    try:
        content = open(inp["file_path"], encoding="utf-8").read()
        # 检查 old_string 是否存在于文件中
        if inp["old_string"] not in content:
            return f"Error: old_string not found in {inp['file_path']}"
        # 统计 old_string 出现的次数，确保唯一性
        count = content.count(inp["old_string"])
        if count > 1:
            return f"Error: old_string found {count} times in {inp['file_path']}. Must be unique."
        # 执行替换操作
        updated = content.replace(inp["old_string"], inp["new_string"])
        with open(inp["file_path"], "w", encoding="utf-8") as f:
            f.write(updated)
        return f"Successfully edited {inp['file_path']}"
    except Exception as e:
        return f"Error editing file: {e}"
#endregion


def _list_files(inp: dict) -> str:
    """按 glob 模式列出匹配的文件列表。

    支持递归搜索，自动排除 node_modules 和 .git 目录。
    最多返回 200 个文件结果，避免输出过大。

    Args:
        inp (dict): 输入参数，必须包含：
            - "pattern" (str): glob 匹配模式，如 "**/*.py"
            - "path" (str, 可选): 搜索的基准目录，默认为当前目录 "."

    Returns:
        str: 匹配文件路径列表（换行分隔）；无匹配时返回提示信息；出错时返回错误信息
    """
    import glob as globmod

    try:
        # 获取基准目录，默认为当前工作目录
        base = inp.get("path") or "."
        # 使用 glob 匹配文件，并过滤掉 node_modules 和 .git 目录中的文件
        hits = [
            f for f in globmod.glob(os.path.join(base, inp["pattern"]), recursive=True)
            if os.path.isfile(f) and "node_modules" not in f and "/.git/" not in f
        ]
        # 最多返回 200 个结果，防止输出过多
        return "\n".join(hits[:200]) if hits else "No files found matching the pattern."
    except Exception as e:
        return f"Error listing files: {e}"


def _grep_search(inp: dict) -> str:
    """使用正则表达式在文件中搜索匹配的内容。

    优先使用系统 grep 命令以获得更好的性能；如果系统中没有 grep，
    则回退到 Python 实现的搜索函数。

    Args:
        inp (dict): 输入参数，必须包含：
            - "pattern" (str): 正则表达式搜索模式
            - "path" (str, 可选): 搜索的目录或文件路径，默认为当前目录 "."

    Returns:
        str: 匹配结果（包含文件路径、行号和内容）；无匹配时返回提示信息；出错时返回错误信息
    """
    # Prefer the system grep; fall back to a tiny Python walker if it isn't there.
    try:
        # 使用系统 grep 命令进行搜索，设置 10 秒超时
        out = subprocess.run(
            ["grep", "--line-number", "--color=never", "-r", "--", inp["pattern"], inp.get("path") or "."],
            capture_output=True, text=True, timeout=10,
        )
        # grep 返回码为 1 表示没有匹配结果
        if out.returncode == 1:
            return "No matches found."
        lines = [ln for ln in out.stdout.split("\n") if ln]
        # 最多返回 100 行匹配结果
        return "\n".join(lines[:100]) if lines else "No matches found."
    except FileNotFoundError:
        # 系统中没有 grep 命令，回退到 Python 实现
        return _grep_py(inp["pattern"], inp.get("path") or ".")
    except Exception as e:
        return f"Error: {e}"


def _grep_py(pattern: str, base: str) -> str:
    """Python 实现的文件内容搜索函数，作为系统 grep 的回退方案。

    使用 os.walk 遍历目录树，逐行匹配正则表达式。
    自动跳过隐藏目录（以 . 开头）和 node_modules 目录。

    Args:
        pattern (str): 正则表达式搜索模式
        base (str): 搜索的基准目录路径

    Returns:
        str: 匹配结果列表（格式为 "文件路径:行号:内容"），最多 100 条；
             无匹配时返回 "No matches found."；正则无效时返回错误信息
    """
    try:
        # 编译正则表达式，提前检查语法是否正确
        rx = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex: {e}"
    matches: list[str] = []
    # 使用 os.walk 递归遍历目录树
    for root, dirs, files in os.walk(base):
        # 过滤掉隐藏目录和 node_modules 目录，避免搜索不必要的文件
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "node_modules"]
        for name in files:
            full = os.path.join(root, name)
            try:
                # 逐行读取文件并进行正则匹配
                for i, line in enumerate(open(full, encoding="utf-8"), 1):
                    if rx.search(line) and len(matches) < 100:
                        matches.append(f"{full}:{i}:{line.rstrip()}")
            except Exception:
                pass  # 跳过无法读取的文件（如二进制文件）
    return "\n".join(matches) if matches else "No matches found."


def _run_shell(inp: dict) -> str:
    """执行 shell 命令并返回输出结果。

    使用 shell=True 模式执行命令，支持管道、重定向等 shell 特性。
    设置 30 秒超时，防止命令长时间挂起。

    Args:
        inp (dict): 输入参数，必须包含：
            - "command" (str): 要执行的 shell 命令

    Returns:
        str: 命令的标准输出；命令失败时返回退出码和错误信息；超时时返回超时提示
    """
    try:
        # 使用 shell 模式执行命令，捕获输出，超时 30 秒
        r = subprocess.run(inp["command"], shell=True, capture_output=True, text=True, timeout=30)
        # 非零退出码表示命令执行失败
        if r.returncode != 0:
            return f"Command failed (exit {r.returncode})\nStdout: {r.stdout}\nStderr: {r.stderr}"
        return r.stdout or "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 30000ms"
    except Exception as e:
        return f"Error: {e}"
#endstep
