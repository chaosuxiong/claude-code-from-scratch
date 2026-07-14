"""
prompt.py — 系统提示词构建模块
=================================
本模块负责组装发送给大语言模型（LLM）的系统提示词（system prompt）。
系统提示词由两部分组成：
  1. 静态核心（STATIC_CORE）：定义助手的身份、行为规则和工具使用偏好，
     内容在所有会话间保持不变，因此可以被 LLM 的缓存机制复用。
  2. 动态环境上下文（_environment_context）：每次运行时实时采集的工作目录、
     操作系统平台、Shell 类型和 Git 分支等信息，确保模型了解当前执行环境。
最终由 build_system_prompt() 将两部分拼接为完整的系统提示词字符串。
"""

import os        # 用于获取当前工作目录（os.getcwd）和环境变量（os.environ）
import platform  # 用于获取操作系统平台信息（如 Linux、Darwin）和架构（如 x86_64、arm64）
import subprocess  # 用于执行外部命令（如 git），采集版本控制信息

# The static core: identity, rules, and tool preferences. Byte-identical across
# sessions, which is exactly what makes it cacheable (a real agent marks this
# block with cache_control).
# 静态核心部分：定义助手的身份、行为规则和工具偏好。
# 该内容在所有会话中完全相同（字节级一致），因此可以被 LLM 端的缓存机制
# （如 cache_control 标记）高效复用，避免每次都重新处理。
#region static_core
STATIC_CORE = """You are Mini Claude Code, a small coding assistant CLI.
You help with software engineering tasks using the tools available to you.

# Doing tasks
 - Do not propose changes to code you haven't read. Read files first.
 - Do not create files unless necessary. Prefer editing existing files.
 - Avoid over-engineering. Only make changes that were requested.

# Executing actions with care
 - Prefer reversible actions. For risky or destructive ones (rm -rf, git push,
   dropping tables), confirm with the user before proceeding.

# Using your tools
 - Use read_file / edit_file / list_files / grep_search instead of shell cat,
   sed, ls, grep. Reserve run_shell for actual shell operations.
 - If several tool calls are independent, make them in parallel.

# Tone and style
 - Keep responses short and concise. Lead with the answer.
 - Reference code as file_path:line_number."""
#endregion


# The dynamic half: environment facts assembled fresh each run. Kept separate
# from the static core so it never pollutes the cache.
# 动态环境上下文函数：每次运行时实时采集环境信息。
# 与静态核心分开维护，确保动态数据不会污染 LLM 的缓存。
def _environment_context() -> str:
    """
    采集当前运行环境的关键信息，返回格式化的环境描述字符串。

    采集的信息包括：
      - 当前工作目录（Working directory）
      - 操作系统平台与架构（Platform）
      - 默认 Shell 类型（Shell）
      - 当前 Git 分支（Git branch，如果处于 Git 仓库中）

    Returns:
        str: 格式化的环境信息字符串，以 "# Environment" 为标题。
    """
    git = ""  # 初始化 Git 分支信息为空字符串
    try:
        # 执行 git 命令获取当前分支名称
        # capture_output=True: 捕获 stdout 和 stderr
        # text=True: 以字符串形式返回输出（而非 bytes）
        # timeout=3: 设置 3 秒超时，避免命令挂起
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()  # 去除输出末尾的换行符和空白字符
        if branch:
            git = f"\nGit branch: {branch}"  # 仅在成功获取分支名时才添加 Git 信息
    except Exception:
        pass  # 如果不在 Git 仓库中，或 git 命令不可用，则静默忽略错误
    return (
        "# Environment\n"
        f"Working directory: {os.getcwd()}\n"  # 当前工作目录的绝对路径
        f"Platform: {platform.system()} {platform.machine()}\n"  # 如 "Linux x86_64"
        f"Shell: {os.environ.get('SHELL', '/bin/sh')}{git}"  # 默认 Shell 路径，未设置时回退到 /bin/sh
    )


# Static core first, then the environment block.
# 构建完整的系统提示词：先拼接静态核心，再附加动态环境信息。
def build_system_prompt() -> str:
    """
    构建并返回完整的系统提示词字符串。

    将静态核心（STATIC_CORE）与动态环境上下文（_environment_context()）
    拼接在一起，形成发送给 LLM 的完整 system prompt。

    Returns:
        str: 完整的系统提示词，包含助手身份定义和当前环境信息。
    """
    return f"{STATIC_CORE}\n\n{_environment_context()}"
