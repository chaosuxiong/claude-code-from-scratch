"""Terminal UI rendering — colored output, spinner, tool display."""

# =============================================================================
# 终端 UI 渲染模块 (Terminal UI Rendering)
#
# 本模块负责 Mini Claude Code 的终端界面显示，包括：
# - 彩色输出：使用 Rich 库实现语法高亮和彩色文本
# - 旋转加载指示器（Spinner）：在等待 AI 响应时显示加载动画
# - 工具调用显示：展示工具名称、参数和执行结果
# - 计划审批界面：显示实现计划并提供选项
# - 子代理显示：展示子代理的启动和完成状态
# - 成本统计：显示 token 使用量和预估费用
#
# 所有输出都通过 Rich Console 库实现，支持 ANSI 颜色代码。
# =============================================================================

from __future__ import annotations

import sys
import threading
import time

from rich.console import Console

console = Console(highlight=False)
# 创建 Rich Console 实例，禁用语法高亮以避免干扰输出

# ─── Basic output ──────────────────────────────────────────
# 基础输出函数：显示欢迎信息、用户输入提示、助手响应等


def print_welcome() -> None:
    """打印欢迎信息和使用说明。"""
    console.print("\n  [bold cyan]Mini Claude Code[/bold cyan][dim] — A minimal coding agent[/dim]\n")
    console.print("[dim]  Type your request, or 'exit' to quit.[/dim]")
    console.print("[dim]  Commands: /clear /plan /cost /compact /memory /skills[/dim]\n")


def print_user_prompt() -> None:
    """打印用户输入提示符。"""
    console.print("\n[bold green]> [/bold green]", end="")


def print_assistant_text(text: str) -> None:
    """直接输出助手的响应文本（不换行）。"""
    sys.stdout.write(text)
    sys.stdout.flush()


def print_tool_call(name: str, inp: dict) -> None:
    """显示工具调用信息，包括图标、工具名和参数摘要。"""
    icon = _get_tool_icon(name)
    summary = _get_tool_summary(name, inp)
    console.print(f"\n  [yellow]{icon} {name}[/yellow][dim] {summary}[/dim]")


def print_tool_result(name: str, result: str) -> None:
    """显示工具执行结果。文件编辑类工具显示差异，其他工具显示截断后的文本。"""
    # 文件编辑类工具使用差异格式显示
    if (name in ("edit_file", "write_file")) and not result.startswith("Error"):
        _print_file_change_result(name, result)
        return
    # 其他工具：截断过长的结果（最多显示 500 字符）
    max_len = 500
    truncated = result
    if len(result) > max_len:
        truncated = result[:max_len] + f"\n  ... ({len(result)} chars total)"
    # 每行添加缩进
    lines = "\n".join("  " + l for l in truncated.split("\n"))
    console.print(f"[dim]{lines}[/dim]")


def _print_file_change_result(_name: str, result: str) -> None:
    """以差异格式显示文件编辑结果，使用颜色区分新增和删除的行。"""
    lines = result.split("\n")
    # 显示第一行（成功消息）
    console.print(f"[dim]  {lines[0]}[/dim]")

    max_display = 40
    content_lines = lines[1:]
    display_lines = content_lines[:max_display]

    for line in display_lines:
        if not line.strip():
            continue
        # 使用不同颜色显示 diff 的不同部分
        if line.startswith("@@"):
            # @@ 行（差异头）：青色
            console.print(f"[cyan]  {line}[/cyan]")
        elif line.startswith("- "):
            # 删除的行：红色
            console.print(f"[red]  {line}[/red]")
        elif line.startswith("+ "):
            # 新增的行：绿色
            console.print(f"[green]  {line}[/green]")
        else:
            # 上下文行：灰色
            console.print(f"[dim]  {line}[/dim]")
    # 如果内容超过最大显示行数，显示省略信息
    if len(content_lines) > max_display:
        console.print(f"[dim]  ... ({len(content_lines) - max_display} more lines)[/dim]")


def print_error(msg: str) -> None:
    """显示红色错误消息。"""
    console.print(f"\n  [red]Error: {msg}[/red]")


def print_confirmation(command: str) -> None:
    """显示危险命令确认提示。"""
    console.print(f"\n  [yellow]⚠ Dangerous command:[/yellow] [white]{command}[/white]")


def print_divider() -> None:
    """打印分隔线。"""
    console.print(f"\n[dim]  {'─' * 50}[/dim]")


def print_cost(input_tokens: int, output_tokens: int, cache_read: int = 0, cache_creation: int = 0) -> None:
    """显示 token 使用量和预估费用。

    费用计算（基于 Claude 3 Opus 定价）：
    - 输入 token：$3/百万
    - 缓存读取：$0.3/百万（0.1x）
    - 缓存创建：$3.75/百万（1.25x）
    - 输出 token：$15/百万
    """
    # Cache read is billed 0.1x, cache write 1.25x (see agent _get_current_cost_usd).
    total = (
        (input_tokens / 1_000_000) * 3
        + (cache_read / 1_000_000) * 0.3
        + (cache_creation / 1_000_000) * 3.75
        + (output_tokens / 1_000_000) * 15
    )
    cache_str = f", {cache_read} cached" if cache_read else ""
    console.print(f"\n[dim]  Tokens: {input_tokens} in / {output_tokens} out{cache_str} (~${total:.4f})[/dim]")


def print_retry(attempt: int, max_retries: int, reason: str) -> None:
    """显示重试信息。"""
    console.print(f"\n  [yellow]↻ Retry {attempt}/{max_retries}: {reason}[/yellow]")


def print_info(msg: str) -> None:
    """显示青色信息消息。"""
    console.print(f"\n  [cyan]ℹ {msg}[/cyan]")


# ─── Spinner ──────────────────────────────────────────────
# 旋转加载指示器：在等待 AI 响应时显示动画，提供视觉反馈

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
# Spinner 动画帧序列（Braille 字符）

_spinner_thread: threading.Thread | None = None
_spinner_stop = threading.Event()


def start_spinner(label: str = "Thinking") -> None:
    """启动旋转加载指示器。

    在后台线程中运行，每 0.08 秒更新一帧动画。
    用于在等待 AI 响应时提供视觉反馈。
    """
    global _spinner_thread
    if _spinner_thread is not None:
        return
    _spinner_stop.clear()

    def _run() -> None:
        """Spinner 动画线程主循环。"""
        frame = 0
        sys.stdout.write(f"\n  {SPINNER_FRAMES[0]} {label}...")
        sys.stdout.flush()
        while not _spinner_stop.is_set():
            time.sleep(0.08)
            # 循环切换动画帧
            frame = (frame + 1) % len(SPINNER_FRAMES)
            # 使用 \r 回到行首，覆盖上一帧
            sys.stdout.write(f"\r  {SPINNER_FRAMES[frame]} {label}...")
            sys.stdout.flush()

    _spinner_thread = threading.Thread(target=_run, daemon=True)
    _spinner_thread.start()


def stop_spinner() -> None:
    """停止旋转加载指示器。"""
    global _spinner_thread
    if _spinner_thread is None:
        return
    # 发送停止信号并等待线程结束
    _spinner_stop.set()
    _spinner_thread.join(timeout=1)
    _spinner_thread = None
    # 清除 spinner 行（\r 回到行首，\033[K 删除该行）
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


# ─── Plan approval display ──────────────────────────────────
# 计划审批界面：显示实现计划并提供执行选项


def print_plan_for_approval(plan_content: str) -> None:
    """显示实现计划内容，用于用户审批。最多显示 60 行。"""
    console.print("\n  [cyan]━━━ Plan for Approval ━━━[/cyan]")
    lines = plan_content.split("\n")
    max_lines = 60
    for line in lines[:max_lines]:
        console.print(f"  [white]{line}[/white]")
    if len(lines) > max_lines:
        console.print(f"[dim]  ... ({len(lines) - max_lines} more lines)[/dim]")
    console.print("  [cyan]━━━━━━━━━━━━━━━━━━━━━━━━[/cyan]\n")


def print_plan_approval_options() -> None:
    """显示计划审批选项菜单。"""
    console.print("  [yellow]Choose an option:[/yellow]")
    console.print("    [white]1) Yes, clear context and execute[/white][dim] — fresh start with auto-accept edits[/dim]")
    console.print("    [white]2) Yes, and execute[/white][dim] — keep context, auto-accept edits[/dim]")
    console.print("    [white]3) Yes, manually approve edits[/white][dim] — keep context, confirm each edit[/dim]")
    console.print("    [white]4) No, keep planning[/white][dim] — provide feedback to revise[/dim]")


# ─── Sub-agent display ──────────────────────────────────────
# 子代理显示：展示子代理的启动和完成状态


def print_sub_agent_start(agent_type: str, description: str) -> None:
    """显示子代理启动信息。"""
    console.print(f"\n  [magenta]┌─ Sub-agent [{agent_type}]: {description}[/magenta]")


def print_sub_agent_end(agent_type: str, _description: str) -> None:
    """显示子代理完成信息。"""
    console.print(f"  [magenta]└─ Sub-agent [{agent_type}] completed[/magenta]")


# ─── Tool icons and summaries ───────────────────────────────
# 工具图标和摘要：为每个工具类型定义显示图标和参数摘要

_TOOL_ICONS = {
    "read_file": "📖",
    "write_file": "✏️",
    "edit_file": "🔧",
    "list_files": "📁",
    "grep_search": "🔍",
    "run_shell": "💻",
    "skill": "⚡",
    "agent": "🤖",
}


def _get_tool_icon(name: str) -> str:
    """获取工具对应的图标，未知工具使用默认图标。"""
    return _TOOL_ICONS.get(name, "🔨")


def _get_tool_summary(name: str, inp: dict) -> str:
    """生成工具调用的摘要信息（用于终端显示）。"""
    if name == "read_file":
        return inp.get("file_path", "")
    if name == "write_file":
        return inp.get("file_path", "")
    if name == "edit_file":
        return inp.get("file_path", "")
    if name == "list_files":
        return inp.get("pattern", "")
    if name == "grep_search":
        return f'"{inp.get("pattern", "")}" in {inp.get("path", ".")}'
    if name == "run_shell":
        # 命令过长时截断显示
        cmd = inp.get("command", "")
        return cmd[:60] + "..." if len(cmd) > 60 else cmd
    if name == "skill":
        return inp.get("skill_name", "")
    if name == "agent":
        return f'[{inp.get("type", "general")}] {inp.get("description", "")}'
    return ""
