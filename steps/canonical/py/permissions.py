"""
权限控制模块 — 权限门禁系统

本模块实现了一个轻量级的权限检查机制，用于在执行工具调用（特别是 shell 命令）
之前判断该操作是否被允许。核心思路是：维护一份危险命令的正则模式列表，
当检测到即将执行的 shell 命令匹配其中任一模式时，拒绝该操作；其余一律放行。

真实版本的 Claude Code 拥有更复杂的多层权限模型（如 ask / accept-edits / yolo
等模式），这里仅保留最核心的"执行前检查"理念。
"""

import re

# A tiny permission gate: dangerous shell commands are blocked, everything else
# runs. The real Claude Code has modes (ask / accept-edits / yolo) and layered
# rules; here we keep just the essential idea — check before you run.
#region permissions

# 危险命令的正则模式列表
# 每个模式都使用 \b（单词边界）来避免误匹配，例如不会把 "drm" 误判为 "rm"
_DANGEROUS = [
    r"\brm\s+-rf\b",           # 递归强制删除文件/目录（rm -rf）
    r"\bgit\s+push\b",         # 推送到远程仓库（git push）
    r"\bgit\s+reset\s+--hard\b",  # 硬重置，丢弃所有未提交的更改（git reset --hard）
    r"\bsudo\b",               # 以超级用户权限执行命令（sudo）
    r"\mkfs\b",                # 创建文件系统，会格式化磁盘（mkfs）
    r">\s*/dev/",              # 写入 /dev/ 设备文件，可能导致数据损坏
]


def check_permission(name: str, inp: dict) -> str:
    """
    检查给定的工具调用是否被允许执行。

    当前仅对 "run_shell" 类型的工具调用进行危险命令检测；
    其他类型的工具调用一律放行。

    参数:
        name (str): 工具名称，例如 "run_shell"。
        inp (dict): 工具调用的输入参数字典。
                    当 name 为 "run_shell" 时，应包含 "command" 键。

    返回:
        str: "deny" 表示拒绝执行，"allow" 表示允许执行。
    """
    # 仅当工具名为 "run_shell" 时才进行危险命令检查
    if name == "run_shell" and any(re.search(p, str(inp.get("command", ""))) for p in _DANGEROUS):
        return "deny"  # 命令匹配任一危险模式，拒绝执行
    return "allow"     # 未检测到危险，允许执行
#endregion
