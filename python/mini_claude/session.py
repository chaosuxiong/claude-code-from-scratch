"""Session management — JSON file persistence for conversation history."""
from __future__ import annotations

# 会话管理模块 — 使用 JSON 文件持久化存储对话历史。
# 本模块提供会话（Session）的保存、加载、列表查询和获取最新会话等功能。
# 所有会话数据以 JSON 文件形式存储在用户主目录下的 .mini-claude/sessions/ 目录中。
# 每个会话对应一个以 session_id 命名的 JSON 文件。

import json
from pathlib import Path
from typing import Any

# 会话文件存储目录：~/.mini-claude/sessions/
SESSION_DIR = Path.home() / ".mini-claude" / "sessions"


def _ensure_dir() -> None:
    """确保会话存储目录存在。

    如果目录不存在，则递归创建该目录及其所有父目录。
    如果目录已存在，则不做任何操作。
    """
    SESSION_DIR.mkdir(parents=True, exist_ok=True)


def save_session(session_id: str, data: dict[str, Any]) -> None:
    """保存会话数据到 JSON 文件。

    将会话数据序列化为 JSON 格式并写入文件。
    文件名为 session_id.json，存储在 SESSION_DIR 目录下。

    参数:
        session_id (str): 会话的唯一标识符，用作文件名。
        data (dict[str, Any]): 要保存的会话数据字典，可包含任意嵌套结构。
                               不可序列化的对象会通过 str() 转换为字符串。

    返回值:
        None（无返回值）
    """
    _ensure_dir()
    # 将数据序列化为 JSON 字符串，缩进为 2 个空格以便人类阅读
    # default=str 表示遇到无法序列化的对象时调用 str() 转换
    (SESSION_DIR / f"{session_id}.json").write_text(json.dumps(data, indent=2, default=str))


def load_session(session_id: str) -> dict[str, Any] | None:
    """从 JSON 文件加载会话数据。

    根据 session_id 查找对应的 JSON 文件并反序列化为字典。
    如果文件不存在或解析失败，返回 None。

    参数:
        session_id (str): 会话的唯一标识符，用于定位对应的 JSON 文件。

    返回值:
        dict[str, Any] | None: 成功时返回会话数据字典；
                               文件不存在或解析失败时返回 None。
    """
    path = SESSION_DIR / f"{session_id}.json"
    # 文件不存在时直接返回 None，避免后续读取报错
    if not path.exists():
        return None
    try:
        # 读取文件内容并解析为 Python 字典
        return json.loads(path.read_text())
    except Exception:
        # JSON 解析失败或其他 IO 错误时返回 None
        return None


def list_sessions() -> list[dict[str, Any]]:
    """列出所有已保存会话的元数据。

    遍历 SESSION_DIR 目录下的所有 .json 文件，
    读取每个文件中 "metadata" 字段并收集到列表中返回。
    解析失败的文件会被静默跳过。

    参数:
        无

    返回值:
        list[dict[str, Any]]: 包含所有会话元数据字典的列表。
                              每个字典对应一个会话文件中的 "metadata" 字段。
                              如果没有任何有效会话，返回空列表。
    """
    _ensure_dir()
    results = []
    # 遍历目录下所有 .json 文件
    for f in SESSION_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            # 只收集包含 "metadata" 字段的会话数据
            if "metadata" in data:
                results.append(data["metadata"])
        except Exception:
            # 静默跳过无法解析的文件（如文件损坏或格式错误）
            pass
    return results


def get_latest_session_id() -> str | None:
    """获取最近一次会话的 ID。

    通过比较所有会话的 "startTime" 字段，找到时间最新的会话并返回其 ID。
    如果没有任何已保存的会话，返回 None。

    参数:
        无

    返回值:
        str | None: 最近会话的 ID 字符串；
                    如果没有会话则返回 None。
    """
    sessions = list_sessions()
    # 没有任何会话时直接返回 None
    if not sessions:
        return None
    # 按 startTime 降序排序（最新的排在最前面）
    sessions.sort(key=lambda s: s.get("startTime", ""), reverse=True)
    # 返回排序后第一个会话的 ID
    return sessions[0].get("id")
