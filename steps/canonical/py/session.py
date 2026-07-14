"""
会话管理模块 - 负责对话历史的持久化存储和加载

本模块提供了简单的会话持久化功能，将对话消息列表保存到磁盘文件，
并在需要时重新加载。使用 JSON 格式存储，无需数据库支持。

主要功能：
- save_session(): 将当前对话消息保存到磁盘文件
- load_session(): 从磁盘文件加载之前的对话历史

存储方式：
- 使用 JSON 文件格式存储会话数据
- 文件路径为当前工作目录下的 .mini-session.json
- 每次对话轮次后自动保存，支持通过 --resume 参数恢复会话
"""

import json
import os

# The session is just the message list on disk. Save after every turn; load it
# back on --resume. No database — the whole conversation is already a plain list.
# 会话就是磁盘上的消息列表。每次对话后保存；通过 --resume 参数加载回来。
# 不需要数据库——整个对话本身就是一个简单的列表。
SESSION_FILE = os.path.join(os.getcwd(), ".mini-session.json")
# 会话文件路径：当前工作目录下的 .mini-session.json 文件


#region session
def save_session(messages) -> None:
    """
    将对话消息列表保存到磁盘文件

    功能：
    - 将当前的对话消息序列化为 JSON 格式
    - 写入到 SESSION_FILE 指定的文件中
    - 如果保存失败则静默忽略错误

    参数：
        messages (list): 对话消息列表，每个消息是一个字典对象

    返回值：
        None: 无返回值

    异常处理：
        - 捕获所有异常并静默处理，确保保存操作不会中断程序运行
    """
    try:
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            # 使用 json.dump 将消息列表序列化并写入文件
            # indent=2: 格式化输出，便于人工阅读
            # default=lambda o: getattr(o, "model_dump", lambda: str(o))():
            #   处理无法直接序列化的对象，优先使用 model_dump() 方法，否则转为字符串
            json.dump(messages, f, indent=2, default=lambda o: getattr(o, "model_dump", lambda: str(o))())
    except Exception:
        # 静默处理所有异常，确保存储失败不会影响主程序运行
        pass


def load_session():
    """
    从磁盘文件加载对话历史

    功能：
    - 检查会话文件是否存在
    - 如果存在，读取并解析 JSON 格式的对话历史
    - 如果文件不存在或读取失败，返回 None

    参数：
        无参数

    返回值：
        list or None:
            - 成功加载时返回消息列表
            - 文件不存在或加载失败时返回 None

    异常处理：
        - 捕获所有异常并返回 None，确保加载操作不会中断程序运行
    """
    # 检查会话文件是否存在
    if not os.path.exists(SESSION_FILE):
        return None  # 文件不存在，返回 None

    try:
        with open(SESSION_FILE, encoding="utf-8") as f:
            # 从 JSON 文件读取并解析消息列表
            return json.load(f)
    except Exception:
        # 文件存在但读取或解析失败，返回 None
        return None
#endregion