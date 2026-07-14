"""
跨会话记忆模块（Cross-session Memory）

本模块实现了基于文件系统的简易跨会话记忆机制。它将用户的关键信息以 Markdown 文件的形式
保存在 .mini-memory/ 目录下。在每次对话开始时，系统会根据用户当前的提问内容，通过
关键词匹配的方式检索相关记忆，并将匹配结果注入到系统提示词（system prompt）中，从而让
AI 能够"记住"之前会话中的重要事实。

记忆检索采用确定性的关键词重叠算法——不需要调用模型，也不需要向量嵌入，仅通过文本分词和
词频交集来计算相关性得分，足以演示记忆召回的基本机制。
"""

import os
import re

# Cross-session memory: small facts saved as files under .mini-memory/. Before
# each turn we recall the ones relevant to what the user asked and drop them
# into the system prompt. Recall is deterministic keyword overlap — no model
# call, no embeddings — enough to see the mechanism.
# 跨会话记忆目录：将小段事实以文件形式保存在 .mini-memory/ 下。在每次对话轮次开始前，
# 系统会检索与用户当前提问相关的记忆，并将其注入到系统提示词中。检索采用确定性的关键词
# 重叠算法——无需模型调用，无需向量嵌入——足以展示记忆召回机制的运作方式。
MEMORY_DIR = os.path.join(os.getcwd(), ".mini-memory")  # 记忆文件存储目录的绝对路径


#region recall
def recall_memories(query: str) -> str:
    """
    根据用户查询内容，从记忆目录中检索相关记忆并返回格式化结果。

    本函数通过以下步骤实现记忆召回：
    1. 检查记忆目录是否存在，不存在则直接返回空字符串
    2. 将查询文本分词，过滤掉长度 <= 2 的短词（如 "is", "a" 等无意义词）
    3. 遍历记忆目录中所有 .md 文件，计算每个文件与查询的关键词重叠数
    4. 按得分降序排列，取前 3 条最相关的记忆，格式化后返回

    Args:
        query (str): 用户的输入查询文本，用于提取关键词进行记忆匹配。

    Returns:
        str: 如果找到相关记忆，返回格式化的 Markdown 字符串，包含标题和记忆列表；
             如果没有相关记忆或记忆目录不存在，返回空字符串。
    """
    # 如果记忆目录不存在，说明从未保存过记忆，直接返回空
    if not os.path.isdir(MEMORY_DIR):
        return ""

    # 将查询文本分词：按非单词字符分割，转小写，过滤掉长度 <= 2 的短词
    # 例如 "How do I fix this bug" -> {"how", "fix", "this", "bug"}
    query_words = {w for w in re.split(r"\W+", query.lower()) if len(w) > 2}

    scored = []  # 存储 (得分, 文件内容) 元组的列表
    for name in os.listdir(MEMORY_DIR):
        # 只处理 .md 格式的记忆文件
        if not name.endswith(".md"):
            continue
        # 读取记忆文件内容并去除首尾空白
        text = open(os.path.join(MEMORY_DIR, name), encoding="utf-8").read().strip()
        # 将文件内容分词为集合，用于后续关键词交集计算
        words = set(re.split(r"\W+", text.lower()))
        # 计算得分：统计查询关键词中有多少个也出现在该记忆文件中
        score = sum(1 for w in query_words if w in words)
        if score > 0:
            scored.append((score, text))

    # 如果没有任何记忆匹配成功，返回空字符串
    if not scored:
        return ""

    # 按得分降序排列，取前 3 条最高分的记忆，格式化为 Markdown 列表
    top = "\n".join(f"- {t}" for _, t in sorted(scored, key=lambda s: -s[0])[:3])
    return f"\n\n# Memory (things you remember about the user and project)\n{top}"
#endregion
