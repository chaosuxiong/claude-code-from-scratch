"""
自主性模块 - 实现无人值守的智能代理功能
==========================================

本模块提供两个核心功能，用于支持智能代理在无需人工干预的情况下持续运行：

1. 目标评估 (Goal Evaluation):
   - 通过 /goal 命令附加停止条件
   - 独立的评估器在每轮对话后判断条件是否满足
   - 若未满足，重新注入原因继续执行

2. 动作分类 (Action Classification):
   - --auto 模式下替代人工确认提示
   - 分类器读取对话记录，决定允许或阻止操作
   - 两者都是一次性的模型侧调用，独立于主循环

Autonomy: keep the agent working across many turns without a human at each
step. /goal attaches a stop condition and an independent evaluator judges,
after every turn, whether it's met — reinjecting the reason if not. --auto
replaces the confirmation prompt with a classifier that reads the transcript
and decides allow/block. Both are one-shot side calls to the model, distinct
from the main loop (they route to their own mock tracks).
"""

import json


#region goal
def evaluate_goal(condition, transcript, client, model):
    """
    评估目标条件是否满足

    调用 AI 模型作为独立评估器，根据给定条件和对话记录判断目标是否已达成。

    参数:
        condition (str): 需要评估的目标条件描述
        transcript (str): 当前的对话记录，包含所有历史消息
        client: AI 客户端对象，用于调用模型 API
        model (str): 使用的模型名称/标识符

    返回:
        dict: 包含评估结果的字典
            - met (bool): 目标是否已满足
                - True: 条件已满足
                - False: 条件未满足
            - reason (str): 未满足时的原因说明，满足时为空字符串

    示例:
        >>> result = evaluate_goal("代码通过所有测试", transcript, client, "claude-3")
        >>> if result["met"]:
        ...     print("目标已达成！")
        ... else:
        ...     print(f"未完成，原因: {result['reason']}")
    """
    # 调用模型进行目标评估
    reply = client.messages.create(
        model=model, max_tokens=256,
        system="You are a goal evaluator. Given a condition and a transcript, reply exactly 'MET' if the condition is satisfied, otherwise 'NOT_MET: <short reason>'.",
        messages=[{"role": "user", "content": f"Condition: {condition}\n\nTranscript so far:\n{transcript}"}],
    )
    # 提取模型回复中的文本内容
    text = "".join(b.text for b in reply.content if b.type == "text").strip()

    # 判断目标是否已满足
    if text.startswith("MET"):
        return {"met": True, "reason": ""}
    # 未满足时，提取并清理原因说明
    return {"met": False, "reason": text.replace("NOT_MET:", "").replace("NOT_MET", "").strip()}
#endregion


#region classifier
def classify_action(tool_name, tool_input, transcript, client, model):
    """
    分类器 - 判断工具调用是否安全可执行

    在自主模式 (--auto) 下替代人工确认，通过 AI 模型评估工具调用的安全性。
    模型会分析对话上下文和待执行的工具调用，决定是否允许执行。

    参数:
        tool_name (str): 工具名称，如 "bash", "write_file" 等
        tool_input (dict): 工具的输入参数，将被序列化为 JSON
        transcript (str): 完整的对话记录，用于上下文分析
        client: AI 客户端对象，用于调用模型 API
        model (str): 使用的模型名称/标识符

    返回:
        dict: 包含分类结果的字典
            - allow (bool): 是否允许执行
                - True: 允许执行该工具调用
                - False: 阻止执行该工具调用
            - reason (str): 阻止时的原因说明，允许时为空字符串

    安全策略:
        - 默认采取保守策略 (Err on the side of blocking)
        - 仅当确认安全时才返回 ALLOW
        - 任何可疑操作都会被阻止

    示例:
        >>> result = classify_action("bash", {"command": "ls -la"}, transcript, client, "claude-3")
        >>> if result["allow"]:
        ...     execute_tool(tool_name, tool_input)
        ... else:
        ...     print(f"操作被阻止: {result['reason']}")
    """
    # 调用模型进行安全分类
    reply = client.messages.create(
        model=model, max_tokens=256,
        system="You are a security monitor for an autonomous coding agent. Given the transcript and a tool call, reply exactly 'ALLOW' if it is safe to run unattended, otherwise 'BLOCK: <short reason>'. Err on the side of blocking.",
        messages=[{"role": "user", "content": f"Transcript:\n{transcript}\n\nTool call: {tool_name}({json.dumps(tool_input)})"}],
    )
    # 提取模型回复中的文本内容
    text = "".join(b.text for b in reply.content if b.type == "text").strip()

    # 判断是否允许执行
    if text.startswith("ALLOW"):
        return {"allow": True, "reason": ""}
    # 被阻止时，提取并清理原因说明
    return {"allow": False, "reason": text.replace("BLOCK:", "").replace("BLOCK", "").strip()}
#endregion
