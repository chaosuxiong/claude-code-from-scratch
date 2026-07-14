# When the conversation gets long, summarize the older messages into one so the
# context window doesn't overflow. Real agents count tokens; we count messages,
# which is enough to see the mechanism work.
# 【模块功能：上下文压缩模块（Context Compaction）】
# 当对话历史变得过长时，将较早的消息压缩成摘要，以防止上下文窗口溢出。
# 真正的代理系统会计算 token 数量；这里我们计算消息数量，足以演示压缩机制的工作原理。
#
# 【压缩策略】
# 将消息分为两部分："较早的消息"和"最近的消息"。
# 较早的消息会被转换为纯文本并发送给 AI 模型生成摘要，
# 最近的消息（KEEP_RECENT 条）保持原样保留。
# 这样可以保留对话的整体上下文，同时确保最近的交互不被丢失。

# 压缩阈值：当消息数量超过此值时触发压缩
COMPACT_THRESHOLD = 6
# 保留最近的消息数量：压缩时保留最近的几条消息不被压缩
KEEP_RECENT = 2


#region compact
def maybe_compact(messages, client, model):
    """
    对话压缩函数 - 当消息数量超过阈值时，将较早的消息压缩成摘要

    参数:
        messages (list): 对话消息列表，每条消息是一个字典，包含 'role' 和 'content' 字段
        client: AI客户端对象，用于调用模型API进行摘要生成
        model (str): 使用的AI模型名称，用于生成对话摘要

    返回值:
        list: 压缩后的消息列表。如果消息数量未超过阈值，返回原始消息；
              否则返回包含摘要和最近消息的新列表

    工作原理:
        1. 检查消息数量是否超过压缩阈值（COMPACT_THRESHOLD）
        2. 如果超过，将消息分为"较早的消息"和"最近的消息"两部分
        3. 调用AI模型将较早的消息压缩成一段摘要
        4. 返回包含摘要和最近消息的新列表
    """
    # 如果消息数量未超过压缩阈值，直接返回原始消息，无需压缩
    if len(messages) <= COMPACT_THRESHOLD:
        return messages

    # 分割消息：较早的消息（将被压缩）和最近的消息（保留原样）
    older = messages[: len(messages) - KEEP_RECENT]  # 较早的消息列表
    recent = messages[len(messages) - KEEP_RECENT :]  # 最近的消息列表，将保留原样

    # One aux model call: summarize the older messages (rendered as plain text so
    # we never split a tool_use / tool_result pair).
    # 辅助模型调用：将较早的消息转换为纯文本格式进行摘要
    # 这样做是为了避免拆分工具调用(tool_use)和工具结果(tool_result)的配对
    transcript = "\n".join(
        f"{m['role']}: {m['content'] if isinstance(m.get('content'), str) else '[tool call / result]'}"
        for m in older
    )
    # 调用AI模型生成对话摘要
    reply = client.messages.create(
        model=model, max_tokens=1024,
        system="Summarize the conversation so far in a few sentences, keeping key facts.",
        messages=[{"role": "user", "content": transcript}],
    )
    # 提取模型回复中的文本内容作为摘要
    summary = "".join(b.text for b in reply.content if b.type == "text")

    # 打印压缩信息：显示压缩了多少条消息
    print(f"  (compacted {len(older)} messages into a summary)")
    # 返回压缩后的消息列表：包含摘要（作为用户消息）和最近的消息
    return [{"role": "user", "content": f"[Summary of earlier conversation]\n{summary}"}, *recent]
#endregion
