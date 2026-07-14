"""
子代理模块 (Sub-agent Module)

本模块实现了一个只读的子代理机制，用于在独立的上下文中调查任务并返回摘要。
它将复杂任务分解为更小的探索任务，避免将所有中间步骤注入主对话中。
子代理在当前进程内运行自己的小循环，只有最终摘要会返回给调用者。

核心思想：分工协作（divide and conquer），子代理只具备只读权限，
只能查看文件和搜索内容，不能执行任何修改操作。
"""

from tools import tool_definitions, execute_tool

# Fork a read-only sub-agent to investigate a task in its own fresh context and
# report back a concise summary — divide and conquer without pouring all the
# intermediate steps into the main conversation. It runs its own little loop,
# in-process, and only the summary comes back.
# 派生一个只读子代理，在其自身的新上下文中调查任务并返回简洁摘要。
# 分工协作，避免将所有中间步骤注入主对话。子代理在当前进程内运行自己的小循环，
# 只有摘要会返回。

# 允许子代理使用的工具列表（仅限只读操作）
# read_file: 读取文件内容
# list_files: 列出目录下的文件
# grep_search: 在文件中搜索文本
EXPLORE_TOOLS = ["read_file", "list_files", "grep_search"]


#region subagent
def run_sub_agent(task, client, model):
    """
    运行一个只读子代理来调查指定任务并返回简洁摘要。

    该函数创建一个独立的对话循环，让子代理使用只读工具（读取文件、列出文件、
    搜索文本）来探索代码库，然后将发现汇总为一段文字摘要返回给调用者。

    参数:
        task (str): 分配给子代理的任务描述，告诉它需要调查什么
        client: Anthropic API 客户端实例，用于调用 Claude 模型
        model (str): 使用的模型名称（如 "claude-sonnet-4-20250514"）

    返回:
        str: 子代理生成的简洁文本摘要，总结调查结果

    工作流程:
        1. 将任务作为用户消息初始化对话
        2. 循环调用模型，获取回复
        3. 如果回复中包含工具调用，则执行只读工具并把结果反馈给模型
        4. 如果回复中没有工具调用（纯文本），则返回文本内容作为最终摘要
    """
    # 初始化对话历史，将任务描述作为第一条用户消息
    messages = [{"role": "user", "content": task}]
    # 从全局工具定义中筛选出子代理允许使用的只读工具
    tools = [t for t in tool_definitions if t["name"] in EXPLORE_TOOLS]

    while True:
        # 调用 Claude API 获取模型回复
        # max_tokens=4096 限制回复长度，system prompt 指定子代理角色
        reply = client.messages.create(
            model=model, max_tokens=4096,
            system="You are an explore sub-agent. Investigate read-only and report back a concise summary.",
            tools=tools, messages=messages,
        )
        # 将模型的回复添加到对话历史中
        messages.append({"role": "assistant", "content": reply.content})

        # 从回复中提取所有工具调用请求
        tool_uses = [b for b in reply.content if b.type == "tool_use"]
        # 如果没有工具调用，说明模型已完成调查，提取纯文本摘要并返回
        if not tool_uses:
            return "".join(b.text for b in reply.content if b.type == "text")
        # 处理所有工具调用请求
        results = []
        for tu in tool_uses:
            # Read-only: a sub-agent can look but not touch.
            # 只读检查：子代理只能查看，不能修改
            # 如果工具在允许列表中则执行，否则返回拒绝信息
            output = execute_tool(tu.name, tu.input) if tu.name in EXPLORE_TOOLS \
                else "Denied: the sub-agent is read-only."
            # 将工具执行结果封装为 tool_result 格式
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": output})
        # 将工具执行结果作为用户消息追加到对话历史，继续下一轮循环
        messages.append({"role": "user", "content": results})
#endregion
