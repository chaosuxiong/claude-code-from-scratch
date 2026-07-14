"""
Agent 模块 - 核心代理实现
==========================

本模块实现了 Mini Claude Code 的核心代理（Agent）类，负责：
1. 与 Anthropic Claude API 进行交互
2. 管理对话消息历史
3. 处理工具调用（tool use）循环
4. 支持多种运行模式（默认、计划、自动）
5. 集成 MCP（Model Context Protocol）外部工具
6. 实现权限检查和安全控制
7. 支持子代理（sub-agent）和自主目标追踪

这是整个系统的核心模块，协调各个组件完成用户请求。
"""

import json
import os

import anthropic

from tools import tool_definitions, execute_tool
#step >=3
from prompt import build_system_prompt
#endstep
#step >=6
from permissions import check_permission
#endstep
#step >=7
from context import maybe_compact
#endstep
#step >=8
from memory import recall_memories
#endstep
#step >=11
from subagent import run_sub_agent
#endstep
#step >=12
from mcp_client import connect_mcp
#endstep
#step >=15
from autonomy import evaluate_goal, classify_action
#endstep

# 模型名称，优先使用环境变量 MINI_MODEL，否则使用默认的 Claude Sonnet 4.5
MODEL = os.environ.get("MINI_MODEL", "claude-sonnet-4-5-20250929")

#step <=2
# A minimal, hard-coded system prompt. Chapter 3 replaces this with a real
# static-core-plus-environment prompt built in prompt.py.
# 最小化、硬编码的系统提示词。第3章会用 prompt.py 中构建的
# 静态核心+环境提示词来替代这里。
SYSTEM_PROMPT = (
    "You are Mini Claude Code, a small coding assistant that helps with software "
    "tasks. Use the tools to read and change files. Keep answers short."
)
#endstep


# The whole agent is one class holding a growing message list and a loop.
# 整个代理是一个类，持有不断增长的消息列表和一个处理循环。
class Agent:
    """
    Agent 类 - 核心代理实现

    这是系统的核心类，负责：
    - 管理与 Anthropic API 的连接
    - 维护对话消息历史
    - 实现工具调用循环（tool use loop）
    - 处理权限检查和安全控制
    - 支持多种运行模式
    """

    def __init__(self) -> None:
        """
        初始化 Agent 实例

        初始化内容包括：
        - 创建 Anthropic 客户端（支持自定义 API 密钥和基础 URL）
        - 初始化消息历史列表
        - 设置运行模式（默认为 "default"）
        - 初始化 MCP 连接（默认为 None）
        """
        kwargs = {}
        # 如果设置了 ANTHROPIC_API_KEY 环境变量，使用自定义 API 密钥
        if os.environ.get("ANTHROPIC_API_KEY"):
            kwargs["api_key"] = os.environ["ANTHROPIC_API_KEY"]
        # Optional: point at an Anthropic-compatible relay via ANTHROPIC_BASE_URL.
        # 可选：通过 ANTHROPIC_BASE_URL 指向兼容 Anthropic 的中继服务器
        if os.environ.get("ANTHROPIC_BASE_URL"):
            kwargs["base_url"] = os.environ["ANTHROPIC_BASE_URL"]
        # 创建 Anthropic 客户端实例
        self.client = anthropic.Anthropic(**kwargs)
        # 对话消息历史，存储所有用户和助手的交互
        self.messages: list = []
#step >=10
        # 运行模式："default"（默认）允许所有操作，"plan"（计划）模式下代理只读
        self.mode = "default"  # "plan" makes the agent read-only
#endstep
#step >=12
        # MCP 连接实例，用于调用外部 MCP 工具服务器
        self.mcp = None
#endstep

    # One user turn. Call the model; if it asks for tools, run them and feed the
    # results back; repeat until it answers with plain text.
    # 一个用户轮次：调用模型；如果模型请求工具，执行工具并将结果返回；
    # 重复此过程直到模型返回纯文本回答。
#region loop
    def chat(self, user_text: str) -> None:
        """
        处理一个用户对话轮次

        这是代理的核心方法，实现了完整的工具调用循环：
        1. 将用户消息添加到历史
        2. 调用模型获取回复
        3. 如果回复包含工具调用，执行工具并将结果返回给模型
        4. 重复步骤 2-3 直到模型返回纯文本（无工具调用）

        参数:
            user_text (str): 用户输入的文本消息

        返回:
            None
        """
        # 将用户消息添加到对话历史
        self.messages.append({"role": "user", "content": user_text})
#step >=12
        # 在循环开始前确保已连接 MCP 服务器，以发现外部工具
        self._ensure_mcp()  # discover external MCP tools before the loop
#endstep

        # 主循环：持续调用模型直到获得纯文本回复（无工具调用）
        while True:
#step >=7
            # Before each model call, compact the history if it has grown too long.
            # 每次调用模型前，如果历史记录过长，进行压缩处理
            self.messages = maybe_compact(self.messages, self.client, MODEL)
#endstep
#step >=3
            # 构建系统提示词（包含工具说明等）
            system = build_system_prompt()
#step <=2
            # 使用硬编码的系统提示词
            system = SYSTEM_PROMPT
#endstep
#step >=8
            # Recall memories relevant to what the user just asked, into the prompt.
            # 召回与用户当前问题相关的记忆，添加到系统提示词中
            system += recall_memories(user_text)
#endstep
#step >=12
            # Merge in any external MCP tools, prefixed so we can route their calls back.
            # 合并外部 MCP 工具，添加前缀以便路由回调
            mcp_tools = [{"name": f"mcp__demo__{t['name']}", "description": t["description"], "input_schema": t["input_schema"]}
                         for t in (self.mcp.tools if self.mcp else [])]
            tools = tool_definitions + mcp_tools
#step <=11
            # 使用本地工具定义
            tools = tool_definitions
#endstep
            # 准备 API 调用参数
            kwargs = dict(model=MODEL, max_tokens=4096, system=system, tools=tools, messages=self.messages)

#step >=5
            # Stream the reply so text shows up as it is generated, then collect
            # the finished message (same shape a non-streaming call would return).
            # 流式传输回复，文本生成时立即显示，然后收集完整消息
            #（与非流式调用返回相同格式）
            with self.client.messages.stream(**kwargs) as stream:
                # 逐块输出文本
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                # 获取最终完整消息
                reply = stream.get_final_message()
            print()
#step <=4
            # 非流式调用方式（较旧版本）
            reply = self.client.messages.create(**kwargs)
            for block in reply.content:
                if block.type == "text":
                    print(block.text, end="", flush=True)
            print()
#endstep

            # Record the assistant's full reply (text + any tool calls).
            # 记录助手的完整回复（文本 + 任何工具调用）
            self.messages.append({"role": "assistant", "content": reply.content})

            # 从回复内容中提取所有工具调用请求
            tool_uses = [b for b in reply.content if b.type == "tool_use"]
            # No tool calls means the model is done with this turn.
            # 没有工具调用意味着模型已完成本轮对话
            if not tool_uses:
                return

            # Run every requested tool; send the outputs back as one user message.
            # 执行所有请求的工具；将所有输出作为一条用户消息发送回模型
            results = []
            for tu in tool_uses:
                # 打印工具调用信息，方便调试
                print(f"  → {tu.name}({json.dumps(tu.input)})")
#step >=11
                # The `agent` tool forks a read-only sub-agent with its own context.
                # `agent` 工具会派生一个只读的子代理，拥有独立的上下文
                if tu.name == "agent":
                    # 运行子代理并获取摘要结果
                    summary = run_sub_agent(tu.input.get("task", ""), self.client, MODEL)
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "content": summary})
                    continue
#endstep
#step >=12
                # MCP tools (mcp__server__tool) go to the MCP server, not run locally.
                # MCP 工具（mcp__server__tool）发送到 MCP 服务器执行，不在本地运行
                if tu.name.startswith("mcp__"):
                    # mcp__<server>__<tool> -> <tool>; drop the first two "__"
                    # segments so it strips the same way the TypeScript side does.
                    # 从 mcp__<server>__<tool> 格式中提取工具名称
                    # 移除前两个 "__" 分段，与 TypeScript 端保持一致
                    tool_name = "__".join(tu.name.split("__")[2:])
                    # 调用 MCP 工具，如果没有连接则返回错误信息
                    output = self.mcp.call_tool(tool_name, tu.input) if self.mcp else "Denied: no MCP server connected."
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "content": output})
                    continue
#endstep
#step >=15
                # Auto mode: a classifier decides block/allow instead of asking a human.
                # 自动模式：使用分类器决定阻止/允许，而不是询问用户
                if self.mode == "auto" and tu.name in ("write_file", "edit_file", "run_shell"):
                    # 使用分类器评估操作是否应该被允许
                    verdict = classify_action(tu.name, tu.input, self._transcript_text(), self.client, MODEL)
                    if not verdict["allow"]:
                        # 操作被自动模式监控器阻止
                        results.append({"type": "tool_result", "tool_use_id": tu.id, "content": f"Blocked by auto-mode monitor: {verdict['reason']}"})
                        continue
#endstep
#step >=10
                # Plan mode is read-only: writes and shell are denied on top of the gate.
                # 计划模式是只读的：在权限检查之上，写入和 shell 操作会被拒绝
                blocked = check_permission(tu.name, tu.input) == "deny" or (
                    self.mode == "plan" and tu.name in ("write_file", "edit_file", "run_shell"))
                # 根据是否被阻止决定输出内容
                output = f"Denied: {tu.name} was blocked ({self.mode} mode)." if blocked \
                    else execute_tool(tu.name, tu.input)
#step >=6
                # Check permission before running the tool; a denied call never runs.
                # 运行工具前检查权限；被拒绝的调用永远不会执行
                if check_permission(tu.name, tu.input) == "deny":
                    output = f"Denied: {tu.name} was blocked by the permission system."
                else:
                    output = execute_tool(tu.name, tu.input)
#step <=5
                # 直接执行工具（无权限检查，较旧版本）
                output = execute_tool(tu.name, tu.input)
#endstep
                # 将工具执行结果添加到结果列表
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": output})
            # 将所有工具结果作为一条用户消息发送回模型
            self.messages.append({"role": "user", "content": results})
#endregion
#step >=4
    # Session support: expose the history so the CLI can save it and restore it.
    # 会话支持：暴露历史记录，以便 CLI 可以保存和恢复会话状态
    def history(self):
        """
        获取对话历史记录

        返回当前对话的所有消息历史，用于保存会话状态。

        返回:
            list: 包含所有消息的列表
        """
        return self.messages

    def load_history(self, messages) -> None:
        """
        加载历史记录

        从外部加载之前保存的对话历史，用于恢复会话状态。

        参数:
            messages (list): 要加载的消息列表

        返回:
            None
        """
        self.messages = messages

    def clear_history(self) -> None:
        """
        清空历史记录

        清除所有对话历史，重置会话状态。

        返回:
            None
        """
        self.messages = []
#endstep
#step >=10
    def set_mode(self, m: str) -> None:
        """
        设置代理运行模式

        支持的模式：
        - "default": 默认模式，允许所有操作
        - "plan": 计划模式，只读操作，禁止写入和 shell 命令
        - "auto": 自动模式，使用分类器自动决定操作权限

        参数:
            m (str): 模式名称，必须是 "default"、"plan" 或 "auto"

        返回:
            None
        """
        self.mode = m
#endstep
#step >=12
    # Connect to the MCP server named in MINI_MCP_SERVER once, on first use.
    # 首次使用时连接到 MINI_MCP_SERVER 环境变量指定的 MCP 服务器
    def _ensure_mcp(self):
        """
        确保已连接 MCP 服务器

        如果尚未连接且设置了 MINI_MCP_SERVER 环境变量，
        则建立到指定 MCP 服务器的连接。

        这是一个内部方法，只在首次使用时调用一次。

        返回:
            None
        """
        # 只在未连接且环境变量存在时才连接
        if self.mcp is None and os.environ.get("MINI_MCP_SERVER"):
            self.mcp = connect_mcp("node", [os.environ["MINI_MCP_SERVER"]])
#endstep
#step >=15
    def _transcript_text(self):
        """
        获取对话记录的文本形式

        将对话历史转换为可读的文本格式，用于自主性评估。
        每行格式为 "role: content"，工具调用显示为 "[tool call / result]"。

        返回:
            str: 对话记录的文本表示
        """
        return "\n".join(
            f"{m['role']}: {m['content'] if isinstance(m.get('content'), str) else '[tool call / result]'}"
            for m in self.messages)

    # Autonomy: keep working until an independent evaluator judges the condition met.
    # 自主性：持续工作直到独立评估器判断条件已满足
    def pursue_goal(self, condition, prompt):
        """
        追踪并实现目标

        实现自主目标追踪功能：
        1. 执行初始提示
        2. 评估目标是否达成
        3. 如果未达成，继续工作并重新评估
        4. 最多尝试 5 次迭代

        参数:
            condition (str): 目标条件的描述，用于评估是否达成
            prompt (str): 初始提示，告诉代理需要做什么

        返回:
            None
        """
        # 执行初始提示
        self.chat(prompt)
        # 最多尝试 5 次迭代
        for _ in range(5):
            # 使用评估器检查目标是否已达成
            verdict = evaluate_goal(condition, self._transcript_text(), self.client, MODEL)
            if verdict["met"]:
                # 目标已达成，输出成功信息
                print(f"✓ goal met: {condition}")
                return
            # 目标未达成，输出原因并继续工作
            print(f"  (goal not met — {verdict['reason']}; continuing)")
            # 告诉代理目标未达成，要求继续工作
            self.chat(f'The goal "{condition}" is not met yet: {verdict["reason"]}. Keep working toward it.')
        # 超过最大迭代次数，放弃
        print(f"  (gave up after 5 iterations without meeting: {condition})")
#endstep
