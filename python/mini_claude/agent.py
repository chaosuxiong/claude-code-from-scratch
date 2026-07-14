"""Agent core loop — dual backend (Anthropic + OpenAI compatible), streaming,
4-layer compression, plan mode, sub-agents, budget control.
Agent architecture inspired by Claude Code's published design."""

# ============================================================================
# 本模块是 mini-claude 的核心 Agent 引擎，实现了完整的智能体循环。
#
# 主要功能：
#   1. 双后端支持 — 同时兼容 Anthropic API 和 OpenAI 兼容 API
#   2. 流式响应 — 支持流式输出和流式工具执行（tool_use 在生成过程中提前执行）
#   3. 四层上下文压缩 — budget/snip/microcompact/autocompact 四级压缩策略
#   4. 计划模式（Plan Mode）— 只读探索 + 生成计划文件 + 用户审批工作流
#   5. 子 Agent（Sub-agent）— 支持 fork 子代理执行独立任务
#   6. 预算控制 — 支持按费用（USD）和轮次（turns）限制
#   7. 自动模式（Auto Mode）— 基于 LLM 分类器的自动权限审批
#   8. Goal 追踪 — 基于提示的 Stop-hook 条件，跨轮次自动判断目标是否达成
#   9. Loop 循环 — 定时/自适应两种循环模式，支持云端持久化
#  10. 记忆系统 — 语义预取（prefetch）+ 会话注入
#  11. MCP 集成 — 动态加载外部工具服务器
#
# 类结构：
#   Agent — 主类，封装所有状态和逻辑
#
# 受 Claude Code 公开设计文档启发，代码中标注了 CC 对应的实现路径。
# ============================================================================

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Awaitable

import anthropic
import openai

from .tools import (
    tool_definitions,
    execute_tool,
    check_permission,
    CONCURRENCY_SAFE_TOOLS,
    get_active_tool_definitions,
    ToolDef,
    PermissionMode,
    _truncate_result,
)
from .memory import (
    start_memory_prefetch,
    format_memories_for_injection,
    MemoryPrefetch,
)
from .autonomy import (
    goal_directive,
    GOAL_EVALUATOR_SYSTEM,
    GOAL_TRANSCRIPT_FRAMING,
    goal_judge_user_message,
    parse_goal_verdict,
    GOAL_MAX_ITERATIONS,
    parse_loop_input,
    is_daily_wording,
    OFFER_CLOUD_THRESHOLD_SECONDS,
    SCHEDULE_WAKEUP_TOOL,
    clamp_wakeup_delay,
    dynamic_loop_directive,
    LOOP_MAX_ITERATIONS,
    load_auto_mode_rules,
    build_classifier_system,
    AUTO_MODE_FAST_PATH_TOOLS,
    DENIAL_LIMITS,
    build_classifier_transcript,
    parse_block_verdict,
    classifier_user_message,
)
from .ui import (
    print_assistant_text,
    print_tool_call,
    print_tool_result,
    print_error,
    print_confirmation,
    print_divider,
    print_cost,
    print_retry,
    print_info,
    print_sub_agent_start,
    print_sub_agent_end,
    start_spinner,
    stop_spinner,
)
from .session import save_session
from .prompt import build_system_prompt, build_static_system_prompt, build_dynamic_system_context, build_user_context_reminder, load_claude_md
from .subagent import get_sub_agent_config
from .mcp_client import McpManager

# ─── Retry with exponential backoff ──────────────────────────
# 指数退避重试机制：当 API 调用遇到瞬态错误（429/503/529/网络超时等）时，
# 自动重试，使用指数退避算法避免雪崩效应。


def _is_retryable(error: Exception) -> bool:
    """判断异常是否为可重试的瞬态错误。

    检查 HTTP 状态码（429=限流、503=服务不可用、529=过载）以及
    错误消息中的关键字（overloaded、ECONNRESET、ETIMEDOUT）。

    Args:
        error: 捕获的异常对象

    Returns:
        True 表示该错误可重试，False 表示不应重试
    """
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status in (429, 503, 529):
        return True
    msg = str(error)
    if "overloaded" in msg or "ECONNRESET" in msg or "ETIMEDOUT" in msg:
        return True
    return False


async def _with_retry(fn, max_retries: int = 3):
    """带指数退避的异步重试包装器。

    对异步函数 fn 进行重试，最多重试 max_retries 次。
    每次重试的延迟按指数增长（1s, 2s, 4s...），最大 30s，
    并加入少量随机抖动以避免雷群效应。

    Args:
        fn: 要执行的异步函数（无参数的 callable）
        max_retries: 最大重试次数，默认 3

    Returns:
        fn() 的返回值

    Raises:
        最终重试耗尽后抛出原始异常
    """
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as error:
            if attempt >= max_retries or not _is_retryable(error):
                raise
            # 计算退避延迟：基础延迟 * 2^attempt，加上随机抖动
            delay = min(1000 * (2 ** attempt), 30000) / 1000 + (hash(str(time.time())) % 1000) / 1000
            status = getattr(error, "status_code", None) or getattr(error, "status", None)
            reason = f"HTTP {status}" if status else (getattr(error, "code", None) or "network error")
            print_retry(attempt + 1, max_retries, reason)
            await asyncio.sleep(delay)


# ─── Model context windows ──────────────────────────────────
# 各模型的上下文窗口大小（token 数），用于判断何时触发压缩。

MODEL_CONTEXT = {
    "claude-opus-4-6": 200000,
    "claude-sonnet-4-6": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-haiku-4-5-20251001": 200000,
    "claude-opus-4-20250514": 200000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
}


def _get_context_window(model: str) -> int:
    """获取指定模型的上下文窗口大小，默认返回 200000。

    Args:
        model: 模型名称字符串

    Returns:
        上下文窗口的 token 数量
    """
    return MODEL_CONTEXT.get(model, 200000)


# ─── Thinking support detection ─────────────────────────────
# 检测模型是否支持扩展思考（extended thinking）模式。
# Claude 3.x 不支持，Claude 4.x 支持；其中 opus-4-6/sonnet-4-6 支持自适应思考。


def _model_supports_thinking(model: str) -> bool:
    """判断模型是否支持扩展思考模式。

    Claude 3.x 系列（包括 3.5、3.7）不支持，Claude 4.x 系列支持。

    Args:
        model: 模型名称字符串

    Returns:
        True 表示模型支持扩展思考
    """
    m = model.lower()
    if "claude-3-" in m or "3-5-" in m or "3-7-" in m:
        return False
    if "claude" in m and any(x in m for x in ("opus", "sonnet", "haiku")):
        return True
    return False


def _model_supports_adaptive_thinking(model: str) -> bool:
    """判断模型是否支持自适应思考（adaptive thinking）。

    自适应思考允许模型根据问题复杂度自动调整思考深度。
    目前仅 opus-4-6 和 sonnet-4-6 支持。

    Args:
        model: 模型名称字符串

    Returns:
        True 表示模型支持自适应思考
    """
    m = model.lower()
    return "opus-4-6" in m or "sonnet-4-6" in m


def _get_max_output_tokens(model: str) -> int:
    """获取模型的最大输出 token 数。

    不同模型有不同的输出限制，opus-4-6 最高支持 64000 token 输出。

    Args:
        model: 模型名称字符串

    Returns:
        最大输出 token 数
    """
    m = model.lower()
    if "opus-4-6" in m:
        return 64000
    if "sonnet-4-6" in m:
        return 32000
    if any(x in m for x in ("opus-4", "sonnet-4", "haiku-4")):
        return 32000
    return 16384


# ─── Convert tools to OpenAI format ─────────────────────────
# 将内部工具定义转换为 OpenAI API 的 function calling 格式。


def _to_openai_tools(tools: list[ToolDef]) -> list[dict]:
    """将内部工具定义列表转换为 OpenAI 兼容格式。

    Args:
        tools: 内部工具定义列表（ToolDef 类型）

    Returns:
        OpenAI 格式的工具定义列表，每个元素包含 type、function.name、
        function.description、function.parameters
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


# ─── Multi-tier compression constants ────────────────────────
# 多层压缩策略的常量配置。

# 可以被 snip（裁剪）的工具结果类型 — 这些工具的输出通常很大，
# 但可以通过重新读取来恢复。
SNIPPABLE_TOOLS = {"read_file", "grep_search", "list_files", "run_shell"}
# 裁剪后的占位符文本，提示模型需要时可重新读取
SNIP_PLACEHOLDER = "[Content snipped - re-read if needed]"
# 上下文利用率超过此阈值时开始裁剪旧结果（60%）
SNIP_THRESHOLD = 0.60
# 当上下文利用率超过此值时，即使缓存仍然有效也会执行裁剪（75%）。
# 这是为了避免上下文溢出，优先级高于缓存保持。
SNIP_HOT_OVERRIDE = 0.75
# 微压缩的空闲阈值：距离上次 API 调用超过 5 分钟才触发微压缩
MICROCOMPACT_IDLE_S = 5 * 60  # 5 minutes
# 保留最近的工具结果数量，不被裁剪/清除
KEEP_RECENT_RESULTS = 3


# ─── Agent ───────────────────────────────────────────────────


class Agent:
    """核心 Agent 类，封装了完整的智能体循环逻辑。

    支持 Anthropic 和 OpenAI 两种后端 API，实现了：
    - 流式响应与流式工具执行
    - 多层上下文压缩（budget/snip/microcompact/autocompact）
    - 计划模式（Plan Mode）
    - 子 Agent 嵌套
    - 预算控制（费用/轮次）
    - 自动模式（LLM 分类器权限审批）
    - Goal 追踪与 Loop 循环
    - 记忆预取与 MCP 工具集成
    """

    def __init__(
        self,
        *,
        permission_mode: str = "default",
        model: str = "claude-opus-4-6",
        api_base: str | None = None,
        anthropic_base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool = False,
        max_cost_usd: float | None = None,
        max_turns: int | None = None,
        confirm_fn: Callable[[str], Awaitable[bool]] | None = None,
        custom_system_prompt: str | None = None,
        custom_tools: list[ToolDef] | None = None,
        is_sub_agent: bool = False,
    ):
        """初始化 Agent 实例。

        Args:
            permission_mode: 权限模式（"default"=默认、"plan"=计划模式、
                "auto"=自动模式、"bypassPermissions"=跳过权限）
            model: 模型名称，默认 claude-opus-4-6
            api_base: OpenAI 兼容 API 地址（设置后使用 OpenAI 后端）
            anthropic_base_url: Anthropic API 的自定义 base URL
            api_key: API 密钥（可选，默认从环境变量读取）
            thinking: 是否启用扩展思考模式
            max_cost_usd: 最大费用限制（美元），None 表示无限制
            max_turns: 最大轮次限制，None 表示无限制
            confirm_fn: 危险操作的异步确认回调函数
            custom_system_prompt: 自定义系统提示（覆盖默认的静态+动态提示）
            custom_tools: 自定义工具列表（覆盖默认工具）
            is_sub_agent: 是否为子 Agent（子 Agent 不显示 UI 元素）
        """
        self.permission_mode = permission_mode
        self.thinking = thinking
        self.model = model
        self.use_openai = bool(api_base)  # 有 api_base 时使用 OpenAI 后端
        self.is_sub_agent = is_sub_agent
        self.tools = custom_tools or tool_definitions
        self.max_cost_usd = max_cost_usd
        self.max_turns = max_turns
        self.confirm_fn = confirm_fn
        # 有效上下文窗口 = 模型窗口 - 20000（预留系统提示和安全余量）
        self.effective_window = _get_context_window(model) - 20000
        self.session_id = uuid.uuid4().hex[:8]  # 8 位十六进制会话 ID
        self.session_start_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # ---- Token 统计 ----
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0       # Prompt-cache 命中（按 0.1x 计费）
        self.total_cache_creation_tokens = 0   # Prompt-cache 写入（按 1.25x 计费）
        self.last_input_token_count = 0  # 上一轮的输入 token 数（用于压缩判断）
        self.current_turns = 0  # 当前已执行的工具调用轮次
        self.last_api_call_time = 0.0  # 上次 API 调用时间戳

        # ---- /goal 追踪状态 ----
        # 基于提示的 Stop-hook 条件，跨轮次自动判断目标是否达成
        self.active_goal: dict | None = None  # 当前活跃的 goal 配置
        self.goal_stop = False  # 中断信号，用于跳出 goal 追踪循环

        # ---- /loop 动态模式状态 ----
        # 模型在 tick 中调用 schedule_wakeup 时设置，
        # loop 驱动器在轮次收敛后读取并清除。
        self.pending_wakeup: dict | None = None
        self.loop_stop = False  # 中断信号，用于跳出运行中的 loop
        # schedule_wakeup 工具仅在动态 loop 期间启用，
        # 防止外部调用或与同名工具冲突。
        self.schedule_wakeup_enabled = False

        # ---- Auto Mode 拒绝追踪 ----
        # 当连续拒绝次数达到阈值时，自动回退到人工确认
        self.auto_consecutive_denials = 0
        self.auto_total_denials = 0

        # ---- 中止支持 ----
        self._aborted = False
        self._current_task: asyncio.Task | None = None  # 当前正在执行的异步任务

        # ---- 权限白名单 ----
        # 用户已确认的路径，后续相同路径不再询问
        self._confirmed_paths: set[str] = set()

        # ---- 计划模式状态 ----
        self._pre_plan_mode: str | None = None  # 进入计划模式前的权限模式
        self._plan_file_path: str | None = None  # 计划文件路径
        self._plan_approval_fn: Callable[[str], Awaitable[dict]] | None = None  # 计划审批回调
        self._context_cleared: bool = False  # 计划审批后是否清除了上下文

        # ---- 思考模式 ----
        self._thinking_mode = self._resolve_thinking_mode()

        # ---- 输出缓冲区（子 Agent 捕获输出时使用） ----
        self._output_buffer: list[str] | None = None

        # ---- 编辑前先读取（Read-before-edit） ----
        # 跟踪文件读取时间戳，确保编辑前已读取过文件
        self._read_file_state: dict[str, float] = {}

        # ---- MCP 集成 ----
        self._mcp_manager = McpManager()  # MCP 工具服务器管理器
        self._mcp_initialized = False  # 是否已初始化 MCP 连接

        # ---- 记忆预取状态 ----
        # 每次用户输入时进行语义预取，handle 保留在实例上，
        # 使得在本轮最后一次 API 调用后才完成的预取结果能顺延到下一轮注入（issue #7）。
        self._already_surfaced_memories: set[str] = set()  # 已展示过的记忆路径
        self._session_memory_bytes = 0  # 会话中已注入的记忆字节数
        self._memory_prefetch: MemoryPrefetch | None = None  # 当前预取任务

        # ---- 消息历史 ----
        # Anthropic 和 OpenAI 使用不同的消息格式，分别维护
        self._anthropic_messages: list[dict] = []
        self._openai_messages: list[dict] = []

        # ---- 构建系统提示（静态/动态分离以支持前缀缓存） ----
        # 自定义系统提示覆盖静态和动态两部分（全部视为静态）。
        # 否则静态核心可缓存，环境/git/技能信息构成动态尾部，
        # CLAUDE.md + 日期作为 <system-reminder> 放入第一条用户消息
        # （Claude Code 的 prependUserContext 机制）。
        # 将项目特定内容排除在系统提示外以最大化缓存共享。
        self._user_context_reminder = ""
        if custom_system_prompt:
            # 使用自定义提示时，全部视为静态
            self._static_system_prompt = custom_system_prompt
            self._dynamic_system_context = ""
        else:
            self._static_system_prompt = build_static_system_prompt()
            self._dynamic_system_context = build_dynamic_system_context()
            self._user_context_reminder = build_user_context_reminder()
        self._base_system_prompt = (
            self._static_system_prompt + "\n\n" + self._dynamic_system_context
            if self._dynamic_system_context else self._static_system_prompt
        )
        # 计划模式下在系统提示末尾追加计划模式指令
        if self.permission_mode == "plan":
            self._plan_file_path = self._generate_plan_file_path()
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
        else:
            self._system_prompt = self._base_system_prompt

        # ---- SDK 重试配置 ----
        # 可选：限制 SDK 自身的重试层（默认 2 次）。
        # 设置 MINI_CLAUDE_SDK_MAX_RETRIES=0 可在测试中隔离 _with_retry。
        _sdk_retries: dict[str, Any] = {}
        _rv = os.environ.get("MINI_CLAUDE_SDK_MAX_RETRIES")
        if _rv is not None and _rv != "":
            try:
                _sdk_retries["max_retries"] = int(_rv)
            except ValueError:
                pass
        # ---- 初始化 API 客户端 ----
        if self.use_openai:
            # OpenAI 后端：使用 AsyncOpenAI 客户端，系统提示作为第一条消息
            self._openai_client = openai.AsyncOpenAI(base_url=api_base, api_key=api_key, **_sdk_retries)
            self._anthropic_client = None
            self._openai_messages.append({"role": "system", "content": self._system_prompt})
        else:
            # Anthropic 后端：使用 AsyncAnthropic 客户端
            kwargs: dict[str, Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if anthropic_base_url:
                kwargs["base_url"] = anthropic_base_url
            kwargs.update(_sdk_retries)
            self._anthropic_client = anthropic.AsyncAnthropic(**kwargs)
            self._openai_client = None

    # ─── Prefix caching (Anthropic) ─────────────────────────────
    # Anthropic API 的前缀缓存机制：在静态系统提示上设置 cache_control 断点，
    # 使服务器端缓存前缀部分，仅处理新增的动态内容。这是 Claude Code 的
    # scope-omitted 路径（参见 how-claude-code-works ch3.6）。

    def _build_anthropic_system(self) -> list[dict]:
        """构建 Anthropic API 的 system 字段，包含缓存控制断点。

        将静态系统提示标记为可缓存（cache_control: ephemeral），
        动态内容（环境信息、计划模式指令）放在断点之后。
        工具 schema 在 system 之前渲染，因此也被缓存覆盖。

        Returns:
            文本块列表，第一个块带有 cache_control 断点
        """
        plan_suffix = self._build_plan_mode_prompt() if self.permission_mode == "plan" else ""
        dynamic_text = (self._dynamic_system_context + plan_suffix).strip()
        blocks: list[dict] = [
            {"type": "text", "text": self._static_system_prompt, "cache_control": {"type": "ephemeral"}}
        ]
        if dynamic_text:
            blocks.append({"type": "text", "text": dynamic_text})
        return blocks

    def _with_cache_breakpoints(self, messages: list[dict]) -> list[dict]:
        """Return a COPY of the message list with a cache_control breakpoint on
        the last message's final content block, so every prior turn stays in the
        cached prefix and only the newest messages are processed. Pure: the
        persistent history is never mutated with this API metadata (Claude Code
        clones request params at the render layer for the same reason, keeping
        session save / compact / restore clean). Faithful to CC's
        assistantMessageToMessageParam, we look only at the very LAST block and
        skip it when it is a thinking block (unstable content → hurts cache
        hits). Only 1 message breakpoint + 1 system breakpoint per request."""
        if not messages:
            return messages
        out = list(messages)
        last = out[-1]
        raw = last.get("content")
        content = [{"type": "text", "text": raw}] if isinstance(raw, str) else list(raw)
        tail = content[-1] if content else None
        if isinstance(tail, dict) and tail.get("type") not in ("thinking", "redacted_thinking"):
            content[-1] = {**tail, "cache_control": {"type": "ephemeral"}}
            out[-1] = {**last, "content": content}
        return out

    def _resolve_thinking_mode(self) -> str:
        """解析当前的思考模式。

        根据用户配置和模型能力，返回 "disabled"、"enabled" 或 "adaptive"。
        adaptive 模式允许模型根据问题复杂度自动调整思考深度。

        Returns:
            思考模式字符串：disabled/enabled/adaptive
        """
        if not self.thinking:
            return "disabled"
        if not _model_supports_thinking(self.model):
            return "disabled"
        if _model_supports_adaptive_thinking(self.model):
            return "adaptive"
        return "enabled"

    @property
    def is_processing(self) -> bool:
        """判断 Agent 当前是否正在处理请求。

        Returns:
            True 表示有正在执行的异步任务
        """
        return self._current_task is not None and not self._current_task.done()

    def _build_side_query(self):
        """构建一个用于记忆召回和 Auto Mode 分类器的 sideQuery 可调用对象。

        返回一个异步函数，接受 system 和 user_message 参数，
        使用 temperature=0 确保确定性输出（相同输入始终产生相同判定）。
        兼容 Anthropic 和 OpenAI 两种后端。

        Returns:
            异步函数 (system, user_message) -> str，或 None（无可用客户端时）
        """
        if self._anthropic_client:
            client = self._anthropic_client
            model = self.model
            async def _sq(system: str, user_message: str) -> str:
                resp = await client.messages.create(
                    model=model, max_tokens=256, system=system, temperature=0,
                    messages=[{"role": "user", "content": user_message}],
                )
                return "".join(b.text for b in resp.content if b.type == "text")
            return _sq
        if self._openai_client:
            client = self._openai_client
            model = self.model
            async def _sq_oai(system: str, user_message: str) -> str:
                resp = await client.chat.completions.create(
                    model=model, max_tokens=256, temperature=0,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message},
                    ],
                )
                return resp.choices[0].message.content or "" if resp.choices else ""
            return _sq_oai
        return None

    def abort(self) -> None:
        """中止当前正在执行的任务。

        设置中止标志并取消当前异步任务，Agent 在下一个检查点退出循环。
        """
        self._aborted = True
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

    def set_confirm_fn(self, fn: Callable[[str], Awaitable[bool]]) -> None:
        """设置危险操作的异步确认回调函数。

        Args:
            fn: 接受命令字符串，返回 True（允许）或 False（拒绝）的异步函数
        """
        self.confirm_fn = fn

    def set_plan_approval_fn(self, fn: Callable[[str], Awaitable[dict]]) -> None:
        """设置计划审批的异步回调函数。

        Args:
            fn: 接受计划内容，返回包含 choice 和 feedback 的字典的异步函数
        """
        self._plan_approval_fn = fn

    # ─── Plan mode toggle ────────────────────────────────────
    # 计划模式切换：进入时保存当前权限模式，退出时恢复。

    def toggle_plan_mode(self) -> str:
        """切换计划模式的开关。

        在计划模式和之前的权限模式之间切换。
        进入计划模式时保存当前模式并生成计划文件路径，
        退出时恢复之前的模式。

        Returns:
            切换后的权限模式字符串
        """
        if self.permission_mode == "plan":
            self.permission_mode = self._pre_plan_mode or "default"
            self._pre_plan_mode = None
            self._plan_file_path = None
            self._system_prompt = self._base_system_prompt
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] = self._system_prompt
            print_info(f"Exited plan mode → {self.permission_mode} mode")
            return self.permission_mode
        else:
            self._pre_plan_mode = self.permission_mode
            self.permission_mode = "plan"
            self._plan_file_path = self._generate_plan_file_path()
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] = self._system_prompt
            print_info(f"Entered plan mode. Plan file: {self._plan_file_path}")
            return "plan"

    def get_token_usage(self) -> dict:
        """获取当前会话的 token 使用统计。

        Returns:
            包含 input 和 output 键的字典，值为 token 数量
        """
        return {"input": self.total_input_tokens, "output": self.total_output_tokens}

    # ─── Main entry point ────────────────────────────────────
    # chat() 是用户消息的主要入口，根据后端类型分发到
    # _chat_anthropic() 或 _chat_openai()。

    async def chat(self, user_message: str) -> None:
        """处理用户消息的主入口点。

        首次调用时懒初始化 MCP 连接，然后根据后端类型
        分发到 Anthropic 或 OpenAI 的聊天实现。
        处理完成后自动保存会话（仅主 Agent）。

        Args:
            user_message: 用户输入的消息文本
        """
        # 懒初始化 MCP 服务器连接（仅主 Agent）
        if not self._mcp_initialized and not self.is_sub_agent:
            self._mcp_initialized = True
            try:
                await self._mcp_manager.load_and_connect()
                mcp_defs = self._mcp_manager.get_tool_definitions()
                if mcp_defs:
                    self.tools = self.tools + mcp_defs
            except Exception as e:
                print(f"[mcp] Init failed: {e}", flush=True)

        self._aborted = False
        coro = self._chat_openai(user_message) if self.use_openai else self._chat_anthropic(user_message)
        self._current_task = asyncio.current_task()
        try:
            await coro
        except asyncio.CancelledError:
            self._aborted = True
        finally:
            self._current_task = None
        if not self.is_sub_agent:
            print_divider()
            self._auto_save()

    # ─── Sub-agent entry point ────────────────────────────────
    # 子 Agent 入口：启用输出缓冲，执行一次 chat，收集输出文本和 token 消耗。

    async def run_once(self, prompt: str) -> dict:
        """子 Agent 的单次执行入口。

        启用输出缓冲区捕获所有输出文本，执行一次完整的 chat 循环，
        然后返回输出文本和本次执行消耗的 token 数。

        Args:
            prompt: 子 Agent 的执行提示

        Returns:
            包含 text（输出文本）和 tokens（input/output token 数）的字典
        """
        self._output_buffer = []
        prev_in = self.total_input_tokens
        prev_out = self.total_output_tokens
        await self.chat(prompt)
        text = "".join(self._output_buffer)
        self._output_buffer = None
        return {
            "text": text,
            "tokens": {
                "input": self.total_input_tokens - prev_in,
                "output": self.total_output_tokens - prev_out,
            },
        }

    # ─── Output helper ────────────────────────────────────────
    # 文本输出辅助：子 Agent 模式下写入缓冲区，主模式下直接打印。

    def _emit_text(self, text: str) -> None:
        """输出文本到缓冲区或控制台。

        子 Agent 模式下将文本追加到输出缓冲区（供父 Agent 收集），
        主 Agent 模式下直接打印到控制台。

        Args:
            text: 要输出的文本
        """
        if self._output_buffer is not None:
            self._output_buffer.append(text)
        else:
            print_assistant_text(text)

    # ─── REPL commands ────────────────────────────────────────
    # REPL 交互命令：清空历史、显示费用、压缩对话等。

    def clear_history(self) -> None:
        """清空对话历史和所有 token 统计。

        OpenAI 后端会重新插入系统提示消息。
        """
        self._anthropic_messages = []
        self._openai_messages = []
        if self.use_openai:
            self._openai_messages.append({"role": "system", "content": self._system_prompt})
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_creation_tokens = 0
        self.last_input_token_count = 0
        print_info("Conversation cleared.")

    def show_cost(self) -> None:
        """显示当前会话的费用和 token 使用详情。

        包括输入/输出 token 数、缓存命中率、预估费用（USD）、
        预算限制和轮次限制信息。
        """
        total = self._get_current_cost_usd()
        budget_info = f" / ${self.max_cost_usd} budget" if self.max_cost_usd else ""
        turn_info = f" | Turns: {self.current_turns}/{self.max_turns}" if self.max_turns else ""
        cached = self.total_cache_read_tokens
        billed_input = self.total_input_tokens + self.total_cache_creation_tokens + cached
        hit_rate = round((cached / billed_input) * 100) if billed_input > 0 else 0
        cache_info = (
            f"\n  Cache: {cached} read / {self.total_cache_creation_tokens} write ({hit_rate}% of input from cache)"
            if (cached or self.total_cache_creation_tokens) else ""
        )
        print_info(f"Tokens: {self.total_input_tokens} in / {self.total_output_tokens} out{cache_info}\n  Estimated cost: ${total:.4f}{budget_info}{turn_info}")

    def _get_current_cost_usd(self) -> float:
        """计算当前会话的预估费用（美元）。

        使用 Claude Code 的固定费率计算：
        - 基础输入：$3/Mtok
        - 缓存读取：$0.3/Mtok（0.1x）
        - 缓存写入：$3.75/Mtok（1.25x）
        - 输出：$15/Mtok

        Returns:
            预估费用（美元）
        """
        M = 1_000_000
        return (
            (self.total_input_tokens / M) * 3
            + (self.total_cache_read_tokens / M) * 0.3
            + (self.total_cache_creation_tokens / M) * 3.75
            + (self.total_output_tokens / M) * 15
        )

    def _check_budget(self) -> dict:
        """检查是否超出预算限制（费用或轮次）。

        Returns:
            包含 exceeded（bool）和 reason（超限时的原因）的字典
        """
        if self.max_cost_usd is not None and self._get_current_cost_usd() >= self.max_cost_usd:
            return {"exceeded": True, "reason": f"Cost limit reached (${self._get_current_cost_usd():.4f} >= ${self.max_cost_usd})"}
        if self.max_turns is not None and self.current_turns >= self.max_turns:
            return {"exceeded": True, "reason": f"Turn limit reached ({self.current_turns} >= {self.max_turns})"}
        return {"exceeded": False}

    async def compact(self) -> None:
        """手动触发对话压缩（公开接口，供 REPL 调用）。"""
        await self._compact_conversation()

    # ─── /goal pursuit ────────────────────────────────────────
    # 基于提示的 Stop-hook 机制：每轮结束后由独立的评估模型判断条件是否满足；
    # 未满足则将原因反馈到下一轮，满足/不可能则停止。
    # 评估器的完整提示参见 autonomy.py。

    def set_goal(self, condition: str) -> str:
        """设置活跃的 goal 并返回首轮执行指令。

        Args:
            condition: 目标条件文本（如 "代码已通过所有测试"）

        Returns:
            首轮执行的指令文本
        """
        self.active_goal = {"condition": condition, "iterations": 0, "started_at": time.time(), "last_reason": None}
        print_info(f'◎ /goal active — Stop hook condition: "{condition}"')
        return goal_directive(condition)

    def show_goal(self) -> None:
        """显示当前 goal 的状态信息。

        无参数时调用，显示条件、迭代次数、已用时间和上次评估原因。
        """
        if not self.active_goal:
            print_info("No active goal. Set one with /goal <condition>.")
            return
        secs = time.time() - self.active_goal["started_at"]
        last = f"\n  last reason: {self.active_goal['last_reason']}" if self.active_goal["last_reason"] else ""
        print_info(
            f"◎ /goal active\n  condition: {self.active_goal['condition']}\n"
            f"  iterations: {self.active_goal['iterations']}\n  elapsed: {secs:.1f}s{last}"
        )

    async def pursue_goal(self, directive: str) -> None:
        """执行 goal 追踪循环。

        流程：执行指令轮 -> 评估条件 -> 未满足则反馈原因到下一轮 ->
        重复直到：条件满足、判断为不可能、预算/迭代上限、或被中断。

        Args:
            directive: 首轮执行的指令文本
        """
        if not self.active_goal:
            return
        self.goal_stop = False
        try:
            await self.chat(directive)
            # Evaluate the turn that just finished *before* any cap or next-turn
            # decision, so the final turn's output is never left unjudged.
            while self.active_goal and not self.goal_stop and not self._aborted:
                verdict = await self._evaluate_goal(self.active_goal["condition"])
                if verdict["ok"]:
                    turns = self.active_goal["iterations"] + 1
                    secs = time.time() - self.active_goal["started_at"]
                    plural = "" if turns == 1 else "s"
                    print_info(f"✓ Goal achieved ({turns} turn{plural}, {secs:.1f}s): {verdict['reason']}")
                    break
                if verdict.get("impossible"):
                    print_info(f"Hooks: Prompt hook condition judged impossible: {verdict['reason']}")
                    break

                # Not met: record and decide whether another turn is allowed.
                self.active_goal["iterations"] += 1
                self.active_goal["last_reason"] = verdict["reason"]
                print_info(f"Hooks: Prompt hook condition was not met: {verdict['reason']}")

                budget = self._check_budget()
                if budget["exceeded"]:
                    print_info(f"Goal stopped: {budget['reason']}")
                    break
                # Hard ceiling regardless of --max-turns: --max-turns only counts
                # tool-executing turns (_check_budget), so a no-tool goal loop
                # needs an unconditional backstop of its own.
                if self.active_goal["iterations"] >= GOAL_MAX_ITERATIONS:
                    print_info(f"Goal stopped: reached {GOAL_MAX_ITERATIONS} iterations without meeting the condition.")
                    break
                if self.goal_stop or self._aborted:
                    break

                await self.chat(
                    f"Hooks: Prompt hook condition was not met: {verdict['reason']}\n\nKeep working toward the goal."
                )
            if self.goal_stop or self._aborted:
                print_info("Goal pursuit interrupted.")
        finally:
            # Clear on any exit (met / impossible / capped / interrupted) so a
            # stale goal never lingers. Real Claude Code keeps it session-scoped
            # and resumable; we don't implement resume.
            self.active_goal = None

    async def _evaluate_goal(self, condition: str) -> dict:
        """对刚完成的轮次进行目标评估。

        将助手回复的转录文本作为独立消息发送给评估模型，
        防止精心构造的轮次向评估器上下文注入伪造的用户/评判文本。
        真实 Claude Code 也使用相同的隔离方式。

        Args:
            condition: 目标条件文本

        Returns:
            包含 ok（是否达成）、reason（原因）、impossible（是否不可能）的字典
        """
        transcript = self._extract_last_assistant_text()
        messages = [
            {"role": "user", "content": GOAL_TRANSCRIPT_FRAMING},
            {"role": "assistant", "content": transcript or "(no assistant output)"},
            {"role": "user", "content": goal_judge_user_message(condition)},
        ]
        try:
            raw = await self._run_evaluator_query(GOAL_EVALUATOR_SYSTEM, messages)
            return parse_goal_verdict(raw)
        except Exception as e:
            # Evaluator error → treat as not-met (never accidentally clears goal).
            return {"ok": False, "reason": f"evaluator error: {e}", "impossible": False}

    async def _run_evaluator_query(self, system: str, messages: list) -> str:
        """在当前后端上发送角色分离的评估查询。

        与 _build_side_query 类似，但接受完整的消息数组
        （_build_side_query 仅接受单条用户消息，用于记忆召回）。

        Args:
            system: 系统提示（评估器指令）
            messages: 消息数组，包含 user/assistant 角色交替

        Returns:
            模型回复的文本内容

        Raises:
            RuntimeError: 无可用的评估模型时抛出
        """
        if self._anthropic_client:
            resp = await self._anthropic_client.messages.create(
                model=self.model, max_tokens=512, system=system, temperature=0, messages=messages,
            )
            return "".join(b.text for b in resp.content if b.type == "text")
        if self._openai_client:
            resp = await self._openai_client.chat.completions.create(
                model=self.model, max_tokens=512, temperature=0,
                messages=[{"role": "system", "content": system}, *messages],
            )
            return resp.choices[0].message.content or "" if resp.choices else ""
        raise RuntimeError("no evaluator model available")

    async def _run_classifier_query(self, system: str, user: str, max_tokens: int) -> str:
        """单消息分类器查询，调用方可指定 max_tokens 预算。

        Auto Mode 的两个阶段使用不同的 token 预算：
        - 阶段 1 是轻量级门控（256 token）
        - 阶段 2 有更多空间进行推理（1024 token）
        temperature=0 确保确定性判定。

        Args:
            system: 分类器系统提示
            user: 用户消息（包含转录和规则）
            max_tokens: 最大输出 token 数

        Returns:
            分类器判定的文本结果

        Raises:
            RuntimeError: 无可用的分类器模型时抛出
        """
        if self._anthropic_client:
            resp = await self._anthropic_client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system, temperature=0,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in resp.content if b.type == "text")
        if self._openai_client:
            resp = await self._openai_client.chat.completions.create(
                model=self.model, max_tokens=max_tokens, temperature=0,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            return resp.choices[0].message.content or "" if resp.choices else ""
        raise RuntimeError("no classifier model available")

    def _extract_last_assistant_text(self) -> str:
        """提取最近一条助手消息的文本内容。

        仅用于评估器判定，获取最新的助手回复文本。
        支持 Anthropic（content 为列表）和 OpenAI（content 为字符串）两种格式。

        Returns:
            助手回复的纯文本，无内容时返回空字符串
        """
        if self.use_openai:
            for m in reversed(self._openai_messages):
                if m.get("role") == "assistant" and isinstance(m.get("content"), str):
                    return m["content"]
            return ""
        for m in reversed(self._anthropic_messages):
            if m.get("role") != "assistant":
                continue
            content = m.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
        return ""

    # ─── /loop — recurring or self-paced prompt ───────────────
    # 与 /goal（Stop-hook 门控）不同，/loop 主动重新调度自身：
    # 固定间隔模式，或无间隔时由主模型通过 schedule_wakeup 工具
    # 自行决定节奏。解析器和工具 schema 参见 autonomy.py。

    async def run_loop(self, raw_input: str) -> None:
        """执行 /loop 命令的入口点。

        解析输入，根据模式（interval/dynamic）驱动相应的循环。
        输入格式错误时直接返回而不循环。

        Args:
            raw_input: 用户的 /loop 命令原始输入
        """
        spec = parse_loop_input(raw_input)
        if "error" in spec:
            print_info(spec["error"])
            return
        # Offer-cloud decision point (interval >=60min or daily wording). Real
        # Claude Code asks whether to convert to a persistent cloud schedule that
        # survives the session; this teaching CLI has no cloud, so we only
        # surface it.
        wants_cloud = (
            (spec["mode"] == "interval" and spec["interval_seconds"] >= OFFER_CLOUD_THRESHOLD_SECONDS)
            or is_daily_wording(raw_input)
        )
        if wants_cloud:
            print_info(
                "(Real Claude Code would offer to convert this to a persistent cloud schedule "
                "that keeps running after the session ends. This teaching build has no cloud "
                "backend — continuing in-session.)"
            )

        self.loop_stop = False
        try:
            if spec["mode"] == "interval":
                await self._run_loop_interval(spec)
            else:
                await self._run_loop_dynamic(spec)
        except asyncio.CancelledError:
            print_info("Loop interrupted.")

    async def _run_loop_interval(self, spec: dict) -> None:
        """固定间隔循环模式：每 N 秒重新执行提示，直到中断或达到迭代上限。

        对应 Claude Code 的会话内 CronCreate 路径（仅会话有效，不持久化）。
        使用简单定时器替代 cron 引擎 + KAIROS 守护进程。

        Args:
            spec: 解析后的循环配置字典，包含 prompt、interval_seconds 等
        """
        print_info(
            f"⟳ /loop scheduled every {spec['interval_label']} (session-only, not persisted — "
            "dies when this process exits). Ctrl+C to stop."
        )
        iterations = 0
        while not self.loop_stop and not self._aborted:
            iterations += 1
            print_info(f"⟳ loop tick {iterations}")
            await self.chat(spec["prompt"])

            budget = self._check_budget()
            if budget["exceeded"]:
                print_info(f"Loop stopped: {budget['reason']}")
                break
            # --max-turns also bounds loop ticks: _check_budget's turn counter
            # only increments on tool-executing turns, so a plain-text loop would
            # never hit it — treat --max-turns as a tick limit here too.
            if self.max_turns is not None and iterations >= self.max_turns:
                print_info(f"Loop stopped: tick limit reached ({iterations} >= {self.max_turns}).")
                break
            if iterations >= LOOP_MAX_ITERATIONS:
                print_info(f"Loop stopped: reached {LOOP_MAX_ITERATIONS} ticks.")
                break
            interrupted = await self._interruptible_sleep(spec["interval_seconds"])
            if interrupted:
                print_info("Loop stopped.")
                break

    async def _run_loop_dynamic(self, spec: dict) -> None:
        """自适应动态循环模式：执行一轮后由主模型通过 schedule_wakeup 自行决定节奏。

        如果模型调度了唤醒，等待（钳位后的）延迟后用返回的提示重新执行；
        如果没有调度，说明循环已收敛。
        忠实于"动态节奏由主模型决定，无独立评估器"的设计。
        schedule_wakeup 工具仅在循环期间暴露。

        Args:
            spec: 解析后的循环配置字典，包含 prompt 等
        """
        print_info(
            "⟳ /loop dynamic (self-paced) — the model schedules its own next run, or ends the "
            "loop. Ctrl+C to stop."
        )
        had_tool = any(t["name"] == "schedule_wakeup" for t in self.tools)
        if not had_tool:
            self.tools = self.tools + [SCHEDULE_WAKEUP_TOOL]
        self.schedule_wakeup_enabled = True
        prompt = spec["prompt"]
        iterations = 0
        try:
            while not self.loop_stop and not self._aborted:
                iterations += 1
                self.pending_wakeup = None
                await self.chat(dynamic_loop_directive(prompt))

                if not self.pending_wakeup:
                    plural = "" if iterations == 1 else "s"
                    print_info(f"⟳ Loop converged after {iterations} tick{plural} (model scheduled no wakeup).")
                    break
                budget = self._check_budget()
                if budget["exceeded"]:
                    print_info(f"Loop stopped: {budget['reason']}")
                    break
                if self.max_turns is not None and iterations >= self.max_turns:
                    print_info(f"Loop stopped: tick limit reached ({iterations} >= {self.max_turns}).")
                    break
                if iterations >= LOOP_MAX_ITERATIONS:
                    print_info(f"Loop stopped: reached {LOOP_MAX_ITERATIONS} ticks.")
                    break
                delay = self.pending_wakeup["delay_seconds"]
                print_info(f"⟳ next run in {delay}s — {self.pending_wakeup['reason']}")
                prompt = self.pending_wakeup["prompt"] or prompt
                interrupted = await self._interruptible_sleep(delay)
                if interrupted:
                    print_info("Loop stopped.")
                    break
        finally:
            # Remove schedule_wakeup so it isn't exposed outside the dynamic loop.
            if not had_tool:
                self.tools = [t for t in self.tools if t["name"] != "schedule_wakeup"]
            self.schedule_wakeup_enabled = False
            self.pending_wakeup = None

    def _execute_schedule_wakeup(self, inp: dict) -> str:
        """执行 schedule_wakeup 工具：记录循环驱动器请求的唤醒。

        延迟被钳位到 [60, 3600] 秒范围内，
        循环驱动器在轮次收敛后读取 pending_wakeup。

        Args:
            inp: 工具输入参数，包含 delaySeconds、reason、prompt

        Returns:
            确认消息文本
        """
        delay = clamp_wakeup_delay(inp.get("delaySeconds"))
        reason = inp.get("reason") if isinstance(inp.get("reason"), str) else ""
        prompt = inp.get("prompt") if isinstance(inp.get("prompt"), str) else ""
        self.pending_wakeup = {"delay_seconds": delay, "reason": reason, "prompt": prompt}
        return f"Wakeup scheduled in {delay}s. The loop will resume then; end your turn now."

    async def _interruptible_sleep(self, seconds: float) -> bool:
        """可中断的异步睡眠。

        在循环停止或任务中止时提前返回 True，
        避免 Ctrl+C 后仍阻塞在长时间间隔中。

        Args:
            seconds: 睡眠时长（秒）

        Returns:
            True 表示被中断，False 表示正常完成
        """
        import time as _time
        start = _time.time()
        while _time.time() - start < seconds:
            if self.loop_stop or self._aborted:
                return True
            await asyncio.sleep(min(0.2, seconds))
        return False

    def stop_loop(self) -> None:
        """停止正在运行的 /loop 循环（由 REPL 中断处理器调用）。"""
        self.loop_stop = True

    def stop_goal(self) -> None:
        """停止正在执行的 /goal 追踪（由 REPL 中断处理器调用）。

        在下一个轮次边界生效 — 正在执行的轮次需通过 abort() 单独中止。
        """
        self.goal_stop = True

    # ─── Auto Mode — transcript-classifier permission gate ────
    # 在 auto 模式下，分类器替代人工确认提示：deny 规则仍然硬性阻止，
    # 只读工具快速通过，其余由 LLM 分类器通过推理盲转录投影来判定。

    async def _classify_tool_call(self, tool_name: str, inp: dict) -> dict:
        """在 Auto Mode 下判定工具调用是否允许。

        两阶段分类，镜像 Claude Code 的 `both` 模式：
        - 阶段 1：激进的廉价门控（不考虑用户意图，无 ALLOW 例外 —
          有任何规则可能适用就阻止）；如果阶段 1 放行则一次调用完成。
        - 阶段 2：如果阶段 1 阻止，进行仔细的裁决（考虑用户意图，
          可以清除阻止）。阶段 2 的判定是最终的。

        Args:
            tool_name: 工具名称
            inp: 工具输入参数

        Returns:
            包含 action（allow/deny/confirm）和 message 的字典
        """
        # Hard floor first: deny rules bind even here.
        base = check_permission(tool_name, inp, "default", self._plan_file_path)
        if base["action"] == "deny":
            return base
        # Fast-path: read-only / side-effect-free tools skip the classifier.
        if tool_name in AUTO_MODE_FAST_PATH_TOOLS:
            return {"action": "allow"}

        if not self._anthropic_client and not self._openai_client:
            # No evaluator available → fail closed. Defer to a human if present
            # (interactive), else deny outright (headless: CC aborts here).
            return self._auto_fallback(f"{tool_name} (auto-mode classifier unavailable)")
        try:
            rules = load_auto_mode_rules()
            history = self._openai_messages if self.use_openai else self._anthropic_messages
            transcript = build_classifier_transcript(history, {"tool_name": tool_name, "input": inp})
            system = build_classifier_system(rules)
            # CLAUDE.md rides in the user message, not the system prompt — it is
            # untrusted repo content.
            claude_md = load_claude_md()
            # Stage 1 — aggressive cheap gate (small token budget: just <block>…).
            s1_raw = await self._run_classifier_query(system, classifier_user_message(rules, transcript, rules["suffix_stage1"], claude_md), 256)
            s1 = parse_block_verdict(s1_raw)
            if not s1["block"]:
                verdict = s1  # stage 1 cleared it → allow (one call)
            else:
                # Stage 2 — careful adjudication (weighs user intent, can clear).
                # More tokens: stage 2 may emit a <thinking> block first.
                s2_raw = await self._run_classifier_query(system, classifier_user_message(rules, transcript, rules["suffix_stage2"], claude_md), 1024)
                verdict = parse_block_verdict(s2_raw)
        except Exception as e:
            # Any setup or classifier error → fail closed (block), matching CC's
            # iron gate. Wrapping the asset load here keeps a missing/bad rules
            # file from crashing the turn and orphaning the tool_use.
            verdict = {"block": True, "reason": f"classifier error: {e}"}

        if not verdict["block"]:
            self.auto_consecutive_denials = 0
            return {"action": "allow"}

        self.auto_consecutive_denials += 1
        self.auto_total_denials += 1
        if (
            self.auto_consecutive_denials >= DENIAL_LIMITS["max_consecutive"]
            or self.auto_total_denials >= DENIAL_LIMITS["max_total"]
        ):
            # Too many denials — the classifier may be stuck. Hand back to a human
            # if interactive; deny in headless (CC aborts the agent here).
            print_info("Auto Mode: denial limit reached — handing back to manual confirmation.")
            return self._auto_fallback(f"[Auto Mode blocked] {verdict['reason']}")
        return {"action": "deny", "message": f"[Auto Mode] {verdict['reason']}"}

    def _auto_fallback(self, message: str) -> dict:
        """Auto Mode 的安全回退机制。

        如果有人工确认回调则交回人工决策，否则在无头模式下直接拒绝。
        永远不会返回 "allow" — 目的是不执行未经过判定的操作。

        Args:
            message: 回退原因消息

        Returns:
            confirm（有人工回调时）或 deny（无头模式）操作字典
        """
        if self.confirm_fn:
            return {"action": "confirm", "message": message}
        return {"action": "deny", "message": f"{message} (headless — denied)"}

    def _child_permission_mode(self) -> str:
        """获取子 Agent 继承的权限模式。

        plan 和 auto 模式必须传递给子 Agent — 否则子 Agent 使用
        bypassPermissions，主模型可能通过 agent(prompt="git push")
        绕过分类器执行被阻止的操作。Claude Code 对每个子 Agent
        工具调用都单独进行 canUseTool 检查。

        Returns:
            子 Agent 应使用的权限模式字符串
        """
        if self.permission_mode == "plan":
            return "plan"
        if self.permission_mode == "auto":
            return "auto"
        return "bypassPermissions"

    # ─── Session ──────────────────────────────────────────────
    # 会话持久化：保存和恢复对话历史。

    def restore_session(self, data: dict) -> None:
        """从保存的数据恢复会话历史。

        Args:
            data: 包含 anthropicMessages 或 openaiMessages 的字典
        """
        if data.get("anthropicMessages"):
            self._anthropic_messages = data["anthropicMessages"]
        if data.get("openaiMessages"):
            self._openai_messages = data["openaiMessages"]
        print_info(f"Session restored ({self._get_message_count()} messages).")

    def _get_message_count(self) -> int:
        """获取当前消息历史中的消息数量。"""
        return len(self._openai_messages) if self.use_openai else len(self._anthropic_messages)

    def _auto_save(self) -> None:
        """自动保存会话到磁盘。

        每次 chat 完成后调用，保存元数据和对应后端的消息历史。
        静默失败，不影响用户体验。
        """
        try:
            save_session(self.session_id, {
                "metadata": {
                    "id": self.session_id,
                    "model": self.model,
                    "cwd": str(Path.cwd()),
                    "startTime": self.session_start_time,
                    "messageCount": self._get_message_count(),
                },
                "anthropicMessages": self._anthropic_messages if not self.use_openai else None,
                "openaiMessages": self._openai_messages if self.use_openai else None,
            })
        except Exception:
            pass

    # ─── Autocompact ──────────────────────────────────────────
    # 自动压缩：当上下文窗口利用率达到 85% 时触发，
    # 通过 LLM 总结历史对话来释放空间。

    async def _check_and_compact(self) -> None:
        """检查上下文窗口使用率，超过 85% 时自动触发压缩。"""
        if self.last_input_token_count > self.effective_window * 0.85:
            print_info("Context window filling up, compacting conversation...")
            await self._compact_conversation()

    async def _compact_conversation(self) -> None:
        """根据当前后端执行对话压缩。"""
        if self.use_openai:
            await self._compact_openai()
        else:
            await self._compact_anthropic()
        print_info("Conversation compacted.")

    async def _compact_anthropic(self) -> None:
        """压缩 Anthropic 后端的对话历史。

        使用 LLM 总结历史对话，然后用总结替换消息历史。
        不变量：调用者必须确保最后一条消息是纯用户文本消息
        （非 tool_result），否则会破坏 tool_use/tool_result 配对。
        """
        if len(self._anthropic_messages) < 4:
            return
        last_user_msg = self._anthropic_messages[-1]
        summary_resp = await self._anthropic_client.messages.create(
            model=self.model,
            max_tokens=2048,
            system="You are a conversation summarizer. Be concise but preserve important details.",
            messages=[
                *self._anthropic_messages[:-1],
                {"role": "user", "content": "Summarize the conversation so far in a concise paragraph, preserving key decisions, file paths, and context needed to continue the work."},
            ],
        )
        summary_text = summary_resp.content[0].text if summary_resp.content and summary_resp.content[0].type == "text" else "No summary available."
        self._anthropic_messages = [
            {"role": "user", "content": f"[Previous conversation summary]\n{summary_text}"},
            {"role": "assistant", "content": "Understood. I have the context from our previous conversation. How can I continue helping?"},
        ]
        if last_user_msg.get("role") == "user":
            self._anthropic_messages.append(last_user_msg)
        self.last_input_token_count = 0

    async def _compact_openai(self) -> None:
        """压缩 OpenAI 后端的对话历史。

        使用 LLM 总结历史对话，然后用总结替换消息历史。
        不变量：调用者必须确保最后一条消息是纯用户文本消息
        （非 tool 角色结果），原理同 _compact_anthropic。
        """
        if len(self._openai_messages) < 5:
            return
        system_msg = self._openai_messages[0]
        last_user_msg = self._openai_messages[-1]
        summary_resp = await self._openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a conversation summarizer. Be concise but preserve important details."},
                *self._openai_messages[1:-1],
                {"role": "user", "content": "Summarize the conversation so far in a concise paragraph, preserving key decisions, file paths, and context needed to continue the work."},
            ],
        )
        summary_text = summary_resp.choices[0].message.content or "No summary available."
        self._openai_messages = [
            system_msg,
            {"role": "user", "content": f"[Previous conversation summary]\n{summary_text}"},
            {"role": "assistant", "content": "Understood. I have the context from our previous conversation. How can I continue helping?"},
        ]
        if last_user_msg.get("role") == "user":
            self._openai_messages.append(last_user_msg)
        self.last_input_token_count = 0

    # ─── Multi-tier compression pipeline ──────────────────────
    # 四层压缩流水线：budget -> snip -> microcompact -> autocompact
    # 逐级释放上下文空间，避免溢出。

    def _run_compression_pipeline(self) -> None:
        """执行多层压缩流水线。

        按顺序执行三层本地压缩：
        1. Budget — 截断过大的工具结果
        2. Snip — 裁剪旧的可重新读取的工具结果
        3. Microcompact — 清除空闲超过 5 分钟的旧结果
        """
        if self.use_openai:
            self._budget_tool_results_openai()
            self._snip_stale_results_openai()
            self._microcompact_openai()
        else:
            self._budget_tool_results_anthropic()
            self._snip_stale_results_anthropic()
            self._microcompact_anthropic()

    # Tier 1: Budget tool results（第一层：工具结果预算截断）
    # 当上下文利用率超过 50% 时，将过大的工具结果截断到预算范围。
    # 利用率 > 70% 时预算收紧到 15000 字符，否则为 30000 字符。

    def _budget_tool_results_anthropic(self) -> None:
        """Anthropic 后端：截断过大的工具结果到预算范围。"""
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        if utilization < 0.5:
            return
        budget = 15000 if utilization > 0.7 else 30000
        for msg in self._anthropic_messages:
            if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                continue
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), str) and len(block["content"]) > budget:
                    keep = (budget - 80) // 2
                    block["content"] = block["content"][:keep] + f"\n\n[... budgeted: {len(block['content']) - keep * 2} chars truncated ...]\n\n" + block["content"][-keep:]

    def _budget_tool_results_openai(self) -> None:
        """OpenAI 后端：截断过大的工具结果到预算范围。"""
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        if utilization < 0.5:
            return
        budget = 15000 if utilization > 0.7 else 30000
        for msg in self._openai_messages:
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and len(msg["content"]) > budget:
                keep = (budget - 80) // 2
                msg["content"] = msg["content"][:keep] + f"\n\n[... budgeted: {len(msg['content']) - keep * 2} chars truncated ...]\n\n" + msg["content"][-keep:]

    # Tier 2: Snip stale results（第二层：裁剪旧结果）
    # 当缓存冷却后，将旧的可重新读取的工具结果替换为占位符。
    # 优先裁剪重复读取的同一文件，然后裁剪最旧的结果。

    def _snip_stale_results_anthropic(self) -> None:
        """Anthropic 后端：裁剪旧的工具结果。

        缓存感知：缓存热时避免修改前缀（除非利用率过高），
        缓存冷后才开始裁剪。保留最近 KEEP_RECENT_RESULTS 个结果。
        """
        # Cache-aware gate (mirrors Claude Code's cached-microcompact split):
        # while the prompt cache is still hot, rewriting an old tool_result in
        # place would invalidate the entire cached message prefix. Claude Code
        # prunes hot caches via a cache_edits API call unavailable on the public
        # API, so we leave the hot prefix alone — UNTIL utilization is high
        # enough (SNIP_HOT_OVERRIDE) that risking an overflow costs more than one
        # cache rebuild. Below that we wait for the cache to go cold.
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        cache_hot = self.last_api_call_time > 0 and (time.time() - self.last_api_call_time) < MICROCOMPACT_IDLE_S
        if cache_hot and utilization < SNIP_HOT_OVERRIDE:
            return
        if utilization < SNIP_THRESHOLD:
            return

        results = []
        for mi, msg in enumerate(self._anthropic_messages):
            if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                continue
            for bi, block in enumerate(msg["content"]):
                if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), str) and block["content"] != SNIP_PLACEHOLDER:
                    tool_use_id = block.get("tool_use_id")
                    tool_info = self._find_tool_use_by_id(tool_use_id)
                    if tool_info and tool_info["name"] in SNIPPABLE_TOOLS:
                        results.append({"mi": mi, "bi": bi, "name": tool_info["name"], "file_path": tool_info.get("input", {}).get("file_path")})

        if len(results) <= KEEP_RECENT_RESULTS:
            return

        to_snip = set()
        seen_files: dict[str, list[int]] = {}
        for i, r in enumerate(results):
            if r["name"] == "read_file" and r.get("file_path"):
                seen_files.setdefault(r["file_path"], []).append(i)

        for indices in seen_files.values():
            if len(indices) > 1:
                for j in indices[:-1]:
                    to_snip.add(j)

        snip_before = len(results) - KEEP_RECENT_RESULTS
        for i in range(snip_before):
            to_snip.add(i)

        for idx in to_snip:
            r = results[idx]
            self._anthropic_messages[r["mi"]]["content"][r["bi"]]["content"] = SNIP_PLACEHOLDER

    def _snip_stale_results_openai(self) -> None:
        """OpenAI 后端：裁剪旧的工具结果。

        与 Anthropic 版本逻辑相同，OpenAI 兼容提供者也自动缓存前缀，
        因此相同的"缓存热时不修改前缀"规则适用。
        """
        # Cache-aware gate — see _snip_stale_results_anthropic. OpenAI-compatible
        # providers cache prefixes automatically, so the same "don't rewrite a
        # hot prefix (unless utilization is high)" rule applies.
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        cache_hot = self.last_api_call_time > 0 and (time.time() - self.last_api_call_time) < MICROCOMPACT_IDLE_S
        if cache_hot and utilization < SNIP_HOT_OVERRIDE:
            return
        if utilization < SNIP_THRESHOLD:
            return
        tool_msgs = []
        for i, msg in enumerate(self._openai_messages):
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and msg["content"] != SNIP_PLACEHOLDER:
                tool_msgs.append(i)
        if len(tool_msgs) <= KEEP_RECENT_RESULTS:
            return
        snip_count = len(tool_msgs) - KEEP_RECENT_RESULTS
        for i in range(snip_count):
            self._openai_messages[tool_msgs[i]]["content"] = SNIP_PLACEHOLDER

    # Tier 3: Microcompact（第三层：微压缩）
    # 仅在空闲超过 5 分钟后触发，清除最旧的工具结果，
    # 保留最近 KEEP_RECENT_RESULTS 个。

    def _microcompact_anthropic(self) -> None:
        """Anthropic 后端：清除空闲超时后的旧工具结果。

        距离上次 API 调用超过 MICROCOMPACT_IDLE_S（5 分钟）才触发，
        将最旧的结果替换为 "[Old result cleared]" 占位符。
        """
        if not self.last_api_call_time or (time.time() - self.last_api_call_time) < MICROCOMPACT_IDLE_S:
            return
        all_results = []
        for mi, msg in enumerate(self._anthropic_messages):
            if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                continue
            for bi, block in enumerate(msg["content"]):
                if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), str) and block["content"] not in (SNIP_PLACEHOLDER, "[Old result cleared]"):
                    all_results.append((mi, bi))
        clear_count = len(all_results) - KEEP_RECENT_RESULTS
        for i in range(max(0, clear_count)):
            mi, bi = all_results[i]
            self._anthropic_messages[mi]["content"][bi]["content"] = "[Old result cleared]"

    def _microcompact_openai(self) -> None:
        """OpenAI 后端：清除空闲超时后的旧工具结果。"""
        if not self.last_api_call_time or (time.time() - self.last_api_call_time) < MICROCOMPACT_IDLE_S:
            return
        tool_msgs = []
        for i, msg in enumerate(self._openai_messages):
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and msg["content"] not in (SNIP_PLACEHOLDER, "[Old result cleared]"):
                tool_msgs.append(i)
        clear_count = len(tool_msgs) - KEEP_RECENT_RESULTS
        for i in range(max(0, clear_count)):
            self._openai_messages[tool_msgs[i]]["content"] = "[Old result cleared]"

    def _find_tool_use_by_id(self, tool_use_id: str) -> dict | None:
        """根据 tool_use_id 查找对应的工具调用信息。

        在 Anthropic 消息历史中搜索 tool_use 块，
        返回工具名称和输入参数。

        Args:
            tool_use_id: 工具调用的唯一 ID

        Returns:
            包含 name 和 input 的字典，未找到时返回 None
        """
        for msg in self._anthropic_messages:
            if msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
                continue
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id") == tool_use_id:
                    return {"name": block["name"], "input": block.get("input", {})}
        return None

    # ─── Large result persistence ─────────────────────────────────
    # 当工具结果超过 30 KB 时，写入磁盘并用简短预览 + 文件路径替换
    # 上下文中的条目。模型可以使用 read_file 稍后获取完整输出，
    # 不会丢失信息。

    def _persist_large_result(self, tool_name: str, result: str) -> str:
        """持久化大型工具结果到磁盘。

        超过 30 KB 的结果写入 ~/.mini-claude/tool-results/ 目录，
        返回包含预览和文件路径的截断版本。

        Args:
            tool_name: 工具名称
            result: 工具执行的原始结果文本

        Returns:
            原始结果（如果小于阈值）或包含预览的截断版本
        """
        THRESHOLD = 30 * 1024  # 30 KB
        if len(result.encode()) <= THRESHOLD:
            return result
        d = Path.home() / ".mini-claude" / "tool-results"
        d.mkdir(parents=True, exist_ok=True)
        # uuid suffix: parallel tools can persist in the same millisecond —
        # a timestamp-only name would let the second write clobber the first.
        filename = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}-{tool_name}.txt"
        filepath = d / filename
        filepath.write_text(result, encoding="utf-8")

        lines = result.split("\n")
        preview = "\n".join(lines[:200])
        size_kb = len(result.encode()) / 1024

        # Truncate AFTER persisting: the full result is already safe on disk,
        # so this only guards against pathological previews (e.g. a single
        # multi-hundred-KB line). Order matters — see issue #6.
        return _truncate_result(
            f"[Result too large ({size_kb:.1f} KB, {len(lines)} lines). "
            f"Full output saved to {filepath}. "
            f"You can use read_file to see the full result.]\n\n"
            f"Preview (first 200 lines):\n{preview}"
        )

    # ─── Execute tool (handles agent/skill/plan mode internally) ─────
    # 工具执行路由：将工具调用分发到对应的处理器。
    # 计划模式、子 Agent、技能、MCP 和普通工具各有专门的执行路径。

    async def _execute_tool_call(self, name: str, inp: dict) -> str:
        """执行工具调用的统一路由器。

        根据工具名称分发到对应的处理器：
        - enter_plan_mode/exit_plan_mode -> 计划模式工具
        - agent -> 子 Agent 执行器
        - skill -> 技能执行器
        - schedule_wakeup -> 循环唤醒调度器
        - MCP 工具 -> MCP 管理器
        - 其他 -> 标准工具执行器

        Args:
            name: 工具名称
            inp: 工具输入参数

        Returns:
            工具执行结果文本
        """
        if name in ("enter_plan_mode", "exit_plan_mode"):
            return await self._execute_plan_mode_tool(name)
        if name == "agent":
            return await self._execute_agent_tool(inp)
        if name == "skill":
            return await self._execute_skill_tool(inp)
        if name == "schedule_wakeup":
            # Only the internal dynamic-loop driver may route here; outside a
            # dynamic loop the tool isn't exposed, and this guard keeps a stray
            # call (or a same-named external tool) from reaching the executor.
            if not self.schedule_wakeup_enabled:
                return "schedule_wakeup is only available during /loop dynamic mode."
            return self._execute_schedule_wakeup(inp)
        # Route MCP tool calls to the MCP manager
        if self._mcp_manager.is_mcp_tool(name):
            return await self._mcp_manager.call_tool(name, inp)
        return await execute_tool(name, inp, self._read_file_state)

    # ─── Skill fork mode ─────────────────────────────────────
    # 技能分叉模式：fork 模式下创建子 Agent 执行技能任务，
    # 注入模式下将技能提示返回给主模型。

    async def _execute_skill_tool(self, inp: dict) -> str:
        """执行技能工具。

        根据技能配置选择执行模式：
        - fork 模式：创建子 Agent 在独立上下文中执行
        - 注入模式：将技能提示返回给主模型在当前上下文中执行

        Args:
            inp: 工具输入，包含 skill_name 和 args

        Returns:
            技能执行结果或激活消息
        """
        from .skills import execute_skill
        result = execute_skill(inp.get("skill_name", ""), inp.get("args", ""))
        if not result:
            return f"Unknown skill: {inp.get('skill_name', '')}"

        if result["context"] == "fork":
            # Never pass schedule_wakeup down — it's a driver-internal tool scoped
            # to this agent's dynamic loop, not something a forked skill inherits.
            tools = [
                t for t in (
                    [t for t in self.tools if t["name"] in result["allowed_tools"]]
                    if result.get("allowed_tools")
                    else [t for t in self.tools if t["name"] != "agent"]
                )
                if t["name"] != "schedule_wakeup"
            ]
            print_sub_agent_start("skill-fork", inp.get("skill_name", ""))
            sub_agent = Agent(
                model=self.model,
                api_base=str(self._openai_client.base_url) if self.use_openai and self._openai_client else None,
                custom_system_prompt=result["prompt"],
                custom_tools=tools,
                is_sub_agent=True,
                permission_mode=self._child_permission_mode(),
            )
            try:
                sub_result = await sub_agent.run_once(inp.get("args") or "Execute this skill task.")
                self.total_input_tokens += sub_result["tokens"]["input"]
                self.total_output_tokens += sub_result["tokens"]["output"]
                print_sub_agent_end("skill-fork", inp.get("skill_name", ""))
                return sub_result["text"] or "(Skill produced no output)"
            except Exception as e:
                print_sub_agent_end("skill-fork", inp.get("skill_name", ""))
                return f"Skill fork error: {e}"

        return f'[Skill "{inp.get("skill_name", "")}" activated]\n\n{result["prompt"]}'

    # ─── Plan mode helpers ──────────────────────────────────────
    # 计划模式辅助方法：生成计划文件路径、构建计划模式提示、
    # 执行计划模式工具（进入/退出计划模式）。

    def _generate_plan_file_path(self) -> str:
        """生成计划文件路径。

        在 ~/.claude/plans/ 目录下创建以会话 ID 命名的 markdown 文件。

        Returns:
            计划文件的完整路径
        """
        d = Path.home() / ".claude" / "plans"
        d.mkdir(parents=True, exist_ok=True)
        return str(d / f"plan-{self.session_id}.md")

    def _build_plan_mode_prompt(self) -> str:
        """构建计划模式的系统提示后缀。

        包含计划模式的工作流指令：探索 -> 设计 -> 写计划 -> 退出。

        Returns:
            计划模式提示文本
        """
        return f"""

# Plan Mode Active

Plan mode is active. You MUST NOT make any edits (except the plan file below), run non-readonly tools, or make any changes to the system.

## Plan File: {self._plan_file_path}
Write your plan incrementally to this file using write_file or edit_file. This is the ONLY file you are allowed to edit.

## Workflow
1. **Explore**: Read code to understand the task. Use read_file, list_files, grep_search.
2. **Design**: Design your implementation approach. Use the agent tool with type="plan" if the task is complex.
3. **Write Plan**: Write a structured plan to the plan file including:
   - **Context**: Why this change is needed
   - **Steps**: Implementation steps with critical file paths
   - **Verification**: How to test the changes
4. **Exit**: Call exit_plan_mode when your plan is ready for user review.

IMPORTANT: When your plan is complete, you MUST call exit_plan_mode. Do NOT ask the user to approve — exit_plan_mode handles that."""

    async def _execute_plan_mode_tool(self, name: str) -> str:
        """执行计划模式工具（进入/退出计划模式）。

        enter_plan_mode：保存当前权限模式，切换到只读计划模式。
        exit_plan_mode：读取计划文件，触发审批流程，恢复权限模式。

        Args:
            name: 工具名称（enter_plan_mode 或 exit_plan_mode）

        Returns:
            执行结果消息文本
        """
        if name == "enter_plan_mode":
            if self.permission_mode == "plan":
                return "Already in plan mode."
            self._pre_plan_mode = self.permission_mode
            self.permission_mode = "plan"
            self._plan_file_path = self._generate_plan_file_path()
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] = self._system_prompt
            print_info("Entered plan mode (read-only). Plan file: " + self._plan_file_path)
            return f"Entered plan mode. You are now in read-only mode.\n\nYour plan file: {self._plan_file_path}\nWrite your plan to this file. This is the only file you can edit.\n\nWhen your plan is complete, call exit_plan_mode."

        if name == "exit_plan_mode":
            if self.permission_mode != "plan":
                return "Not in plan mode."
            plan_content = "(No plan file found)"
            if self._plan_file_path and Path(self._plan_file_path).exists():
                plan_content = Path(self._plan_file_path).read_text()

            # Interactive approval flow
            if self._plan_approval_fn:
                result = await self._plan_approval_fn(plan_content)
                choice = result.get("choice", "manual-execute")

                if choice == "keep-planning":
                    feedback = result.get("feedback") or "Please revise the plan."
                    return (
                        f"User rejected the plan and wants to keep planning.\n\n"
                        f"User feedback: {feedback}\n\n"
                        f"Please revise your plan based on this feedback. When done, call exit_plan_mode again."
                    )

                # User approved — determine target mode
                if choice == "clear-and-execute":
                    target_mode = "acceptEdits"
                elif choice == "execute":
                    target_mode = "acceptEdits"
                else:  # manual-execute
                    target_mode = self._pre_plan_mode or "default"

                # Exit plan mode
                self.permission_mode = target_mode
                self._pre_plan_mode = None
                saved_plan_path = self._plan_file_path
                self._plan_file_path = None
                self._system_prompt = self._base_system_prompt
                if self.use_openai and self._openai_messages:
                    self._openai_messages[0]["content"] = self._system_prompt

                if choice == "clear-and-execute":
                    self._clear_history_keep_system()
                    self._context_cleared = True
                    print_info(f"Plan approved. Context cleared, executing in {target_mode} mode.")
                    return (
                        f"User approved the plan. Context was cleared. Permission mode: {target_mode}\n\n"
                        f"Plan file: {saved_plan_path}\n\n"
                        f"## Approved Plan:\n{plan_content}\n\n"
                        f"Proceed with implementation."
                    )

                print_info(f"Plan approved. Executing in {target_mode} mode.")
                return (
                    f"User approved the plan. Permission mode: {target_mode}\n\n"
                    f"## Approved Plan:\n{plan_content}\n\n"
                    f"Proceed with implementation."
                )

            # Fallback: no approval function (e.g. sub-agents)
            self.permission_mode = self._pre_plan_mode or "default"
            self._pre_plan_mode = None
            self._plan_file_path = None
            self._system_prompt = self._base_system_prompt
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] = self._system_prompt
            print_info("Exited plan mode. Restored to " + self.permission_mode + " mode.")
            return f"Exited plan mode. Permission mode restored to: {self.permission_mode}\n\n## Your Plan:\n{plan_content}"

        return f"Unknown plan mode tool: {name}"

    def _clear_history_keep_system(self) -> None:
        """清空对话历史但保留系统提示（用于计划审批后的上下文清除）。"""
        self._anthropic_messages = []
        self._openai_messages = []
        if self.use_openai:
            self._openai_messages.append({"role": "system", "content": self._system_prompt})
        self.last_input_token_count = 0

    async def _execute_agent_tool(self, inp: dict) -> str:
        """执行子 Agent 工具。

        根据 agent_type 创建对应的子 Agent，执行指定的 prompt，
        累加 token 消耗到父 Agent。

        Args:
            inp: 工具输入，包含 type、description、prompt

        Returns:
            子 Agent 的输出文本
        """
        agent_type = inp.get("type", "general")
        description = inp.get("description", "sub-agent task")
        prompt = inp.get("prompt", "")

        print_sub_agent_start(agent_type, description)

        config = get_sub_agent_config(agent_type)
        sub_agent = Agent(
            model=self.model,
            api_base=str(self._openai_client.base_url) if self.use_openai and self._openai_client else None,
            custom_system_prompt=config["system_prompt"],
            custom_tools=config["tools"],
            is_sub_agent=True,
            permission_mode=self._child_permission_mode(),
        )

        try:
            result = await sub_agent.run_once(prompt)
            self.total_input_tokens += result["tokens"]["input"]
            self.total_output_tokens += result["tokens"]["output"]
            print_sub_agent_end(agent_type, description)
            return result["text"] or "(Sub-agent produced no output)"
        except Exception as e:
            print_sub_agent_end(agent_type, description)
            return f"Sub-agent error: {e}"

    # ─── Anthropic backend ───────────────────────────────────────
    # Anthropic 后端的聊天实现，包含流式响应、工具执行、记忆预取等。

    async def close(self) -> None:
        """释放外部资源（MCP 子进程），确保干净退出（参见 issue #8）。"""
        if self._mcp_initialized:
            await self._mcp_manager.disconnect_all()

    def _consume_memory_prefetch_if_ready(self, messages: list) -> None:
        """如果预取已完成，注入召回的记忆（非阻塞）。

        将记忆追加到最后一条用户消息中，保持用户/助手交替格式。
        防止记忆注入破坏消息结构。

        Args:
            messages: 当前消息历史列表
        """
        pf = self._memory_prefetch
        if not (pf and pf.settled and not pf.consumed):
            return
        pf.consumed = True
        try:
            memories = pf.task.result()
            if not memories:
                return
            injection_text = format_memories_for_injection(memories)
            last = messages[-1] if messages else None
            if last and last.get("role") == "user":
                content = last.get("content", "")
                if isinstance(content, str) or content is None:
                    last["content"] = (content or "") + "\n\n" + injection_text
                elif isinstance(content, list):
                    content.append({"type": "text", "text": injection_text})
            else:
                messages.append({"role": "user", "content": injection_text})
            for m in memories:
                self._already_surfaced_memories.add(m.path)
                self._session_memory_bytes += len(m.content.encode())
        except Exception:
            pass  # prefetch errors already logged

    def _start_memory_prefetch_for_turn(self, user_message: str, messages: list) -> None:
        """为当前轮次启动记忆预取。

        先排空上一轮遗留的预取结果（在上次 API 调用后才完成的预取
        会被丢弃 — issue #7），然后为当前查询启动新的预取。

        Args:
            user_message: 用户消息文本
            messages: 当前消息历史列表
        """
        self._consume_memory_prefetch_if_ready(messages)
        if self.is_sub_agent:
            return
        if self._memory_prefetch and not self._memory_prefetch.settled:
            self._memory_prefetch.task.cancel()
        sq = self._build_side_query()
        if sq:
            self._memory_prefetch = start_memory_prefetch(
                user_message, sq,
                self._already_surfaced_memories, self._session_memory_bytes,
            )

    def _push_anthropic_user_message(self, content: str) -> None:
        """推送 Anthropic 格式的用户消息。

        在（可能刚清除的）上下文的第一条用户消息前插入
        CLAUDE.md/日期的 system-reminder（Claude Code 的 prependUserContext），
        嵌入用户消息而非独立消息以保持 user/assistant 交替。
        也用于计划审批后从空历史重建上下文的路径。

        Args:
            content: 用户消息文本
        """
        if not self._anthropic_messages and self._user_context_reminder:
            self._anthropic_messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": self._user_context_reminder},
                    {"type": "text", "text": content},
                ],
            })
        else:
            self._anthropic_messages.append({"role": "user", "content": content})

    def _push_openai_user_message(self, content: str) -> None:
        """推送 OpenAI 格式的用户消息。

        与 Anthropic 版本相同逻辑，在第一条用户消息中嵌入 system-reminder。

        Args:
            content: 用户消息文本
        """
        is_first_user = not any(m.get("role") == "user" for m in self._openai_messages)
        if is_first_user and self._user_context_reminder:
            self._openai_messages.append({"role": "user", "content": f"{self._user_context_reminder}\n\n{content}"})
        else:
            self._openai_messages.append({"role": "user", "content": content})

    async def _chat_anthropic(self, user_message: str) -> None:
        """Anthropic 后端的聊天主循环。

        流程：
        1. 推送用户消息
        2. 检查并触发自动压缩
        3. 启动记忆预取
        4. 循环：运行压缩管道 -> 调用 API -> 处理工具调用 -> 直到无工具调用

        支持流式工具执行：tool_use 块在流式传输完成时立即开始执行，
        无需等待整个响应完成。
        """
        self._push_anthropic_user_message(user_message)
        # Auto-compact at turn boundary only — the last message is now plain
        # user text, so the slice in _compact_anthropic won't sever a
        # tool_use ↔ tool_result pair from the previous turn's tool execution.
        await self._check_and_compact()

        # Memory prefetch: drain carry-over, then start fresh (issue #7)
        self._start_memory_prefetch_for_turn(user_message, self._anthropic_messages)

        while True:
            if self._aborted:
                break

            self._run_compression_pipeline()

            # Consume memory prefetch if settled (non-blocking poll, zero-wait)
            self._consume_memory_prefetch_if_ready(self._anthropic_messages)

            if not self.is_sub_agent:
                start_spinner()

            # ── Streaming tool execution ──────────────────────────────
            # As each tool_use content block completes during streaming, check
            # if it's concurrency-safe and auto-allowed. If so, start execution
            # immediately — the tool runs while the model still generates.
            early_executions: dict[str, asyncio.Task] = {}

            def _on_tool_block(block: dict):
                # In Auto Mode, only fast-path (classifier-exempt) tools may start
                # early — otherwise a concurrency-safe-but-classified tool (e.g.
                # web_fetch) would run before the classifier ever sees it.
                if self.permission_mode == "auto" and block["name"] not in AUTO_MODE_FAST_PATH_TOOLS:
                    return
                if block["name"] in CONCURRENCY_SAFE_TOOLS:
                    perm = check_permission(block["name"], block["input"], self.permission_mode, self._plan_file_path)
                    if perm["action"] == "allow":
                        task = asyncio.create_task(self._execute_tool_call(block["name"], block["input"]))
                        early_executions[block["id"]] = task

            response = await self._call_anthropic_stream(on_tool_block_complete=_on_tool_block)

            if not self.is_sub_agent:
                stop_spinner()

            self.last_api_call_time = time.time()
            # Anthropic reports cached tokens separately: `input_tokens` counts
            # only the uncached prefix, while cache_read/cache_creation are
            # billed at 0.1x/1.25x. Track them apart for cost.
            cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
            cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
            self.total_input_tokens += response.usage.input_tokens
            self.total_cache_read_tokens += cache_read
            self.total_cache_creation_tokens += cache_creation
            self.total_output_tokens += response.usage.output_tokens
            # Estimate next-turn context size for the compaction gauge: the full
            # prompt we just sent (input + cache_read + cache_creation) plus the
            # output we just generated, which becomes part of the next request.
            self.last_input_token_count = (
                response.usage.input_tokens + cache_read + cache_creation + response.usage.output_tokens
            )

            tool_uses = [b for b in response.content if b.type == "tool_use"]

            self._anthropic_messages.append({
                "role": "assistant",
                "content": [self._block_to_dict(b) for b in response.content],
            })

            if not tool_uses:
                if not self.is_sub_agent:
                    print_cost(self.total_input_tokens, self.total_output_tokens, self.total_cache_read_tokens, self.total_cache_creation_tokens)
                break

            self.current_turns += 1
            budget = self._check_budget()
            if budget["exceeded"]:
                print_info(f"Budget exceeded: {budget['reason']}")
                # Every tool_use needs a paired tool_result or the message
                # history is invalid for the next API call. Pair each pending
                # call with a refusal instead of silently dropping it.
                for task in early_executions.values():
                    task.cancel()
                self._anthropic_messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": tu.id,
                     "content": f"Tool call not executed: {budget['reason']}"}
                    for tu in tool_uses
                ]})
                break

            # Process tools: early-started ones (from streaming) just await
            # their result; others go through permission check + execution.
            tool_results: list[dict] = []
            context_break = False
            for tu in tool_uses:
                if context_break or self._aborted:
                    break
                inp = dict(tu.input) if hasattr(tu.input, 'items') else tu.input
                print_tool_call(tu.name, inp)

                # Was this tool already started during streaming?
                early_task = early_executions.get(tu.id)
                if early_task:
                    raw = await early_task
                    res = self._persist_large_result(tu.name, raw)
                    print_tool_result(tu.name, res)
                    tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": res})
                    continue

                # Permission check for tools not started early. Auto Mode routes
                # through the transcript classifier; other modes use static rules.
                if self.permission_mode == "auto":
                    perm = await self._classify_tool_call(tu.name, inp)
                else:
                    perm = check_permission(tu.name, inp, self.permission_mode, self._plan_file_path)
                if perm["action"] == "deny":
                    print_info(f"Denied: {perm.get('message', '')}")
                    tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": f"Action denied: {perm.get('message', '')}"})
                    continue
                if perm["action"] == "confirm" and perm.get("message"):
                    # Auto Mode confirms carry a reason, not a path — never cache
                    # them, or one approval would whitelist every later action
                    # with the same reason.
                    cacheable = self.permission_mode != "auto"
                    if not cacheable or perm["message"] not in self._confirmed_paths:
                        confirmed = await self._confirm_dangerous(perm["message"])
                        if not confirmed:
                            tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": "User denied this action."})
                            continue
                        if cacheable:
                            self._confirmed_paths.add(perm["message"])

                raw = await self._execute_tool_call(tu.name, inp)
                res = self._persist_large_result(tu.name, raw)
                print_tool_result(tu.name, res)

                if self._context_cleared:
                    self._context_cleared = False
                    # History was just cleared — route through the helper so the
                    # rebuilt context's first user message carries the reminder.
                    self._push_anthropic_user_message(res)
                    context_break = True
                    break
                tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": res})

            if not context_break and tool_results:
                self._anthropic_messages.append({"role": "user", "content": tool_results})
            self._context_cleared = False

    @staticmethod
    def _block_to_dict(block) -> dict:
        """将 Anthropic 内容块转换为纯字典格式以便存储。"""
        if block.type == "text":
            return {"type": "text", "text": block.text}
        if block.type == "tool_use":
            return {"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.input) if hasattr(block.input, 'items') else block.input}
        # Fallback
        return {"type": block.type}

    async def _call_anthropic_stream(self, on_tool_block_complete=None):
        """流式调用 Anthropic API。

        当 tool_use 内容块在流式传输中完成时，立即触发 on_tool_block_complete
        回调，使调用方可以在完整响应到达前开始执行（流式工具执行 ——
        受 Claude Code 的 content_block_stop 流式模式启发）。

        Args:
            on_tool_block_complete: 工具块完成时的回调函数

        Returns:
            Anthropic API 的完整响应消息
        """
        async def _do():
            max_output = _get_max_output_tokens(self.model)
            create_params: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_output if self._thinking_mode != "disabled" else 16384,
                "system": self._build_anthropic_system(),
                "tools": get_active_tool_definitions(self.tools),
                # Rolling message-array cache breakpoint, applied to a copy so
                # the persistent history stays free of cache_control metadata.
                "messages": self._with_cache_breakpoints(self._anthropic_messages),
            }

            if self._thinking_mode in ("adaptive", "enabled"):
                create_params["thinking"] = {"type": "enabled", "budget_tokens": max_output - 1}

            first_text = True
            # Track in-flight tool_use blocks by index for streaming execution
            tool_blocks_by_index: dict[int, dict] = {}

            async with self._anthropic_client.messages.stream(**create_params) as stream:
                async for event in stream:
                    if not hasattr(event, 'type'):
                        continue

                    if event.type == "content_block_start":
                        cb = getattr(event, 'content_block', None)
                        if cb and getattr(cb, 'type', None) == "tool_use":
                            tool_blocks_by_index[event.index] = {
                                "id": cb.id, "name": cb.name, "input_json": "",
                            }

                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if hasattr(delta, 'text'):
                            if first_text:
                                stop_spinner()
                                self._emit_text("\n")
                                first_text = False
                            self._emit_text(delta.text)
                        elif hasattr(delta, 'thinking'):
                            if first_text:
                                stop_spinner()
                                self._emit_text("\n  [thinking] ")
                                first_text = False
                            self._emit_text(delta.thinking)
                        elif hasattr(delta, 'partial_json'):
                            tb = tool_blocks_by_index.get(event.index)
                            if tb:
                                tb["input_json"] += delta.partial_json

                    elif event.type == "content_block_stop":
                        tb = tool_blocks_by_index.pop(event.index, None)
                        if tb and on_tool_block_complete:
                            import json as _json
                            try:
                                parsed = _json.loads(tb["input_json"] or "{}")
                            except Exception:
                                parsed = {}
                            on_tool_block_complete({
                                "type": "tool_use", "id": tb["id"],
                                "name": tb["name"], "input": parsed,
                            })

                final_message = await stream.get_final_message()

            # Filter out thinking blocks
            final_message.content = [b for b in final_message.content if b.type != "thinking"]
            return final_message

        return await _with_retry(_do)

    # ─── OpenAI-compatible backend ───────────────────────────────
    # OpenAI 兼容后端的聊天实现，支持并行工具执行。

    async def _chat_openai(self, user_message: str) -> None:
        """OpenAI 后端的聊天主循环。

        流程与 Anthropic 版本类似，但工具执行支持并行：
        连续的安全工具（如 grep_search）可以分组并行执行。

        Args:
            user_message: 用户输入的消息文本
        """
        self._push_openai_user_message(user_message)
        # Auto-compact at turn boundary only — see _chat_anthropic for rationale.
        # The last message is now plain user text, so the slice in
        # _compact_openai won't orphan a tool_calls / tool message pair.
        await self._check_and_compact()

        # Memory prefetch: drain carry-over, then start fresh (issue #7)
        self._start_memory_prefetch_for_turn(user_message, self._openai_messages)

        while True:
            if self._aborted:
                break

            self._run_compression_pipeline()

            # Consume memory prefetch if settled (non-blocking poll, zero-wait)
            self._consume_memory_prefetch_if_ready(self._openai_messages)

            if not self.is_sub_agent:
                start_spinner()

            response = await self._call_openai_stream()

            if not self.is_sub_agent:
                stop_spinner()

            self.last_api_call_time = time.time()

            if response.get("usage"):
                # OpenAI-compatible providers cache prefixes automatically; the
                # cached portion is included in prompt_tokens, so split it out to
                # avoid double-counting. Clamp to [0, prompt_tokens] since
                # compatible gateways don't guarantee the field. NOTE: priced at
                # Anthropic's 0.1x for simplicity; actual cached rates vary by
                # provider (OpenAI ~0.5x, gateways vary), so the estimate may be
                # off in either direction.
                prompt = response["usage"]["prompt_tokens"] or 0
                cached_oa = min(max(response["usage"].get("cached_tokens", 0) or 0, 0), prompt)
                completion = response["usage"]["completion_tokens"]
                self.total_input_tokens += prompt - cached_oa
                self.total_cache_read_tokens += cached_oa
                self.total_output_tokens += completion
                # Estimate next-turn context size: this prompt + the output we
                # just generated (which becomes part of the next request).
                self.last_input_token_count = prompt + completion

            choice = response.get("choices", [{}])[0] if response.get("choices") else {}
            message = choice.get("message", {})

            self._openai_messages.append(message)

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                if not self.is_sub_agent:
                    print_cost(self.total_input_tokens, self.total_output_tokens, self.total_cache_read_tokens, self.total_cache_creation_tokens)
                break

            self.current_turns += 1
            budget = self._check_budget()
            if budget["exceeded"]:
                print_info(f"Budget exceeded: {budget['reason']}")
                # Same pairing requirement as the Anthropic path: every
                # tool_call needs a role="tool" response.
                for tc in tool_calls:
                    if tc.get("id"):
                        self._openai_messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": f"Tool call not executed: {budget['reason']}",
                        })
                break

            # Phase 1: Parse & permission-check (serial)
            oai_checked: list[dict] = []
            for tc in tool_calls:
                if self._aborted:
                    break
                if tc.get("type") != "function":
                    continue
                fn_name = tc["function"]["name"]
                try:
                    inp = json.loads(tc["function"]["arguments"])
                except Exception:
                    inp = {}

                print_tool_call(fn_name, inp)

                if self.permission_mode == "auto":
                    perm = await self._classify_tool_call(fn_name, inp)
                else:
                    perm = check_permission(fn_name, inp, self.permission_mode, self._plan_file_path)
                if perm["action"] == "deny":
                    print_info(f"Denied: {perm.get('message', '')}")
                    oai_checked.append({"tc": tc, "fn": fn_name, "inp": inp, "allowed": False, "result": f"Action denied: {perm.get('message', '')}"})
                    continue
                if perm["action"] == "confirm" and perm.get("message"):
                    # Auto Mode confirms carry a reason, not a path — never cache
                    # them, or one approval would whitelist every later action
                    # with the same reason.
                    cacheable = self.permission_mode != "auto"
                    if not cacheable or perm["message"] not in self._confirmed_paths:
                        confirmed = await self._confirm_dangerous(perm["message"])
                        if not confirmed:
                            oai_checked.append({"tc": tc, "fn": fn_name, "inp": inp, "allowed": False, "result": "User denied this action."})
                            continue
                        if cacheable:
                            self._confirmed_paths.add(perm["message"])
                oai_checked.append({"tc": tc, "fn": fn_name, "inp": inp, "allowed": True})

            # Phase 2: Group & execute (parallel for consecutive safe tools)
            oai_batches: list[dict] = []
            for ct in oai_checked:
                safe = ct["allowed"] and ct["fn"] in CONCURRENCY_SAFE_TOOLS
                if safe and oai_batches and oai_batches[-1]["concurrent"]:
                    oai_batches[-1]["items"].append(ct)
                else:
                    oai_batches.append({"concurrent": safe, "items": [ct]})

            oai_context_break = False
            for batch in oai_batches:
                if oai_context_break or self._aborted:
                    break

                if batch["concurrent"]:
                    async def _run_oai_safe(ct_item: dict) -> tuple[dict, str]:
                        raw = await self._execute_tool_call(ct_item["fn"], ct_item["inp"])
                        res = self._persist_large_result(ct_item["fn"], raw)
                        print_tool_result(ct_item["fn"], res)
                        return ct_item, res

                    results = await asyncio.gather(*[_run_oai_safe(ct) for ct in batch["items"]])
                    for ct_item, res in results:
                        self._openai_messages.append({"role": "tool", "tool_call_id": ct_item["tc"]["id"], "content": res})
                else:
                    for ct in batch["items"]:
                        if not ct["allowed"]:
                            self._openai_messages.append({"role": "tool", "tool_call_id": ct["tc"]["id"], "content": ct["result"]})
                            continue
                        raw = await self._execute_tool_call(ct["fn"], ct["inp"])
                        res = self._persist_large_result(ct["fn"], raw)
                        print_tool_result(ct["fn"], res)

                        if self._context_cleared:
                            self._context_cleared = False
                            # History was just cleared — route through the helper
                            # so the first user message carries the reminder.
                            self._push_openai_user_message(res)
                            oai_context_break = True
                            break
                        self._openai_messages.append({"role": "tool", "tool_call_id": ct["tc"]["id"], "content": res})

            self._context_cleared = False

    async def _call_openai_stream(self) -> dict:
        """流式调用 OpenAI 兼容 API。

        从流式响应中组装完整的助手消息，包括文本内容和工具调用。
        支持 usage 统计（包括缓存 token）。

        Returns:
            模拟 Anthropic 格式的响应字典，包含 choices 和 usage
        """
        async def _do():
            stream = await self._openai_client.chat.completions.create(
                model=self.model,
                max_tokens=16384,
                tools=_to_openai_tools(get_active_tool_definitions(self.tools)),
                messages=self._openai_messages,
                stream=True,
                stream_options={"include_usage": True},
            )

            content = ""
            first_text = True
            tool_calls: dict[int, dict] = {}
            finish_reason = ""
            usage = None

            async for chunk in stream:
                if chunk.usage:
                    details = getattr(chunk.usage, "prompt_tokens_details", None)
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "cached_tokens": getattr(details, "cached_tokens", 0) or 0,
                    }

                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if delta and delta.content:
                    if first_text:
                        stop_spinner()
                        self._emit_text("\n")
                        first_text = False
                    self._emit_text(delta.content)
                    content += delta.content

                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        existing = tool_calls.get(tc.index)
                        if existing:
                            if tc.function and tc.function.arguments:
                                existing["arguments"] += tc.function.arguments
                        else:
                            tool_calls[tc.index] = {
                                "id": tc.id or "",
                                "name": (tc.function.name if tc.function else "") or "",
                                "arguments": (tc.function.arguments if tc.function else "") or "",
                            }

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            assembled = None
            if tool_calls:
                assembled = [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for _, tc in sorted(tool_calls.items())
                ]

            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": assembled,
                    },
                    "finish_reason": finish_reason or "stop",
                }],
                "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0},
            }

        return await _with_retry(_do)

    # ─── Shared ──────────────────────────────────────────────────
    # 后端共享的工具方法。

    async def _confirm_dangerous(self, command: str) -> bool:
        """请求用户确认危险操作。

        显示操作详情，通过 confirm_fn 回调或阻塞式输入获取用户许可。

        Args:
            command: 需要确认的操作描述

        Returns:
            True 表示用户允许，False 表示拒绝
        """
        print_confirmation(command)
        if self.confirm_fn:
            return await self.confirm_fn(command)
        # Fallback: blocking input
        try:
            answer = input("  Allow? (y/n): ")
            return answer.lower().startswith("y")
        except EOFError:
            return False
