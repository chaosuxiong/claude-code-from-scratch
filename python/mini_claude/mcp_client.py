"""
MCP Client — connects to stdio-based MCP servers, discovers and forwards tool calls.
Uses raw JSON-RPC over stdio (no SDK dependency for simplicity).

Config is read from .claude/settings.json and ~/.claude/settings.json:
  { "mcpServers": { "name": { "command": "...", "args": [...], "env": {...} } } }

Each MCP tool is exposed with a "mcp__serverName__toolName" prefix to avoid conflicts.
"""

# ──────────────────────────────────────────────────────────────────────────────
# MCP 客户端模块 (Model Context Protocol Client)
# ──────────────────────────────────────────────────────────────────────────────
# 本模块实现了与基于 stdio 的 MCP 服务器的连接、工具发现和工具调用转发。
# 使用原始的 JSON-RPC 协议通过 stdio 通信，不依赖任何 SDK。
#
# 配置来源（按优先级合并）：
#   1. ~/.claude/settings.json    — 全局用户配置
#   2. .claude/settings.json      — 项目级配置
#   3. .mcp.json                  — Claude Code 约定的 MCP 配置文件
#
# 配置格式：
#   { "mcpServers": { "name": { "command": "...", "args": [...], "env": {...} } } }
#
# 每个 MCP 工具通过 "mcp__serverName__toolName" 前缀命名，避免与内置工具冲突。
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any


# ─── 单个 MCP 连接（每个服务器一个连接实例） ──────────────────

# McpConnection 类负责管理与单个 MCP 服务器进程的生命周期和 JSON-RPC 通信。
# 每个实例对应一个子进程，通过 stdin/stdout 进行双向 JSON-RPC 消息传递。
class McpConnection:
    """Manages a single MCP server process and JSON-RPC communication."""

    def __init__(self, server_name: str, command: str, args: list[str] | None = None,
                 env: dict[str, str] | None = None):
        """初始化 MCP 连接实例

        Args:
            server_name: 服务器名称（用于日志和工具前缀）
            command: 要执行的命令（如 "npx", "python"）
            args: 命令参数列表
            env: 额外的环境变量（会与当前进程环境合并）
        """
        self.server_name = server_name
        self.command = command
        self.args = args or []
        self.env = env or {}  # 额外环境变量，与 os.environ 合并后传给子进程
        self._process: asyncio.subprocess.Process | None = None  # 子进程引用
        self._next_id = 1  # JSON-RPC 请求 ID 递增计数器
        self._pending: dict[int, asyncio.Future] = {}  # 待响应的请求 {id: Future}
        self._reader_task: asyncio.Task | None = None  # stdout 读取后台任务

    async def connect(self) -> None:
        """启动 MCP 服务器子进程

        将当前环境变量与用户配置的额外环境变量合并，然后以管道模式启动子进程。
        同时启动后台任务持续读取 stdout 的 JSON-RPC 响应。
        """
        merged_env = {**os.environ, **self.env}  # 合并环境变量（用户配置优先）
        self._process = await asyncio.create_subprocess_exec(
            self.command, *self.args,
            stdin=subprocess.PIPE,   # stdin 用于发送请求
            stdout=subprocess.PIPE,  # stdout 用于接收响应
            stderr=subprocess.PIPE,  # stderr 捕获但不处理
            env=merged_env,
        )
        # 启动后台任务持续从 stdout 读取 JSON-RPC 响应行
        self._reader_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        """从 stdout 持续读取换行分隔的 JSON-RPC 响应

        每读取一行就尝试 JSON 解析，根据响应中的 id 匹配待响应的请求：
        - 有 error 字段 → 设置 Future 异常
        - 无 error → 设置 Future 结果值
        - id 不匹配任何待响应请求 → 静默忽略（可能是通知消息）
        当 stdout 关闭（子进程退出）时循环结束。
        """
        assert self._process and self._process.stdout
        while True:
            line = await self._process.stdout.readline()
            if not line:
                break  # 子进程关闭了 stdout，退出循环
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # 无法解析的行静默忽略
            msg_id = msg.get("id")
            if msg_id is not None and msg_id in self._pending:
                fut = self._pending.pop(msg_id)
                if "error" in msg:
                    # 服务器返回了错误响应，将错误设置到对应的 Future 上
                    e = msg["error"]
                    fut.set_exception(
                        RuntimeError(f"MCP error {e.get('code')}: {e.get('message')}")
                    )
                else:
                    # 正常响应，将结果设置到对应的 Future 上
                    fut.set_result(msg.get("result"))

    async def _send_request(self, method: str, params: dict | None = None) -> Any:
        """发送 JSON-RPC 请求并等待响应

        构造符合 JSON-RPC 2.0 规范的请求消息，写入子进程 stdin，
        然后创建一个 Future 并注册到待响应字典中，等待 _read_loop 匹配响应。
        """
        assert self._process and self._process.stdin
        req_id = self._next_id
        self._next_id += 1  # 递增请求 ID
        # 构造 JSON-RPC 2.0 请求并写入 stdin
        msg = json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}})
        self._process.stdin.write((msg + "\n").encode())
        await self._process.stdin.drain()  # 确保数据已刷新到管道
        # 创建 Future 并注册到待响应字典，等待 _read_loop 处理响应
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut
        return await fut  # 阻塞直到 _read_loop 设置结果或异常

    def _send_notification(self, method: str, params: dict | None = None) -> None:
        """发送 JSON-RPC 通知（不需要响应）

        通知消息与请求的区别在于没有 "id" 字段，服务器不会返回响应。
        用于发送如 "initialized" 等确认通知。
        """
        if not self._process or not self._process.stdin:
            return
        # 注意：通知消息没有 "id" 字段，服务器不会响应
        msg = json.dumps({"jsonrpc": "2.0", "method": method, "params": params or {}})
        self._process.stdin.write((msg + "\n").encode())

    async def initialize(self) -> None:
        """执行 MCP 初始化握手

        按照 MCP 协议规范，客户端首先发送 "initialize" 请求告知协议版本和客户端信息，
        然后发送 "notifications/initialized" 通知表示初始化完成。
        """
        await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",  # MCP 协议版本
            "capabilities": {},  # 客户端能力声明（此处为空）
            "clientInfo": {"name": "mini-claude", "version": "1.0.0"},  # 客户端标识
        })
        # 发送初始化完成通知（不需要响应）
        self._send_notification("notifications/initialized")

    async def list_tools(self) -> list[dict]:
        """从服务器发现可用工具列表

        调用 "tools/list" 方法获取服务器提供的所有工具定义，
        并为每个工具附加 serverName 字段以便后续路由调用。
        """
        result = await self._send_request("tools/list")
        if not result or not isinstance(result.get("tools"), list):
            return []  # 服务器未返回有效工具列表
        # 将服务器返回的工具定义规范化，并附加服务器名称
        return [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "inputSchema": t.get("inputSchema"),
                "serverName": self.server_name,  # 记录来源服务器
            }
            for t in result["tools"]
        ]

    async def call_tool(self, name: str, args: dict) -> str:
        """调用服务器上的工具并返回文本结果

        发送 "tools/call" 请求，从响应中提取所有 type="text" 的内容块，
        用换行符连接后返回。如果响应格式不符合预期，则将整个结果 JSON 序列化返回。
        """
        result = await self._send_request("tools/call", {"name": name, "arguments": args})
        # MCP 工具响应格式：{ "content": [{ "type": "text", "text": "..." }, ...] }
        if isinstance(result, dict) and isinstance(result.get("content"), list):
            # 提取所有文本类型的内容块并拼接
            return "\n".join(
                c["text"] for c in result["content"] if c.get("type") == "text"
            )
        return json.dumps(result)  # 降级：将整个结果序列化为 JSON 字符串

    def close(self) -> None:
        """关闭连接：终止服务器子进程并清理所有资源

        执行顺序：
          1. 取消 stdout 读取后台任务
          2. 强制终止子进程
          3. 拒绝所有待响应的请求（设置异常）
          4. 清空待响应字典
        """
        if self._reader_task:
            self._reader_task.cancel()
            self._reader_task = None
        if self._process:
            try:
                self._process.kill()  # 强制终止子进程
            except ProcessLookupError:
                pass  # 进程已退出，忽略
            self._process = None
        # 拒绝所有待响应的请求，通知调用方服务器已关闭
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError(f"MCP server '{self.server_name}' closed"))
        self._pending.clear()


# ─── MCP 管理器 ─────────────────────────────────────────────

# McpManager 是 MCP 功能的顶层入口，负责：
#   1. 从多个配置文件加载 MCP 服务器配置
#   2. 建立与所有服务器的连接
#   3. 发现并注册所有可用工具
#   4. 将带前缀的工具调用路由到正确的服务器
class McpManager:
    """Manages all MCP server connections. Call load_and_connect() once, then
    use get_tool_definitions() and call_tool() to integrate with the agent."""

    def __init__(self):
        self._connections: dict[str, McpConnection] = {}  # 已连接的服务器 {名称: 连接实例}
        self._tools: list[dict] = []  # 所有服务器发现的工具列表
        self._connected = False  # 是否已完成连接（防止重复连接）

    async def load_and_connect(self) -> None:
        """读取配置、连接所有配置的 MCP 服务器、发现工具

        执行流程：
          1. 检查是否已连接（防止重复调用）
          2. 从配置文件加载所有 MCP 服务器配置
          3. 逐个建立连接、执行初始化握手、发现工具
          4. 失败的连接会被静默跳过（不阻断其他服务器）
        """
        if self._connected:
            return  # 已连接，跳过重复初始化
        self._connected = True

        configs = self._load_configs()
        if not configs:
            return  # 没有配置任何 MCP 服务器

        timeout = 15.0  # 每个操作的超时时间（秒）

        # 逐个连接每个配置的 MCP 服务器
        for name, cfg in configs.items():
            conn = McpConnection(
                name,
                cfg["command"],
                cfg.get("args"),
                cfg.get("env"),
            )
            try:
                await conn.connect()  # 启动子进程
                await asyncio.wait_for(conn.initialize(), timeout=timeout)  # MCP 握手
                server_tools = await asyncio.wait_for(conn.list_tools(), timeout=timeout)  # 发现工具
                self._connections[name] = conn
                self._tools.extend(server_tools)
                print(f"[mcp] Connected to '{name}' — {len(server_tools)} tools", flush=True)
            except Exception as e:
                print(f"[mcp] Failed to connect to '{name}': {e}", flush=True)
                conn.close()  # 连接失败时清理资源

    def get_tool_definitions(self) -> list[dict]:
        """返回 Anthropic API 格式的工具定义列表（带 mcp__ 前缀）

        每个工具名称格式为 "mcp__<serverName>__<toolName>"，确保全局唯一。
        返回格式与 Anthropic API 的 tool 定义兼容，可直接传给模型。
        """
        return [
            {
                # 三段式前缀命名：mcp__服务器名__工具名
                "name": f"mcp__{t['serverName']}__{t['name']}",
                "description": t.get("description") or f"MCP tool {t['name']} from {t['serverName']}",
                "input_schema": t.get("inputSchema") or {"type": "object", "properties": {}},
            }
            for t in self._tools
        ]

    def is_mcp_tool(self, name: str) -> bool:
        """判断工具名称是否为 MCP 工具（以 "mcp__" 前缀开头）"""
        return name.startswith("mcp__")

    async def call_tool(self, prefixed_name: str, args: dict) -> str:
        """将带前缀的工具调用路由到正确的服务器

        解析 "mcp__<serverName>__<toolName>" 格式的工具名，
        提取服务器名称和工具名称，转发到对应的 McpConnection 实例。
        注意：工具名本身可能包含 "__"，所以使用 split("__", 2) 并合并剩余部分。
        """
        parts = prefixed_name.split("__")
        if len(parts) < 3:
            raise ValueError(f"Invalid MCP tool name: {prefixed_name}")  # 格式不合法
        server_name = parts[1]
        tool_name = "__".join(parts[2:])  # 工具名可能包含 "__"，合并剩余部分
        conn = self._connections.get(server_name)
        if not conn:
            raise RuntimeError(f"MCP server '{server_name}' not connected")  # 服务器未连接
        return await conn.call_tool(tool_name, args)

    async def disconnect_all(self) -> None:
        """断开所有 MCP 服务器连接并清理状态

        逐个关闭每个连接（终止子进程、取消读取任务、拒绝待响应请求），
        然后清空连接字典和工具列表，重置连接标志。
        """
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()
        self._tools.clear()
        self._connected = False  # 允许下次重新连接

    # ─── 配置加载 ──────────────────────────────────────

    def _load_configs(self) -> dict[str, dict]:
        """从多个配置源加载并合并 MCP 服务器配置

        加载顺序（后者覆盖前者）：
          1. 全局配置：~/.claude/settings.json
          2. 项目配置：.claude/settings.json（当前工作目录）
          3. MCP 约定配置：.mcp.json（Claude Code 约定格式）
        """
        merged: dict[str, dict] = {}

        # 1. 全局配置：~/.claude/settings.json
        global_path = Path.home() / ".claude" / "settings.json"
        self._merge_config_file(global_path, merged)

        # 2. 项目配置：当前工作目录下的 .claude/settings.json
        project_path = Path.cwd() / ".claude" / "settings.json"
        self._merge_config_file(project_path, merged)

        # 3. Claude Code 约定的 .mcp.json 配置文件
        mcp_json_path = Path.cwd() / ".mcp.json"
        self._merge_config_file(mcp_json_path, merged)

        return merged

    def _merge_config_file(self, path: Path, target: dict[str, dict]) -> None:
        """从单个配置文件读取并合并 MCP 服务器配置到目标字典

        支持两种配置格式：
          - 标准格式：{ "mcpServers": { "name": { ... } } }
          - 简写格式：{ "name": { "command": "..." } }（直接以服务器名为顶级键）
        仅接受包含 "command" 字段的字典配置，其他格式会被跳过。
        格式错误的文件会被静默跳过，不阻断启动流程。
        """
        if not path.exists():
            return  # 文件不存在，跳过
        try:
            raw = json.loads(path.read_text())
            # 支持两种格式：带 "mcpServers" 包装的和直接以名称为键的
            servers = raw.get("mcpServers", raw)
            for name, config in servers.items():
                # 仅接受包含 "command" 字段的有效服务器配置
                if isinstance(config, dict) and "command" in config:
                    target[name] = config  # 后加载的配置会覆盖先加载的（同名时）
        except Exception:
            pass  # 格式错误的配置文件静默跳过
