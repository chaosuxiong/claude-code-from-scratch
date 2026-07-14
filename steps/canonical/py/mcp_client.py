import json  # 用于构造和解析 JSON-RPC 消息
import subprocess  # 用于启动 MCP 服务器子进程
import threading  # 用于线程锁，保证请求的原子性

# A minimal MCP client: spawn the server as a subprocess and speak
# line-delimited JSON-RPC over its stdio — initialize, then discover its tools,
# then call them. Real MCP has more (multiple transports, auth); the stdio
# handshake is the essence, and it's how you plug external tools into the agent
# without changing its code.

# 模块说明：这是一个最小化的 MCP（Model Context Protocol）客户端实现。
# 功能概述：
#   - 将 MCP 服务器作为子进程启动，通过标准输入/输出（stdio）进行通信
#   - 使用 JSON-RPC 2.0 协议进行行分隔的请求/响应交互
#   - 支持初始化握手、工具发现和工具调用三大核心流程
#   - 真正的 MCP 协议还包含多种传输方式和认证机制，但 stdio 握手是其核心本质
#   - 这种方式允许将外部工具无缝集成到 Agent 中，而无需修改 Agent 的代码


#region mcp
# MCP 连接类：封装与 MCP 服务器子进程的所有通信逻辑
class McpConnection:
    # 初始化 MCP 连接
    # 参数：
    #   command (str): MCP 服务器的可执行命令，例如 "python" 或 "node"
    #   args (list): 传递给服务器命令的参数列表
    # 属性：
    #   self.proc: 子进程对象，通过 Popen 启动，使用管道进行 stdin/stdout 通信
    #   self._id: JSON-RPC 请求的自增 ID 计数器，用于匹配请求和响应
    #   self._lock: 线程锁，确保多线程环境下请求的原子性和 ID 的唯一性
    #   self.tools: 存储服务器提供的工具列表（在 connect 后填充）
    def __init__(self, command, args):
        self.proc = subprocess.Popen([command, *args], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        self._id = 0
        self._lock = threading.Lock()

    # 发送 JSON-RPC 请求并等待响应（同步阻塞）
    # 参数：
    #   method (str): 要调用的远程方法名，例如 "initialize"、"tools/list"
    #   params (dict, 可选): 方法参数字典，默认为空字典
    # 返回值：
    #   dict: 服务器返回的完整 JSON-RPC 响应消息；若进程无输出则返回空字典
    # 实现细节：
    #   - 使用线程锁保证同一时间只有一个请求在进行（防止 ID 冲突和响应错乱）
    #   - 自增请求 ID，写入 JSON-RPC 2.0 格式的请求到 stdin
    #   - 循环读取 stdout 直到收到与当前请求 ID 匹配的响应（跳过通知等无关消息）
    def _request(self, method, params=None):
        with self._lock:
            self._id += 1
            rid = self._id
            self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}) + "\n")
            self.proc.stdin.flush()
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    return {}
                msg = json.loads(line)
                if msg.get("id") == rid:
                    return msg

    # 发送 JSON-RPC 通知（无需等待响应）
    # 参数：
    #   method (str): 通知方法名，例如 "notifications/initialized"
    # 说明：
    #   - 通知与请求的区别在于没有 "id" 字段，服务器不会返回响应
    #   - 此方法不使用线程锁，因为通知不需要匹配响应
    def _notify(self, method):
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
        self.proc.stdin.flush()

    # 建立与 MCP 服务器的完整连接（执行初始化握手流程）
    # 返回值：
    #   McpConnection: 返回自身（self），支持链式调用
    # 初始化流程：
    #   1. 发送 "initialize" 请求：声明协议版本、客户端能力和客户端信息
    #   2. 发送 "notifications/initialized" 通知：告知服务器初始化完成
    #   3. 发送 "tools/list" 请求：获取服务器提供的所有可用工具列表
    #   4. 将工具列表标准化后存储到 self.tools 中
    def connect(self):
        # 步骤 1: 发送 initialize 请求，声明协议版本和客户端信息
        self._request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "mini-claude", "version": "1.0"}})
        # 步骤 2: 发送 initialized 通知，告知服务器客户端已准备就绪
        self._notify("notifications/initialized")
        # 步骤 3: 请求服务器列出所有可用工具
        listed = self._request("tools/list")
        # 步骤 4: 将工具信息标准化为统一格式并存储
        self.tools = [{"name": t["name"], "description": t.get("description", ""), "input_schema": t.get("inputSchema")}
                      for t in listed.get("result", {}).get("tools", [])]
        return self  # 支持链式调用：connect_mcp(cmd, args).call_tool(...)

    # 调用服务器上的指定工具
    # 参数：
    #   name (str): 工具名称，需与 self.tools 中的某个工具名匹配
    #   args (dict): 传递给工具的参数字典
    # 返回值：
    #   str: 工具执行结果的文本内容
    #       - 优先提取结果中 type 为 "text" 的内容并拼接返回
    #       - 若无文本内容，则返回整个 result 或 error 的 JSON 字符串
    def call_tool(self, name, args):
        # 发送 tools/call 请求，传递工具名称和参数
        r = self._request("tools/call", {"name": name, "arguments": args})
        # 从响应中提取 content 数组
        content = r.get("result", {}).get("content", [])
        # 拼接所有 type 为 "text" 的内容片段
        text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
        # 如果没有文本内容，则将整个 result 或 error 转为 JSON 字符串返回
        return text or json.dumps(r.get("result") or r.get("error"))

    # 关闭 MCP 连接，终止子进程
    # 说明：调用 terminate() 向子进程发送 SIGTERM 信号，请求其优雅退出
    def close(self):
        self.proc.terminate()


# 便捷函数：一步完成 MCP 连接的创建和初始化
# 参数：
#   command (str): MCP 服务器的可执行命令
#   args (list): 传递给服务器命令的参数列表
# 返回值：
#   McpConnection: 已完成初始化握手的 MCP 连接对象，可直接用于工具调用
def connect_mcp(command, args):
    return McpConnection(command, args).connect()
#endregion
