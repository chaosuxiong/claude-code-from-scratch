"""Python twin of mock-anthropic.mjs. Speaks the real Anthropic /v1/messages
protocol (create + SSE stream, tool_use, usage, errors) and replays a scripted
scenario. Runs in-thread so a Python step in the same process can reach it
(cross-process loopback is intercepted by the dev host's proxy). Reads the same
steps/scenarios/*.json as the Node mock, so the two cannot behave differently.
"""

# =============================================================================
# 模块说明:
# 本文件是 mock-anthropic.mjs 的 Python 版本，用于模拟 Anthropic API 的 /v1/messages 端点。
# 主要功能:
#   - 支持标准的 Anthropic 消息协议（创建消息 + SSE 流式响应、工具调用、token 用量统计、错误处理）
#   - 回放预设的脚本化场景（scenario），用于测试目的
#   - 在当前线程内运行 HTTP 服务器，使同进程中的 Python 测试步骤可以直接访问
#   - 读取与 Node.js 版本相同的 steps/scenarios/*.json 场景文件，确保两个版本行为一致
#   - 支持多轨道（track）路由：根据系统提示词中的子字符串匹配将请求路由到不同的场景轨道
# =============================================================================

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


def _message_from_turn(turn, model, req_index):
    """Build an Anthropic-style message dict from a scenario turn.
    从场景轮次（turn）构建符合 Anthropic API 格式的消息字典。

    Args / 参数:
        turn (dict): 场景中的一个轮次对象，包含 text（文本内容）、tools（工具调用列表）、usage（token 用量）等字段。
        model (str): 模型名称，用于填充响应中的 model 字段。
        req_index (int): 当前请求的序号，用于生成唯一的消息 ID 和工具调用 ID。

    Returns / 返回值:
        dict: 符合 Anthropic API 规范的消息响应字典，包含 id、type、role、model、content、stop_reason、usage 等字段。
    """
    content = []
    if turn.get("text"):
        content.append({"type": "text", "text": turn["text"]})
    for j, t in enumerate(turn.get("tools", [])):
        content.append({"type": "tool_use", "id": f"toolu_mock_{req_index}_{j}",
                        "name": t["name"], "input": t.get("input", {})})
    # 如果轮次包含工具调用，则 stop_reason 为 "tool_use"，否则为 "end_turn"
    stop_reason = "tool_use" if turn.get("tools") else "end_turn"
    # 如果轮次未指定 usage，则使用默认的 token 估算值
    usage = turn.get("usage", {"input_tokens": 100, "output_tokens": 20})
    return {"id": f"msg_mock_{req_index}", "type": "message", "role": "assistant",
            "model": model, "content": content, "stop_reason": stop_reason,
            "stop_sequence": None, "usage": usage}


def _sse(block_lines):
    """Convert a list of (event_name, data_dict) pairs into an SSE text body.
    将 (事件名, 数据字典) 对的列表转换为 SSE（Server-Sent Events）格式的文本响应体。

    Args / 参数:
        block_lines (list[tuple[str, dict]]): SSE 事件列表，每个元素为 (事件名, 数据字典) 元组。

    Returns / 返回值:
        str: 符合 SSE 协议格式的完整文本，每个事件格式为 "event: <name>\ndata: <json>\n\n"。
    """
    return "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in block_lines)


def _stream_body(msg):
    """Build the full SSE stream body for a message, mimicking the real Anthropic
    streaming protocol (message_start, content_block_start/delta/stop, message_delta, message_stop).
    构建消息的完整 SSE 流式响应体，模拟真实的 Anthropic 流式协议。
    流式事件顺序为: message_start -> [content_block_start -> content_block_delta* -> content_block_stop]* -> message_delta -> message_stop。

    Args / 参数:
        msg (dict): 由 _message_from_turn 构建的消息字典，包含 content 列表和 stop_reason 等字段。

    Returns / 返回值:
        str: 完整的 SSE 流式响应文本，可直接作为 HTTP 响应体发送。
    """
    lines = [("message_start", {"type": "message_start", "message": {**msg, "content": [], "stop_reason": None,
              "usage": {**msg["usage"], "output_tokens": 0}}})]
    for i, block in enumerate(msg["content"]):
        if block["type"] == "text":
            # 文本块：先发送 content_block_start，然后分片发送文本内容（每片 24 字符），最后发送 content_block_stop
            lines.append(("content_block_start", {"type": "content_block_start", "index": i,
                          "content_block": {"type": "text", "text": ""}}))
            text = block["text"]
            for k in range(0, len(text), 24):
                lines.append(("content_block_delta", {"type": "content_block_delta", "index": i,
                              "delta": {"type": "text_delta", "text": text[k:k + 24]}}))
            lines.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
        else:
            # 工具调用块：先发送带工具元信息的 content_block_start，然后一次性发送完整的 input JSON，最后发送 content_block_stop
            lines.append(("content_block_start", {"type": "content_block_start", "index": i,
                          "content_block": {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}}))
            lines.append(("content_block_delta", {"type": "content_block_delta", "index": i,
                          "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}}))
            lines.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    # 发送消息级别的 delta（包含 stop_reason）和 stop 事件
    lines.append(("message_delta", {"type": "message_delta",
                  "delta": {"stop_reason": msg["stop_reason"], "stop_sequence": None},
                  "usage": {"output_tokens": msg["usage"]["output_tokens"]}}))
    lines.append(("message_stop", {"type": "message_stop"}))
    return _sse(lines)


def start_mock(scenario, log_path=None):
    """Start the mock in a daemon thread. Returns (url, close). A scenario is
    flat ({turns}) or multi-track ({tracks: {main:{turns}, compact:{match,turns}}});
    requests route to a track by a substring its `match` finds in the system
    prompt (default "main"), each track with its own counter.

    在守护线程中启动模拟服务器。返回值为 (服务器 URL, 关闭函数)。
    场景(scenario)可以是简单的扁平结构({turns})，也可以是多轨道结构({tracks: {main:{turns}, compact:{match,turns}}})。
    请求根据系统提示词中是否包含各轨道的 match 子字符串来路由(默认路由到 "main" 轨道)，
    每个轨道维护独立的轮次计数器。

    Args:
        scenario (dict): 场景配置字典，包含 turns(轮次列表) 或 tracks(多轨道配置)。
        log_path (str, optional): 日志文件路径，如果提供则将请求/响应日志写入该文件。

    Returns:
        tuple[str, callable]: (服务器 URL, 关闭函数)。URL 形如 "http://127.0.0.1:<port>"，
                              关闭函数调用后会停止服务器。
    """
    tracks = (scenario or {}).get("tracks") or {"main": {"turns": (scenario or {}).get("turns", [])}}
    state = {"req": 0, "counters": {}}  # 全局状态：req 为请求序号，counters 记录每个轨道的当前轮次索引

    def log(obj):
        """将日志对象以 JSON 格式追加写入日志文件。
        Args / 参数:
            obj (dict): 要记录的日志对象。
        """
        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps(obj) + "\n")

    class Handler(BaseHTTPRequestHandler):
        """HTTP 请求处理器，实现 Anthropic /v1/messages API 的模拟端点。
        处理 POST 请求，根据请求内容路由到对应的场景轨道，并返回模拟的 API 响应。
        支持普通 JSON 响应和 SSE 流式响应两种模式。
        """

        def do_POST(self):
            """处理 POST 请求，模拟 Anthropic 的 /v1/messages 端点。
            主要流程:
              1. 验证请求路径
              2. 解析请求体（system 提示词、工具列表、用户消息等）
              3. 根据系统提示词匹配场景轨道
              4. 提取工具调用结果（tool_result）
              5. 查找对应的场景轮次并构建响应
              6. 根据请求是否为流式模式，返回 SSE 流或普通 JSON 响应
            """
            if not self.path.startswith("/v1/messages"):
                self.send_response(404); self.end_headers(); self.wfile.write(b"not found"); return
            body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
            req_index = state["req"]

            # 提取系统提示词文本（可能是字符串或内容块列表）
            sys_text = body.get("system", "")
            if isinstance(sys_text, list):
                sys_text = "".join(b.get("text", "") for b in sys_text)
            tool_names = [t["name"] for t in body.get("tools", [])]
            # 获取第一条用户消息的文本内容
            first_user = next((m for m in body.get("messages", []) if m.get("role") == "user"), None)
            first_user_text = first_user["content"] if first_user and isinstance(first_user.get("content"), str) else ""

            # Route to an aux track by a structured match (system substring +
            # optional firstUser / tools). Check ALL tracks and fail loudly on
            # ambiguity — a request must never match two tracks.
            # 通过结构化匹配将请求路由到辅助轨道（系统提示词子字符串 + 可选的 firstUser/tools 匹配）。
            # 检查所有轨道，如果匹配到多个轨道则报错——一个请求绝不能同时匹配两个轨道。
            def _matches(t):
                """检查请求是否匹配指定轨道的匹配条件。
                Args / 参数:
                    t (dict): 轨道配置，可包含 match（系统提示词子字符串）、firstUserContains（用户消息子字符串）、toolsInclude（必须包含的工具名列表）。
                Returns / 返回值:
                    bool: 如果请求满足所有匹配条件则返回 True。
                """
                return ((not t.get("match") or t["match"] in sys_text)
                        and (not t.get("firstUserContains") or t["firstUserContains"] in first_user_text)
                        and (not t.get("toolsInclude") or all(n in tool_names for n in t["toolsInclude"])))
            hits = [name for name, t in tracks.items() if name != "main" and t.get("match") and _matches(t)]
            if len(hits) > 1:
                # 匹配到多个轨道，返回歧义错误
                log({"type": "ambiguous", "req": req_index, "tracks": hits, "system": sys_text})
                err = json.dumps({"type": "error", "error": {"type": "mock_ambiguous_route",
                      "message": f"request matched multiple tracks: {', '.join(hits)}"}}).encode()
                self.send_response(500); self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(err))); self.end_headers(); self.wfile.write(err); return
            track = hits[0] if len(hits) == 1 else "main"  # 确定最终路由的轨道
            turn_index = state["counters"].get(track, 0)  # 获取该轨道当前的轮次索引
            track_turns = tracks.get(track, {}).get("turns", [])
            turn = track_turns[turn_index] if turn_index < len(track_turns) else None

            # tool_result blocks the agent sent back — proof the tool actually
            # ran, with its real output.
            # 提取代理（agent）发回的 tool_result 块——这些是工具实际执行的证据及其真实输出。
            tool_results = []
            for m in body.get("messages", []):
                if isinstance(m.get("content"), list):
                    for b in m["content"]:
                        if b.get("type") == "tool_result":
                            c = b.get("content")
                            tool_results.append({"tool_use_id": b.get("tool_use_id"),
                                                 "content": c if isinstance(c, str) else json.dumps(c)})
            # 记录请求日志
            log({"type": "request", "req": req_index, "track": track, "turnIndex": turn_index,
                 "system": sys_text, "tools": tool_names,
                 "toolResults": tool_results, "messageCount": len(body.get("messages", [])),
                 "firstUserText": first_user_text, "stream": bool(body.get("stream"))})

            if turn is None:
                # 当前轨道的轮次已用尽，返回耗尽错误
                log({"type": "exhausted", "req": req_index, "track": track, "turnIndex": turn_index})
                err = json.dumps({"type": "error", "error": {"type": "mock_exhausted",
                      "message": f'track "{track}" has no turn {turn_index}'}}).encode()
                self.send_response(500); self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(err))); self.end_headers(); self.wfile.write(err); return
            state["counters"][track] = turn_index + 1  # 推进该轨道的轮次计数器

            msg = _message_from_turn(turn, body.get("model", "mock"), req_index)
            # 记录响应日志
            log({"type": "response", "req": req_index, "stop_reason": msg["stop_reason"],
                 "tool_use": [{"name": b["name"], "input": b["input"]} for b in msg["content"] if b["type"] == "tool_use"]})
            state["req"] += 1  # 推进全局请求序号

            if body.get("stream"):
                # 流式响应模式：构建 SSE 流并发送
                b = _stream_body(msg).encode()
                self.send_response(200); self.send_header("content-type", "text/event-stream")
                self.send_header("content-length", str(len(b))); self.end_headers(); self.wfile.write(b)
            else:
                # 普通 JSON 响应模式：直接序列化消息字典并发送
                b = json.dumps(msg).encode()
                self.send_response(200); self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def log_message(self, *a):
            """覆盖父类方法，禁止 BaseHTTPRequestHandler 默认的请求日志输出到 stderr。
            Args / 参数:
                *a: 父类方法的可变参数（格式化字符串及参数），此处全部忽略。
            """
            pass

    # 在本地回环地址上启动 HTTP 服务器，端口由操作系统自动分配（端口 0）
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]  # 获取实际分配的端口号
    # 在守护线程中启动服务器，确保主程序退出时线程自动终止
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}", srv.shutdown
