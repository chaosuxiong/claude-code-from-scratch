"""Conformance selftest for the Python mock: the real Python SDK must work
against it for create, streaming (text chunks + tool input), usage, scenario
exhaustion (extra call -> error), and a malformed request. Same-process, which
is the only loopback shape that works on the dev host. The Node mock has its own
selftest (mock-anthropic.selftest.mjs); test.mjs exercises both end-to-end.
"""

# =============================================================================
# 模块说明：
# 本文件是 Python 模拟 Anthropic API 服务端的一致性自测试脚本。
# 它使用真实的 Python Anthropic SDK 作为客户端，连接到本地启动的 mock 服务，
# 验证以下核心功能是否正常工作：
#   1. 基本的消息创建（create）请求，并返回 tool_use 类型的响应及 usage 统计
#   2. 流式（streaming）请求，能正确累积文本分块并返回最终消息
#   3. 流式请求中 tool input 参数的正确往返传递
#   4. 场景耗尽（exhaustion）时，多余的请求应抛出 APIError 而非静默成功
# 整个测试在同一进程内完成（same-process loopback），无需外部网络。
# =============================================================================

import os
import sys

# 获取当前脚本所在目录的绝对路径，并将其加入 Python 模块搜索路径的最前面
# 这样可以确保后续 import 能找到同目录下的 mock_anthropic 模块
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 清除可能存在的代理环境变量，确保 mock 客户端直连本地服务，不走代理
os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None); os.environ.pop("all_proxy", None)

import anthropic  # noqa: E402  -- 导入 Anthropic 官方 Python SDK
from mock_anthropic import start_mock  # noqa: E402  -- 导入本地 mock 服务启动函数

# 定义测试场景（scenario）：包含三个轮次（turns），对应本自测试中的三次独立 API 调用
# （一次 create、两次 stream）；mock 服务端会按轮次顺序依次返回预设响应。
# Three turns because this selftest makes three independent calls against one
# mock (create, stream, stream); the mock counts requests per track.
scenario = {"id": "selftest", "turns": [
    {"tools": [{"name": "read_file", "input": {"file_path": "x.txt"}}]},  # 第一轮：返回 tool_use 调用
    {"text": "all done reading the file"},                                  # 第二轮：返回纯文本响应
    {"tools": [{"name": "read_file", "input": {"file_path": "x.txt"}}]},  # 第三轮：再次返回 tool_use 调用
]}

# 用于收集所有失败的测试项名称，最终统一报告
fails = []


def check(name, cond):
    """断言检查函数：打印测试结果并记录失败项。

    参数:
        name (str): 测试项的名称/描述
        cond (bool): 测试条件，True 表示通过，False 表示失败

    返回值:
        无。副作用：打印测试结果到标准输出，失败时将测试名追加到 fails 列表。
    """
    print(f"{'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        fails.append(name)


# 启动本地 mock 服务器，获取服务地址和关闭函数
# url: mock 服务的 HTTP 地址
# close: 用于在测试结束后关闭 mock 服务的回调函数
url, close = start_mock(scenario)

# 创建 Anthropic SDK 客户端实例，连接到本地 mock 服务
# api_key 使用任意测试值，base_url 指向 mock 服务地址
c = anthropic.Anthropic(api_key="test", base_url=url, timeout=10, max_retries=0)

# 定义测试用的工具列表，包含一个名为 "read_file" 的工具
tools = [{"name": "read_file", "description": "d", "input_schema": {"type": "object", "properties": {}, "required": []}}]

# ===================== 测试 1：基本创建请求 =====================
# create -> tool_use, with usage
# 发送同步 create 请求，验证返回的 stop_reason 为 "tool_use"，
# 且 content 中包含对 "read_file" 工具的调用，同时 usage 信息存在
m0 = c.messages.create(model="mock", max_tokens=50, tools=tools, messages=[{"role": "user", "content": "hi"}])
check("create -> tool_use", m0.stop_reason == "tool_use" and any(b.type == "tool_use" and b.name == "read_file" for b in m0.content))
check("usage present", m0.usage.output_tokens > 0)

# ===================== 测试 2：流式请求 - 文本分块累积 =====================
# streaming -> text chunks accumulate + final message
# 构建包含工具执行结果的多轮对话消息历史
msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": m0.content},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_mock_0_0", "content": "data"}]}]
chunks = 0  # 计数器：统计收到的文本分块数量

# 使用流式接口发送请求，逐块读取文本流
with c.messages.stream(model="mock", max_tokens=50, tools=tools, messages=msgs) as s:
    for _ in s.text_stream:  # 遍历流式文本分块
        chunks += 1
    fin = s.get_final_message()  # 获取流式传输完成后的最终完整消息

# 验证最终消息的停止原因为 "end_turn"（正常结束）
check("stream -> final end_turn", fin.stop_reason == "end_turn")
# 验证文本至少被分成一个分块，且最终消息中包含 "done" 文本
check("stream -> text chunked", chunks >= 1 and "done" in "".join(b.text for b in fin.content if b.type == "text"))

# ===================== 测试 3：流式请求 - tool input 往返传递 =====================
# streaming tool input round-trips
# 验证流式响应中 tool_use 块的 input 参数能被正确传递（与请求中的工具参数一致）
with c.messages.stream(model="mock", max_tokens=50, tools=tools, messages=[{"role": "user", "content": "hi"}]) as s:
    for _ in s.text_stream:
        pass  # 仅消费流以触发完整响应解析，不需要对分块做额外处理
    fin2 = s.get_final_message()
check("stream -> tool input intact", any(b.type == "tool_use" and b.input == {"file_path": "x.txt"} for b in fin2.content))

# ===================== 测试 4：场景耗尽时的错误处理 =====================
# exhaustion: a third assistant turn was never scripted -> error, not silent success
# 构造一个包含多余 assistant 轮次的消息列表，超出 mock 场景预设的三轮。
# 预期行为：mock 服务端应抛出 APIError，而不是静默返回成功响应。
try:
    over = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "b"}]}]
    c.messages.create(model="mock", max_tokens=10, messages=over)
    check("exhaustion -> error", False)  # 如果没抛异常，说明测试失败
except anthropic.APIError:
    check("exhaustion -> error", True)   # 捕获到 APIError，符合预期

# ===================== 清理与结果报告 =====================
close()  # 关闭 mock 服务器

# 打印最终测试结果：如果有失败项则报告失败，否则报告通过
print("\n" + ("SELFTEST FAILED: " + ", ".join(fails) if fails else "SELFTEST PASSED"))

# 退出码：有失败则返回 1（非零），全部通过返回 0
sys.exit(1 if fails else 0)
