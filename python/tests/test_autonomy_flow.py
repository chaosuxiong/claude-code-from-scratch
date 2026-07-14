"""Control-flow tests for the two-stage Auto Mode classifier (Python mirror of
test/autonomy-flow.test.ts). Stubs the per-stage classifier query (no network)
and drives Agent._classify_tool_call directly, covering: stage-1 gate,
stage-1→stage-2 escalation, denial counting, and fail-closed parsing.

Agent imports the anthropic/openai SDKs, so this module skips cleanly when those
deps aren't installed (the pure-function suite in test_autonomy.py has no such
dependency and always runs). Run with `python3 -B python/tests/test_autonomy_flow.py`.

模块级说明：
本文件是两阶段自动模式（Auto Mode）分类器的控制流测试模块。
它是 test/autonomy-flow.test.ts 的 Python 镜像版本。
通过桩（stub）替换分类器查询（无需网络），直接驱动 Agent._classify_tool_call 方法，
覆盖以下场景：
  - 第一阶段（stage-1）的门控逻辑（gate）
  - 第一阶段到第二阶段（stage-1 -> stage-2）的升级/降级流程
  - 拒绝计数（denial counting）
  - 解析失败时的关闭策略（fail-closed parsing）

由于 Agent 依赖 anthropic/openai SDK，当这些依赖未安装时，本模块会自动跳过
（相比之下，test_autonomy.py 中的纯函数测试不依赖这些库，始终可以运行）。
运行方式：python3 -B python/tests/test_autonomy_flow.py
"""
import sys
import unittest
from pathlib import Path

# 计算 python 目录的绝对路径，用于后续模块导入
_PYTHON_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PYTHON_DIR))

# 尝试导入 Agent 类；如果依赖未安装则标记 HAVE_DEPS 为 False，测试将被跳过
try:
    from mini_claude.agent import Agent
    HAVE_DEPS = True  # 标记依赖是否可用
except Exception:
    HAVE_DEPS = False  # 依赖不可用，跳过需要 Agent 的测试


def _mk_agent(responses):
    """An Agent whose classifier query returns canned per-stage responses.
    Returns (agent, calls) where calls['n'] counts stage queries made.

    创建一个使用预制响应的 Agent 实例（工厂函数）。
    该函数将 Agent 的分类器查询替换为桩函数，按顺序返回预设的响应内容，
    从而在无需网络请求的情况下测试分类器的控制流。

    参数：
        responses (list[str]): 预制的分类器响应列表，每个元素对应一个阶段的返回值。
            例如 ["<block>no</block>"] 表示第一阶段允许通过。

    返回值：
        tuple[Agent, dict]: 返回一个元组：
            - agent: 已替换分类器查询的 Agent 实例
            - calls: 一个字典，其中 calls['n'] 记录了分类器查询被调用的次数
    """
    # 创建一个处于 auto 权限模式的 Agent 实例（使用测试用 API key）
    agent = Agent(api_key="test-key", permission_mode="auto")
    # 调用计数器，用于验证分类器查询被调用的次数
    calls = {"n": 0}

    async def _stub(system, user, max_tokens):
        """桩函数：模拟分类器查询，按顺序返回预制响应。

        参数：
            system: 系统提示词（在此测试中未使用）
            user: 用户提示词（在此测试中未使用）
            max_tokens: 最大生成 token 数（在此测试中未使用）

        返回值：
            str: 预制的分类器响应字符串；如果预制响应用尽，返回默认的阻止响应
        """
        i = calls["n"]  # 获取当前调用的索引
        calls["n"] += 1  # 递增调用计数
        if i < len(responses):
            return responses[i]  # 返回对应的预制响应
        # 如果预制响应用尽，返回一个默认的阻止响应（fail-closed 策略）
        return "<block>yes</block><reason>[fallback] ran out of canned responses</reason>"

    # 将桩函数替换到 Agent 实例上，替代真实的网络查询
    agent._run_classifier_query = _stub
    return agent, calls


@unittest.skipUnless(HAVE_DEPS, "anthropic/openai not installed")
class TestTwoStageFlow(unittest.IsolatedAsyncioTestCase):
    """两阶段自动模式分类器的控制流测试类。

    测试分类器的两阶段工作流程：
    - 第一阶段（stage-1）：快速判断工具调用是否允许或需要进一步审查
    - 第二阶段（stage-2）：对第一阶段标记为阻止的调用进行二次确认

    本类覆盖以下核心场景：
    1. 第一阶段直接允许的场景
    2. 第一阶段阻止但第二阶段允许的升级场景
    3. 两个阶段都阻止时的拒绝计数逻辑
    4. 第二阶段返回无法解析内容时的 fail-closed 行为
    5. 快速路径（只读工具）跳过分类器的场景
    """

    async def test_stage1_allow_one_call(self):
        """测试第一阶段直接允许的场景。

        当第一阶段分类器返回 <block>no</block>（即不阻止）时，
        工具调用应被直接允许，且第二阶段分类器不应被调用。

        验证点：
        - 返回的 action 应为 "allow"
        - 分类器查询应仅被调用 1 次（只经过第一阶段）
        """
        # 设置第一阶段返回"不阻止"的预制响应
        agent, calls = _mk_agent(["<block>no</block>"])
        # 调用分类方法，模拟执行 run_shell 工具
        r = await agent._classify_tool_call("run_shell", {"command": "echo hi"})
        # 验证结果为允许
        self.assertEqual(r["action"], "allow")
        # 验证分类器只被调用了一次（第二阶段未执行）
        self.assertEqual(calls["n"], 1, "stage 2 must not run when stage 1 allows")

    async def test_stage1_block_then_stage2_allow(self):
        """测试第一阶段阻止但第二阶段允许的升级场景。

        当第一阶段分类器认为需要阻止（如检测到 Git Push 到默认分支），
        但第二阶段分类器重新评估后允许时，最终结果应为允许。

        验证点：
        - 返回的 action 应为 "allow"
        - 分类器查询应被调用 2 次（经过两个阶段）
        """
        # 第一阶段返回"阻止"，第二阶段返回"不阻止"
        agent, calls = _mk_agent([
            "<block>yes</block><reason>[Git Push to Default Branch] main</reason>",
            "<block>no</block>",
        ])
        r = await agent._classify_tool_call("run_shell", {"command": "echo hi"})
        # 验证最终结果为允许（第二阶段覆盖了第一阶段的阻止决定）
        self.assertEqual(r["action"], "allow")
        # 验证分类器被调用了两次（两个阶段都执行了）
        self.assertEqual(calls["n"], 2, "stage 2 must run after a stage 1 block")

    async def test_stage2_block_denial_counted_once(self):
        """测试两个阶段都阻止时的拒绝计数逻辑。

        当第一阶段和第二阶段分类器都认为应阻止时，工具调用应被拒绝，
        且拒绝计数器应只增加 1（而不是 2，避免重复计数）。

        验证点：
        - 返回的 action 应为 "deny"
        - 返回的 message 应包含 "[Auto Mode]" 标记
        - 分类器查询应被调用 2 次
        - 连续拒绝计数器 auto_consecutive_denials 应为 1
        """
        # 两个阶段都返回"阻止"
        agent, calls = _mk_agent([
            "<block>yes</block><reason>[Git Push to Default Branch] a</reason>",
            "<block>yes</block><reason>[Git Push to Default Branch] b</reason>",
        ])
        r = await agent._classify_tool_call("run_shell", {"command": "echo hi"})
        # 验证最终结果为拒绝
        self.assertEqual(r["action"], "deny")
        # 验证拒绝消息包含 Auto Mode 标识
        self.assertIn("[Auto Mode]", r["message"])
        # 验证分类器被调用了两次
        self.assertEqual(calls["n"], 2)
        # 验证拒绝计数器只增加 1（两次阻止只算一次拒绝）
        self.assertEqual(agent.auto_consecutive_denials, 1, "one blocked action → one denial")

    async def test_stage2_unparseable_blocks(self):
        """测试第二阶段返回无法解析内容时的 fail-closed 行为。

        当第二阶段分类器返回的内容无法被解析为有效的判定结果时，
        系统应采用 fail-closed（默认关闭）策略，将操作视为被阻止。

        验证点：
        - 返回的 action 应为 "deny"（无法解析时默认拒绝）
        """
        # 第一阶段返回阻止，第二阶段返回无法解析的垃圾内容
        agent, _ = _mk_agent([
            "<block>yes</block><reason>[X] a</reason>",
            "garbage, not a verdict",
        ])
        r = await agent._classify_tool_call("run_shell", {"command": "echo hi"})
        # 验证即使第二阶段返回无法解析的内容，最终结果仍为拒绝（fail-closed）
        self.assertEqual(r["action"], "deny")

    async def test_fast_path_skips_classifier(self):
        """测试快速路径（只读工具）跳过分类器的场景。

        对于只读类工具（如 read_file），系统应走快速路径直接允许，
        而不需要调用分类器进行安全审查。这是一种性能优化。

        验证点：
        - 返回的 action 应为 "allow"
        - 分类器查询应被调用 0 次（完全跳过分类器）
        """
        # 即使预制了一个"阻止"的响应，也不应该被使用
        agent, calls = _mk_agent(["<block>yes</block><reason>should not be used</reason>"])
        # 调用分类方法，模拟执行只读工具 read_file
        r = await agent._classify_tool_call("read_file", {"file_path": "x"})
        # 验证结果为允许（走快速路径）
        self.assertEqual(r["action"], "allow")
        # 验证分类器未被调用（只读工具跳过了分类器）
        self.assertEqual(calls["n"], 0, "read-only tools must not call the classifier")


# 当直接运行此脚本时，执行所有测试（verbosity=2 显示详细输出）
if __name__ == "__main__":
    unittest.main(verbosity=2)
