"""Golden-fixture tests for the autonomy pure functions, using the stdlib
unittest runner (zero extra deps). Reads test/fixtures/autonomy-golden.json --
the SAME file the TS suite reads -- so /goal, /loop, and Auto Mode stay in sync
across the two language mirrors. autonomy.py imports only stdlib, so this runs on
a plain python3 without the anthropic/openai deps.

Run with `python3 python/tests/test_autonomy.py` (or `npm run test:py`)."""

# ============================================================================
# 模块说明:
# 本文件是 autonomy（自治模式）纯函数的"黄金数据"单元测试模块。
# 它使用 Python 标准库 unittest 运行，无需额外依赖。
# 测试数据来自 test/fixtures/autonomy-golden.json，该文件同时被
# TypeScript 测试套件和 Python 测试套件共享，以确保 /goal、/loop
# 和 Auto Mode 在两种语言实现中保持行为一致。
# autonomy.py 仅使用标准库，因此可在没有 anthropic/openai 依赖的
# 普通 python3 环境下运行。
# ============================================================================

import json      # JSON 解析模块，用于加载测试用的黄金数据文件
import sys       # 系统模块，用于修改模块搜索路径
import unittest  # Python 标准库单元测试框架
from pathlib import Path  # 路径处理模块，用于构建文件路径

# repo layout: python/tests/test_autonomy.py -> python/ (add to path) -> repo/
# 项目目录结构说明:
# python/tests/test_autonomy.py  -> python/ (添加到搜索路径) -> repo/ (仓库根目录)
_HERE = Path(__file__).resolve()       # 当前测试文件的绝对路径
_PYTHON_DIR = _HERE.parent.parent      # python/ 目录（当前文件的上两级）
_REPO = _PYTHON_DIR.parent             # 仓库根目录（python/ 的上一级）
sys.path.insert(0, str(_PYTHON_DIR))   # 将 python/ 目录添加到模块搜索路径首位，以便导入 mini_claude 模块

from mini_claude import autonomy as a  # noqa: E402  # 导入 autonomy 模块并使用别名 a

# 加载黄金测试数据文件（JSON 格式），该文件包含所有测试用例的输入和期望输出
GOLDEN = json.loads((_REPO / "test" / "fixtures" / "autonomy-golden.json").read_text(encoding="utf-8"))


class TestAutonomyGolden(unittest.TestCase):
    """
    自治模式纯函数的黄金数据测试类。

    本类包含多个测试方法，每个方法对应 autonomy 模块中的一个纯函数。
    测试数据来自共享的 JSON 黄金数据文件，确保 Python 实现与
    TypeScript 实现的行为完全一致。
    """

    def test_parse_duration_to_seconds(self):
        """
        测试 parse_duration_to_seconds 函数。

        功能：验证将人类可读的时间字符串（如 "5m"、"1h"）解析为秒数的正确性。
        测试逻辑：遍历黄金数据中的所有测试用例，逐一比对函数输出与期望值。
        """
        for c in GOLDEN["parseDurationToSeconds"]:
            # 对每个测试用例，调用函数并断言结果等于期望值
            self.assertEqual(a.parse_duration_to_seconds(c["token"]), c["expected"], msg=f"token={c['token']}")

    def test_clamp_wakeup_delay(self):
        """
        测试 clamp_wakeup_delay 函数。

        功能：验证将唤醒延迟时间限制在合法范围内的正确性。
        该函数确保延迟值不会过小（过于频繁）或过大（过于稀疏）。
        测试逻辑：遍历黄金数据中的所有测试用例，逐一比对。
        """
        for c in GOLDEN["clampWakeupDelay"]:
            # 对每个测试用例，调用函数并断言结果等于期望值
            self.assertEqual(a.clamp_wakeup_delay(c["seconds"]), c["expected"], msg=f"seconds={c['seconds']}")

    def test_is_daily_wording(self):
        """
        测试 is_daily_wording 函数。

        功能：验证判断输入文本是否包含"每日"相关表述的正确性。
        例如检测 "daily"、"every day" 等关键词。
        测试逻辑：遍历黄金数据中的所有测试用例，逐一比对。
        """
        for c in GOLDEN["isDailyWording"]:
            # 对每个测试用例，调用函数并断言结果等于期望值
            self.assertEqual(a.is_daily_wording(c["raw"]), c["expected"], msg=f"raw={c['raw']}")

    def test_project_action_for_classifier(self):
        """
        测试 project_action_for_classifier 函数。

        功能：验证根据工具名称和输入内容推断分类器动作的正确性。
        该函数用于将工具调用映射到对应的分类动作类别。
        测试逻辑：遍历黄金数据中的所有测试用例，逐一比对。
        """
        for c in GOLDEN["projectActionForClassifier"]:
            # 对每个测试用例，调用函数并断言结果等于期望值
            self.assertEqual(a.project_action_for_classifier(c["tool"], c["input"]), c["expected"], msg=f"tool={c['tool']}")

    def test_parse_goal_verdict(self):
        """
        测试 parse_goal_verdict 函数。

        功能：验证解析目标判定结果（goal verdict）的正确性。
        该函数从原始文本中提取目标是否达成的判定结论。
        测试逻辑：遍历黄金数据中的所有测试用例，逐一比对。
        """
        for c in GOLDEN["parseGoalVerdict"]:
            # 对每个测试用例，调用函数并断言结果等于期望值
            self.assertEqual(a.parse_goal_verdict(c["raw"]), c["expected"], msg=f"raw={c['raw']}")

    def test_parse_block_verdict(self):
        """
        测试 parse_block_verdict 函数。

        功能：验证解析阻塞判定结果（block verdict）的正确性。
        该函数从原始文本中提取阻塞/卡住状态的判定结论。
        测试逻辑：遍历黄金数据中的所有测试用例，逐一比对。
        """
        for c in GOLDEN["parseBlockVerdict"]:
            # 对每个测试用例，调用函数并断言结果等于期望值
            self.assertEqual(a.parse_block_verdict(c["raw"]), c["expected"], msg=f"raw={c['raw']}")

    def test_parse_loop_input(self):
        """
        测试 parse_loop_input 函数。

        功能：验证解析循环输入（loop input）的正确性。
        该函数将用户输入的循环指令解析为结构化数据，包括模式（mode）、
        提示词（prompt）以及可选的间隔时间（interval_seconds）和间隔标签（interval_label）。
        测试逻辑：遍历黄金数据中的所有测试用例，分别验证各个字段。
        """
        for c in GOLDEN["parseLoopInput"]:
            r = a.parse_loop_input(c["raw"])  # 调用被测函数，获取实际结果
            e = c["expected"]                  # 获取期望结果
            # 如果期望结果包含 error 字段，则验证函数返回了对应的错误信息
            if "error" in e:
                self.assertEqual(r.get("error"), e["error"], msg=f"raw={c['raw']}")
                continue
            # 验证解析出的模式（mode）字段
            self.assertEqual(r["mode"], e["mode"], msg=f"raw={c['raw']} mode")
            # 验证解析出的提示词（prompt）字段
            self.assertEqual(r["prompt"], e["prompt"], msg=f"raw={c['raw']} prompt")
            # 如果模式为 "interval"（定时间隔模式），还需验证间隔秒数和标签
            if e["mode"] == "interval":
                self.assertEqual(r["interval_seconds"], e["seconds"], msg=f"raw={c['raw']} seconds")
                self.assertEqual(r["interval_label"], e["label"], msg=f"raw={c['raw']} label")

    def test_build_classifier_transcript(self):
        """
        测试 build_classifier_transcript 函数。

        功能：验证构建分类器转录文本（classifier transcript）的正确性。
        该函数将历史记录和待处理的工具调用组合成一段转录文本，
        供分类器分析当前的自治操作状态。
        测试逻辑：遍历黄金数据中的所有测试用例，逐一比对。
        """
        for c in GOLDEN["buildClassifierTranscript"]:
            # 调用函数，传入历史记录和待处理的工具调用（包含工具名和输入）
            out = a.build_classifier_transcript(c["history"], {"tool_name": c["pending"]["tool"], "input": c["pending"]["input"]})
            self.assertEqual(out, c["expected"])


# 当直接运行此脚本时（而非作为模块导入），执行所有测试
# verbosity=2 表示输出详细的测试信息（每个测试方法的名称和结果）
if __name__ == "__main__":
    unittest.main(verbosity=2)
