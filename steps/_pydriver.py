"""Drive one Python step against the in-process Python mock.

Usage: _pydriver.py <pyStepDir> <scenarioPath> <logPath> <workdir>

Sets up a temp workspace, starts the mock in-thread (same process, so loopback
works everywhere), points the real Anthropic SDK at it via env, and runs the
scenario. A scenario is either chat mode ({prompt}) — import the Agent and call
.chat() — or CLI mode ({runs: [{argv}, ...]}) — import the CLI entry and call
main(argv) once per run, sharing one mock and workdir (so session save/resume
works). Writes the mock event log for the test harness to assert on.
"""

# 模块级中文注释:
# 本文件是 Python 步骤的测试驱动器，用于在进程内运行的 Python mock 服务器上执行测试场景。
# 功能概述:
#   1. 设置临时工作空间
#   2. 在当前线程中启动 mock 服务器（同一进程内，确保回环地址可用）
#   3. 通过环境变量将 Anthropic SDK 指向 mock 服务器
#   4. 执行测试场景（支持两种模式: 聊天模式和 CLI 模式）
#   5. 写入 mock 事件日志，供测试断言使用
#
# 使用方式: python _pydriver.py <pyStepDir> <scenarioPath> <logPath> <workdir>
#   - pyStepDir: Python 步骤所在的目录路径
#   - scenarioPath: 测试场景 JSON 文件路径
#   - logPath: mock 事件日志的输出路径
#   - workdir: 临时工作目录路径

import importlib.util  # 用于动态加载模块（不需要模块在 sys.path 中）
import json  # 用于解析场景配置文件
import os  # 用于文件系统操作（路径拼接、目录创建、工作目录切换）
import sys  # 用于命令行参数获取和 sys.path 操作

# 获取当前文件所在目录的绝对路径
HERE = os.path.dirname(os.path.abspath(__file__))
# 将当前目录加入 Python 搜索路径，以便导入同目录下的模块
sys.path.insert(0, HERE)
# 从 mock_anthropic 模块导入 start_mock 函数，用于启动 mock 服务器
# noqa: E402 抑制 "module level import not at top of file" 的 lint 警告
from mock_anthropic import start_mock  # noqa: E402


def main() -> None:
    """主函数: 驱动整个测试流程。

    流程:
      1. 解析命令行参数，获取步骤目录、场景文件路径、日志路径和工作目录
      2. 加载场景配置 JSON 文件
      3. 在工作目录中创建场景所需的预置文件
      4. 启动 mock 服务器并配置环境变量
      5. 根据场景类型执行对应的测试逻辑（CLI 模式或聊天模式）
      6. 关闭 mock 服务器

    参数: 无（通过 sys.argv 获取命令行参数）
    返回值: 无
    """
    # 从命令行参数中提取四个必需的路径参数
    py_step_dir, scenario_path, log_path, workdir = sys.argv[1:5]
    # 从 JSON 文件加载场景配置
    scenario = json.load(open(scenario_path))

    # 在工作目录中创建场景所需的预置文件
    os.makedirs(workdir, exist_ok=True)
    # 遍历场景中 setup.files 字典，将每个文件写入工作目录
    for name, content in (scenario.get("setup", {}).get("files", {})).items():
        p = os.path.join(workdir, name)
        # 确保文件的父目录存在
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w") as f:
            f.write(content)
    # 切换到工作目录，确保后续操作在此目录下进行
    os.chdir(workdir)

    # 启动 mock 服务器，获取服务器 URL 和关闭函数
    url, close = start_mock(scenario, log_path)
    # 设置环境变量，将 Anthropic SDK 指向本地 mock 服务器
    os.environ["ANTHROPIC_BASE_URL"] = url
    # 设置虚拟的 API 密钥（测试环境不需要真实密钥）
    os.environ["ANTHROPIC_API_KEY"] = "test"

    # 将步骤目录加入 Python 搜索路径，以便导入步骤中的模块
    sys.path.insert(0, py_step_dir)

    if scenario.get("runs"):
        # CLI 模式: 场景配置中包含 "runs" 列表，每个 run 对应一次命令行调用
        # 加载 __main__.py 模块，但不使用 __main__ 作为模块名
        # 以避免 if __name__ == "__main__" 守卫自动执行
        # 使用 "stepcli" 作为模块名来加载
        spec = importlib.util.spec_from_file_location("stepcli", os.path.join(py_step_dir, "__main__.py"))
        cli = importlib.util.module_from_spec(spec)
        # 执行模块代码，完成模块初始化
        spec.loader.exec_module(cli)
        # 遍历场景中的每个运行配置，依次调用 CLI 的 main 函数
        # 共享同一个 mock 和 workdir，这样 session 的保存/恢复功能也能正常工作
        for run in scenario["runs"]:
            cli.main(run["argv"])
    else:
        # 聊天模式: 场景配置中包含 "prompt" 字段，模拟用户与 Agent 的对话
        # 导入步骤目录下的 agent 模块
        import agent  # the step's agent.py
        # 创建 Agent 实例并调用 chat 方法，传入场景中的 prompt
        agent.Agent().chat(scenario["prompt"])

    # 关闭 mock 服务器，清理资源
    close()


# 程序入口点
if __name__ == "__main__":
    main()
