"""
模块入口文件 - mini-claude 命令行交互程序

本文件是 mini-claude 项目的主入口点，实现了一个轻量级的 REPL（Read-Eval-Print Loop）
交互式命令行界面。主要功能包括：
- 通过命令行参数支持单次执行模式（one-shot）和交互式循环模式
- 支持会话恢复（--resume）、计划模式（--plan）、自动模式（--auto）和目标追踪（--goal）
- 集成技能解析系统，支持 "/name ..." 格式的技能命令调用
- 管理对话历史的保存与加载
"""

import os
import sys

from agent import Agent  # 导入 Agent 类，核心对话代理
#step >=4
from session import save_session, load_session  # 导入会话保存和加载函数
#endstep
#step >=9
from skills import resolve_skill  # 导入技能解析函数，用于处理 "/name ..." 格式的技能命令
#endstep


# A tiny REPL: read a line, hand it to the agent, repeat. One-shot mode runs a
# single prompt from argv and exits (handy for scripts and testing). Takes argv
# so it can be driven in-process without spawning a shell.
# 主函数 - 程序入口点，实现迷你 REPL 交互循环
# 支持两种运行模式：
#   1. 单次执行模式（one-shot）：通过命令行参数传入提示词，执行后立即退出
#   2. 交互式循环模式：持续读取用户输入并处理，直到用户退出
# 参数:
#   argv (list[str] | None): 命令行参数列表，None 时使用 sys.argv[1:]
# 返回值: None
def main(argv=None) -> None:
    # 如果未提供参数，则从系统命令行参数获取
    if argv is None:
        argv = sys.argv[1:]
    # 检查是否设置了 ANTHROPIC_API_KEY 环境变量，未设置则报错退出
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY (and optionally ANTHROPIC_BASE_URL) first.", file=sys.stderr)
        sys.exit(1)

    # 创建 Agent 实例，用于处理对话逻辑
    agent = Agent()
#step >=4
    # --resume: reload the saved conversation before doing anything else.
    # --resume 参数处理：在执行其他操作之前，重新加载之前保存的对话历史
    resume = "--resume" in argv  # 检查是否包含 --resume 标志
    argv = [a for a in argv if a != "--resume"]  # 从参数列表中移除 --resume 标志
    if resume:
        saved = load_session()  # 加载保存的会话数据
        if saved:
            agent.load_history(saved)  # 将历史记录加载到 Agent 中
            print(f"(resumed {len(saved)} messages)")  # 提示恢复了多少条消息
#endstep
#step >=10
    # --plan: read-only mode. The agent may read and think, but not write or run shell.
    # --plan 参数处理：只读模式，Agent 只能读取和思考，不能写入文件或执行 shell 命令
    if "--plan" in argv:
        agent.set_mode("plan")  # 设置 Agent 为计划模式
        argv = [a for a in argv if a != "--plan"]  # 从参数列表中移除 --plan 标志
        print("(plan mode: read-only)")  # 提示进入只读模式
#endstep
#step >=15
    # --auto: a classifier gates each write instead of asking; --goal pursues a condition.
    # --auto 参数处理：自动模式，使用分类器控制每次写入操作，无需人工确认
    if "--auto" in argv:
        agent.set_mode("auto")  # 设置 Agent 为自动模式
        argv = [a for a in argv if a != "--auto"]  # 从参数列表中移除 --auto 标志
        print("(auto mode: a classifier gates each write)")  # 提示进入自动模式
    # --goal 参数处理：目标追踪模式，Agent 将持续执行直到满足指定条件
    if "--goal" in argv:
        gi = argv.index("--goal")  # 获取 --goal 在参数列表中的索引位置
        condition = argv[gi + 1] if gi + 1 < len(argv) else ""  # 获取目标条件
        agent.pursue_goal(condition, " ".join(argv[gi + 2:]))  # 执行目标追踪，传入条件和剩余参数
        save_session(agent.history())  # 保存当前会话历史
        return  # 目标模式执行完毕后直接返回
#endstep

    # 处理单次执行模式（one-shot）：将剩余参数合并为单个提示词
    one_shot = " ".join(argv).strip()
    if one_shot:
#step >=9
        # "/name ..." runs a skill's prompt template; anything else is a message.
        # 技能解析：以 "/name ..." 开头的输入将被解析为技能命令，其他输入作为普通消息处理
        text = resolve_skill(one_shot) or one_shot  # 尝试解析技能，失败则使用原始文本
#step <=8
        text = one_shot  # 在 step 8 及以下版本中，直接使用原始文本（无技能解析功能）
#endstep
        agent.chat(text)  # 将文本发送给 Agent 进行对话处理
#step >=4
        save_session(agent.history())  # 保存对话历史到会话文件
#endstep
        return  # 单次执行模式完成后退出

    # 交互式循环模式：持续读取用户输入并处理
    print("mini-claude — type a message, or 'exit' to quit.\n")
    while True:
        try:
            line = input("you: ").strip()  # 读取用户输入并去除首尾空白
        except (EOFError, KeyboardInterrupt):
            # 捕获 EOF（Ctrl+D）和键盘中断（Ctrl+C），优雅退出
            print()
            break
        if line in ("exit", "quit"):
            break  # 用户输入 exit 或 quit 时退出循环
#step >=4
        # /clear 命令：清空对话历史
        if line == "/clear":
            agent.clear_history()  # 清除 Agent 中的对话历史
            save_session(agent.history())  # 保存清空后的历史（实际为空）
            print("(history cleared)")  # 提示历史已清空
            continue  # 跳过后续处理，继续下一轮循环
#endstep
#step >=9
        # 处理用户输入：尝试解析技能命令，否则作为普通消息发送
        if line:
            agent.chat(resolve_skill(line) or line)  # 解析技能或直接发送消息
#step <=8
        # 在 step 8 及以下版本中，直接将用户输入作为消息发送（无技能解析功能）
        if line:
            agent.chat(line)
#endstep
#step >=4
        # 每次对话后保存会话历史，确保数据持久化
        if line:
            save_session(agent.history())
#endstep


# 程序主入口：当文件被直接执行时调用 main() 函数
if __name__ == "__main__":
    main()
