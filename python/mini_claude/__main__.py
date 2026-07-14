"""CLI entry point and interactive REPL — mirrors cli.ts."""
# CLI 入口和交互式 REPL（Read-Eval-Print Loop）——对应 TypeScript 版本的 cli.ts
# 本文件是 mini-claude 的主入口模块，负责：
#   1. 解析命令行参数（如 --yolo, --plan, --model 等）
#   2. 配置 API 密钥和后端地址（支持 Anthropic 和 OpenAI 兼容格式）
#   3. 创建 Agent 实例并启动交互式 REPL 或一次性（one-shot）模式
#   4. 处理 REPL 内置命令（/clear, /plan, /cost, /compact, /goal, /loop 等）
#   5. 处理技能（skill）调用（以 / 开头的自定义命令）

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys

from .agent import Agent
from .ui import print_welcome, print_user_prompt, print_error, print_info, print_plan_for_approval, print_plan_approval_options
from .session import load_session, get_latest_session_id
from .memory import list_memories
from .skills import discover_skills, resolve_skill_prompt, get_skill_by_name, execute_skill


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    使用 argparse 定义并解析 mini-claude 支持的所有命令行选项，
    包括运行模式（--yolo, --plan, --auto）、模型选择（--model）、
    API 配置（--api-base）、会话恢复（--resume）以及资源限制（--max-cost, --max-turns）。

    返回:
        argparse.Namespace: 解析后的参数对象，每个选项对应一个属性。
    """
    parser = argparse.ArgumentParser(
        prog="mini-claude",
        description="Mini Claude Code — a minimal coding agent",
        add_help=False,
    )
    parser.add_argument("prompt", nargs="*", help="One-shot prompt")  # 一次性提示词（非交互模式）
    parser.add_argument("--yolo", "-y", action="store_true", help="Skip all confirmation prompts")  # 跳过所有确认提示
    parser.add_argument("--plan", action="store_true", help="Plan mode: read-only")  # 计划模式：只读，不执行修改
    parser.add_argument("--accept-edits", action="store_true", help="Auto-approve file edits")  # 自动批准文件编辑
    parser.add_argument("--dont-ask", action="store_true", help="Auto-deny confirmations (for CI)")  # 自动拒绝确认（CI 环境用）
    parser.add_argument("--auto", action="store_true", help="Auto Mode: LLM classifier judges each action")  # 自动模式：由 LLM 分类器判断每个操作
    parser.add_argument("--thinking", action="store_true", help="Enable extended thinking")  # 启用扩展思考（深度推理）
    parser.add_argument("--model", "-m", default=None, help="Model to use")  # 指定使用的模型名称
    parser.add_argument("--api-base", default=None, help="OpenAI-compatible API base URL")  # OpenAI 兼容的 API 基础 URL
    parser.add_argument("--resume", action="store_true", help="Resume last session")  # 恢复上一次会话
    parser.add_argument("--max-cost", type=float, default=None, help="Max USD spend")  # 最大花费上限（美元）
    parser.add_argument("--max-turns", type=int, default=None, help="Max agentic turns")  # 最大代理轮次
    parser.add_argument("--help", "-h", action="store_true", help="Show help")  # 显示帮助信息
    return parser.parse_args()


def _resolve_permission_mode(args: argparse.Namespace) -> str:
    """根据命令行参数确定权限模式。

    按优先级从高到低检查各个标志位，返回对应的权限模式字符串。
    优先级顺序：yolo > plan > accept_edits > dont_ask > auto > default。

    参数:
        args (argparse.Namespace): 解析后的命令行参数。

    返回:
        str: 权限模式字符串，可选值为:
            - "bypassPermissions": 跳过所有权限检查（--yolo）
            - "plan": 只读计划模式（--plan）
            - "acceptEdits": 自动批准编辑（--accept-edits）
            - "dontAsk": 自动拒绝需确认的操作（--dont-ask）
            - "auto": 由 LLM 分类器判断（--auto）
            - "default": 默认模式，交互式确认
    """
    if args.yolo:
        return "bypassPermissions"
    if args.plan:
        return "plan"
    if args.accept_edits:
        return "acceptEdits"
    if args.dont_ask:
        return "dontAsk"
    if args.auto:
        return "auto"
    return "default"


async def run_repl(agent: Agent) -> None:
    """Interactive REPL loop."""
    # 交互式 REPL 主循环
    # 持续读取用户输入，分发到对应的命令处理逻辑或普通对话。
    # 支持内置命令（/clear, /plan, /cost 等）和技能调用（/<skill-name>）。

    async def confirm_fn(message: str) -> bool:
        """确认回调函数 —— 在需要用户确认操作时调用。

        参数:
            message (str): 向用户展示的确认提示信息。

        返回:
            bool: 用户是否同意该操作。True 表示同意，False 表示拒绝。
        """
        try:
            answer = input("  Allow? (y/n): ")
            return answer.lower().startswith("y")
        except EOFError:
            return False

    agent.set_confirm_fn(confirm_fn)  # 将确认回调注册到 Agent

    async def plan_approval_fn(plan_content: str) -> dict:
        """计划审批回调函数 —— 在计划模式下，展示计划供用户审批。

        用户可以选择：1) 清空并执行 2) 执行 3) 手动执行 4) 继续规划并提供反馈。

        参数:
            plan_content (str): 生成的计划内容文本。

        返回:
            dict: 包含用户选择的字典，键 "choice" 为选择项，
                  键 "feedback"（可选）为用户提供的修改反馈。
        """
        print_plan_for_approval(plan_content)  # 展示计划内容
        print_plan_approval_options()  # 展示审批选项菜单
        while True:
            try:
                choice = input("  Enter choice (1-4): ").strip()
            except EOFError:
                return {"choice": "manual-execute"}  # EOF 时默认手动执行
            if choice == "1":
                return {"choice": "clear-and-execute"}  # 清空上下文并执行计划
            elif choice == "2":
                return {"choice": "execute"}  # 直接执行计划
            elif choice == "3":
                return {"choice": "manual-execute"}  # 用户手动执行
            elif choice == "4":
                try:
                    feedback = input("  Feedback (what to change): ").strip()
                except EOFError:
                    feedback = ""
                return {"choice": "keep-planning", "feedback": feedback or None}  # 继续规划并附带反馈
            else:
                print("  Invalid choice. Enter 1, 2, 3, or 4.")

    agent.set_plan_approval_fn(plan_approval_fn)  # 将计划审批回调注册到 Agent

    sigint_count = 0  # 记录连续 Ctrl+C 的次数，两次则退出

    def handle_sigint(sig, frame):
        """SIGINT（Ctrl+C）信号处理器。

        行为逻辑：
        - 如果 Agent 正在处理任务：中止当前任务并重置计数。
        - 如果 Agent 空闲：第一次提示再次按 Ctrl+C 退出，第二次直接退出。

        参数:
            sig: 信号编号。
            frame: 当前栈帧（未使用）。
        """
        nonlocal sigint_count
        # Always signal a running /loop or /goal to stop — during its inter-tick
        # wait or between-turn evaluation the agent isn't "processing", so the
        # abort path below wouldn't catch it.
        # 停止正在运行的 /loop 或 /goal 任务
        agent.stop_loop()
        agent.stop_goal()
        # is_processing tracks the live task; _output_buffer is only set for
        # SUB-agents, so testing it here meant the main agent could never be
        # interrupted mid-task.
        if agent._aborted is False and agent.is_processing:
            # Agent 正在处理中 —— 中止当前任务
            agent.abort()
            print("\n  (interrupted)")
            sigint_count = 0
            print_user_prompt()
        else:
            # Agent 空闲 —— 计数并判断是否退出
            sigint_count += 1
            if sigint_count >= 2:
                print("\nBye!\n")
                sys.exit(0)
            print("\n  Press Ctrl+C again to exit.")
            print_user_prompt()

    signal.signal(signal.SIGINT, handle_sigint)  # 注册 SIGINT 信号处理器
    print_welcome()  # 打印欢迎信息

    # ===== REPL 主循环 =====
    while True:
        print_user_prompt()  # 打印输入提示符
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!\n")
            break

        inp = line.strip()
        sigint_count = 0  # 每次有效输入重置 Ctrl+C 计数

        if not inp:
            continue  # 空输入，跳过
        if inp in ("exit", "quit"):
            print("\nBye!\n")
            break

        # ===== REPL 内置命令处理 =====
        # REPL commands
        if inp == "/clear":
            agent.clear_history()  # 清空对话历史
            continue
        if inp == "/plan":
            agent.toggle_plan_mode()  # 切换计划模式（只读/正常）
            continue
        if inp == "/cost":
            agent.show_cost()  # 显示 token 使用量和费用
            continue
        if inp == "/compact":
            try:
                await agent.compact()  # 手动压缩对话历史（减少上下文长度）
            except Exception as e:
                print_error(str(e))
            continue
        if inp == "/goal" or inp.startswith("/goal "):
            condition = inp[len("/goal"):].strip()
            if not condition:
                agent.show_goal()  # 无参数时显示当前目标状态
                continue
            directive = agent.set_goal(condition)  # 设置新目标
            try:
                await agent.pursue_goal(directive)  # 追逐目标直到评估器判断已达成
            except Exception as e:
                if "abort" not in str(e).lower():
                    print_error(str(e))
            continue
        if inp == "/loop" or inp.startswith("/loop "):
            rest = inp[len("/loop"):].strip()
            try:
                await agent.run_loop(rest)  # 按间隔循环运行指定提示词
            except Exception as e:
                if "abort" not in str(e).lower():
                    print_error(str(e))
            continue
        if inp == "/memory":
            memories = list_memories()  # 列出所有已保存的记忆
            if not memories:
                print_info("No memories saved yet.")
            else:
                print_info(f"{len(memories)} memories:")
                for m in memories:
                    print(f"    [{m.type}] {m.name} — {m.description}")
            continue
        if inp == "/skills":
            skills = discover_skills()  # 发现所有可用技能
            if not skills:
                print_info("No skills found. Add skills to .claude/skills/<name>/SKILL.md")
            else:
                print_info(f"{len(skills)} skills:")
                for s in skills:
                    tag = f"/{s.name}" if s.user_invocable else s.name  # 用户可调用的技能显示为 /name
                    print(f"    {tag} ({s.source}) — {s.description}")
            continue

        # Skill invocation: /<skill-name> [args]
        # 技能调用：解析 /<skill-name> [args] 格式的输入
        if inp.startswith("/"):
            space_idx = inp.find(" ")  # 查找空格分隔符
            cmd_name = inp[1:space_idx] if space_idx > 0 else inp[1:]  # 提取技能名称
            cmd_args = inp[space_idx + 1:] if space_idx > 0 else ""  # 提取技能参数
            skill = get_skill_by_name(cmd_name)  # 根据名称查找技能
            if skill and skill.user_invocable:
                print_info(f"Invoking skill: {skill.name}")
                try:
                    if skill.context == "fork":
                        # fork 上下文：先在子进程中执行技能，再将结果作为聊天上下文
                        result = execute_skill(skill.name, cmd_args)
                        if result:
                            await agent.chat(f'Use the skill tool to invoke "{skill.name}" with args: {cmd_args or "(none)"}')
                    else:
                        # 默认上下文：将技能提示词解析后直接作为聊天输入
                        resolved = resolve_skill_prompt(skill, cmd_args)
                        await agent.chat(resolved)
                except Exception as e:
                    if "abort" not in str(e).lower():
                        print_error(str(e))
                continue

        # Normal chat
        # 普通对话 —— 将用户输入发送给 Agent 处理
        try:
            await agent.chat(inp)
        except Exception as e:
            if "abort" not in str(e).lower():
                print_error(str(e))

    # Loop exited (EOF / exit / quit) — release MCP subprocesses (issue #8)
    # REPL 循环结束（EOF / exit / quit）—— 释放 MCP 子进程资源（修复 issue #8）
    await agent.close()


def main() -> None:
    """程序主入口函数。

    处理流程：
    1. 解析命令行参数
    2. 如果请求帮助（--help），打印帮助信息并退出
    3. 确定权限模式、模型名称和 API 配置
    4. 创建 Agent 实例
    5. 如需恢复会话（--resume），加载上次会话历史
    6. 根据是否提供了 prompt 决定进入一次性模式或交互式 REPL
    """
    args = parse_args()  # 解析命令行参数

    if args.help:
        # 打印帮助信息并退出
        print("""
Usage: mini-claude [options] [prompt]

Options:
  --yolo, -y          Skip all confirmation prompts (bypassPermissions mode)
  --plan              Plan mode: read-only, describe changes without executing
  --accept-edits      Auto-approve file edits, still confirm dangerous shell
  --dont-ask          Auto-deny anything needing confirmation (for CI)
  --auto              Auto Mode: an LLM classifier judges each action instead of asking
  --thinking          Enable extended thinking (Anthropic only)
  --model, -m         Model to use (default: claude-opus-4-6, or MINI_CLAUDE_MODEL env)
  --api-base URL      Use OpenAI-compatible API endpoint (key via env var)
  --resume            Resume the last session
  --max-cost USD      Stop when estimated cost exceeds this amount
  --max-turns N       Stop after N agentic turns
  --help, -h          Show this help

REPL commands:
  /clear              Clear conversation history
  /plan               Toggle plan mode (read-only <-> normal)
  /cost               Show token usage and cost
  /compact            Manually compact conversation
  /goal <condition>   Pursue a goal across turns until an evaluator judges it met
  /goal               Show the active goal's status
  /loop [interval] <prompt>  Re-run a prompt on an interval (5m/2h) or self-paced
  /memory             List saved memories
  /skills             List available skills
  /<skill-name>       Invoke a skill (e.g. /commit "fix types")

Examples:
  mini-claude "fix the bug in src/app.ts"
  mini-claude --yolo "run all tests and fix failures"
  mini-claude --plan "how would you refactor this?"
  mini-claude --max-cost 0.50 --max-turns 20 "implement feature X"
  OPENAI_API_KEY=sk-xxx mini-claude --api-base https://aihubmix.com/v1 --model gpt-4o "hello"
  mini-claude --resume
  mini-claude  # starts interactive REPL
""")
        sys.exit(0)

    # 确定权限模式（yolo / plan / acceptEdits / dontAsk / auto / default）
    permission_mode = _resolve_permission_mode(args)
    # 确定使用的模型：优先命令行参数，其次环境变量，最后默认值
    model = args.model or os.environ.get("MINI_CLAUDE_MODEL", "claude-opus-4-6")
    api_base = args.api_base

    # Resolve API config
    # 解析 API 配置 —— 支持 Anthropic 和 OpenAI 兼容两种格式
    resolved_api_base = api_base
    resolved_api_key: str | None = None
    resolved_use_openai = bool(api_base)

    # 优先级：同时设置了 OPENAI_API_KEY 和 OPENAI_BASE_URL > ANTHROPIC_API_KEY > 单独的 OPENAI_API_KEY
    if os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL"):
        # 使用 OpenAI 兼容格式（同时设置了 key 和 base URL）
        resolved_api_key = os.environ["OPENAI_API_KEY"]
        resolved_api_base = resolved_api_base or os.environ.get("OPENAI_BASE_URL")
        resolved_use_openai = True
    elif os.environ.get("ANTHROPIC_API_KEY"):
        # 使用 Anthropic 原生格式
        resolved_api_key = os.environ["ANTHROPIC_API_KEY"]
        resolved_api_base = resolved_api_base or os.environ.get("ANTHROPIC_BASE_URL")
        resolved_use_openai = False
    elif os.environ.get("OPENAI_API_KEY"):
        # 仅有 OPENAI_API_KEY，使用 OpenAI 兼容格式
        resolved_api_key = os.environ["OPENAI_API_KEY"]
        resolved_api_base = resolved_api_base or os.environ.get("OPENAI_BASE_URL")
        resolved_use_openai = True

    # 如果指定了 api_base 但未找到密钥，尝试从环境变量中获取
    if not resolved_api_key and api_base:
        resolved_api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        resolved_use_openai = True

    if not resolved_api_key:
        # 未找到有效的 API 密钥，报错并退出
        print_error(
            "API key is required.\n"
            "  Set ANTHROPIC_API_KEY (+ optional ANTHROPIC_BASE_URL) for Anthropic format,\n"
            "  or OPENAI_API_KEY + OPENAI_BASE_URL for OpenAI-compatible format."
        )
        sys.exit(1)

    # 创建 Agent 实例，传入所有配置参数
    agent = Agent(
        permission_mode=permission_mode,
        model=model,
        thinking=args.thinking,
        max_cost_usd=args.max_cost,
        max_turns=args.max_turns,
        api_base=resolved_api_base if resolved_use_openai else None,
        anthropic_base_url=resolved_api_base if not resolved_use_openai else None,
        api_key=resolved_api_key,
    )

    # Resume session
    # 恢复上次会话（如指定了 --resume 参数）
    if args.resume:
        session_id = get_latest_session_id()  # 获取最近一次会话的 ID
        if session_id:
            session = load_session(session_id)  # 加载会话历史
            if session:
                agent.restore_session({
                    "anthropicMessages": session.get("anthropicMessages"),
                    "openaiMessages": session.get("openaiMessages"),
                })
            else:
                print_info("No session found to resume.")
        else:
            print_info("No previous sessions found.")

    # 将命令行中的 prompt 参数拼接为完整提示词
    prompt = " ".join(args.prompt) if args.prompt else None

    if prompt:
        # One-shot mode — always release MCP subprocesses on the way out (issue #8)
        # 一次性模式 —— 处理完单个 prompt 后退出，始终释放 MCP 子进程资源
        async def _one_shot() -> None:
            """一次性模式的异步包装函数。"""
            try:
                await agent.chat(prompt)
            finally:
                await agent.close()  # 确保在退出时释放资源
        try:
            asyncio.run(_one_shot())
        except Exception as e:
            print_error(str(e))
            sys.exit(1)
    else:
        # Interactive REPL
        # 交互式 REPL 模式 —— 进入持续对话循环
        asyncio.run(run_repl(agent))


if __name__ == "__main__":
    # 模块直接运行时调用 main 函数
    main()
