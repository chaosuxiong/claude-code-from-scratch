#!/usr/bin/env python3
"""终端贪吃蛇游戏 - 使用 curses 实现

【模块说明】
本模块实现了一个基于终端的贪吃蛇游戏，使用 Python 标准库 curses 进行终端界面渲染。

【功能特性】
  - 使用方向键控制蛇的移动方向
  - 蛇每吃一个食物（"*"）得 10 分，同时速度加快
  - 蛇头用 "@" 表示，蛇身用 "o" 表示
  - 撞墙或撞到自身则游戏结束
  - 游戏结束后可按 "r" 重新开始，按 "q" 退出

【运行方式】
  python snake.py
"""

import curses
import random
import time


def main(stdscr):
    """主函数：初始化 curses 环境并管理游戏主循环。

    【功能说明】
    设置终端颜色对（蛇身、食物、分数、边框），然后反复调用 game_loop
    运行游戏。当玩家选择退出时，游戏主循环结束。

    【参数】
        stdscr: curses 标准屏幕对象，由 curses.wrapper 自动传入

    【返回值】
        无
    """
    # 初始化 curses
    curses.curs_set(0)  # 隐藏光标，使界面更整洁
    curses.start_color()  # 启用颜色支持
    # 定义颜色对：(编号, 前景色, 背景色)
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)   # 蛇身 - 绿色
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)     # 食物 - 红色
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # 分数 - 黄色
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)    # 边框 - 青色

    # 游戏主循环：反复启动新一局游戏，直到玩家选择退出
    while True:
        result = game_loop(stdscr)
        if result == "quit":
            break


def game_loop(stdscr):
    """单局游戏循环：处理游戏逻辑、渲染和用户输入。

    【功能说明】
    管理一局完整的游戏流程，包括：
      1. 初始化游戏区域和边框
      2. 初始化蛇的位置和方向
      3. 在游戏循环中处理用户输入、移动蛇、碰撞检测和食物生成
      4. 渲染画面（分数、食物、蛇）

    【参数】
        stdscr: curses 标准屏幕对象

    【返回值】
        str: "quit" 表示退出游戏，"restart" 表示重新开始（由 game_over 返回）

    【游戏区域布局】
        - 第 0 行：分数显示栏
        - 第 1 行起：游戏边框，内部为蛇的活动区域
    """
    stdscr.clear()  # 清空屏幕
    stdscr.nodelay(True)  # 非阻塞输入：getch() 不会等待按键
    stdscr.timeout(100)  # 设置刷新间隔为 100 毫秒，控制蛇的移动速度

    # 获取终端窗口尺寸
    sh, sw = stdscr.getmaxyx()

    # 游戏区域定义（留出顶部 2 行给边框和分数栏，左侧 1 列边距）
    top, left = 2, 1  # 游戏区域左上角坐标
    height = sh - 3  # 游戏区域高度（减去顶部边框和底部边框）
    width = sw - 2  # 游戏区域宽度（减去左右边框）

    # 终端尺寸检查：太小则无法正常显示游戏
    if height < 10 or width < 20:
        stdscr.addstr(sh // 2, 0, "终端太小，请调整窗口大小！")
        stdscr.nodelay(False)  # 恢复阻塞输入，等待用户按键
        stdscr.getch()
        return "quit"

    # 绘制游戏区域边框（使用 curses 特殊字符）
    # 上下边框：水平线
    for x in range(left, left + width):
        stdscr.addch(top, x, curses.ACS_HLINE, curses.color_pair(4))
        stdscr.addch(top + height - 1, x, curses.ACS_HLINE, curses.color_pair(4))
    # 左右边框：垂直线
    for y in range(top, top + height):
        stdscr.addch(y, left, curses.ACS_VLINE, curses.color_pair(4))
        stdscr.addch(y, left + width - 1, curses.ACS_VLINE, curses.color_pair(4))
    # 四个角：使用对应的角落字符
    stdscr.addch(top, left, curses.ACS_ULCORNER, curses.color_pair(4))          # 左上角
    stdscr.addch(top, left + width - 1, curses.ACS_URCORNER, curses.color_pair(4))  # 右上角
    stdscr.addch(top + height - 1, left, curses.ACS_LLCORNER, curses.color_pair(4))  # 左下角
    stdscr.addch(top + height - 1, left + width - 1, curses.ACS_LRCORNER, curses.color_pair(4))  # 右下角

    # 初始化蛇：在游戏区域中心生成初始蛇身（3 节，向右移动）
    cy, cx = top + height // 2, left + width // 2  # 蛇头初始位置（中心）
    snake = [(cy, cx), (cy, cx - 1), (cy, cx - 2)]  # 蛇身坐标列表，头部在前
    direction = curses.KEY_RIGHT  # 初始移动方向：向右

    # 在空白位置生成第一个食物
    food = spawn_food(snake, top, left, height, width)

    score = 0  # 当前得分
    speed = 100  # 初始移动速度（毫秒），数值越小速度越快

    # === 游戏主循环 ===
    while True:
        # 显示分数和操作提示（居中显示在屏幕顶部）
        score_text = f" Score: {score} | Arrow keys to move | q to quit "
        stdscr.addstr(0, (sw - len(score_text)) // 2, score_text, curses.color_pair(3) | curses.A_BOLD)

        # 绘制食物（红色 "*"）
        stdscr.addch(food[0], food[1], "*", curses.color_pair(2) | curses.A_BOLD)

        # 绘制蛇：头部为 "@"，身体为 "o"（绿色加粗）
        for i, (y, x) in enumerate(snake):
            ch = "@" if i == 0 else "o"  # 第一个元素是蛇头，其余是蛇身
            stdscr.addch(y, x, ch, curses.color_pair(1) | curses.A_BOLD)

        stdscr.refresh()  # 刷新屏幕，使绘制内容生效

        # 获取用户按键输入（非阻塞）
        key = stdscr.getch()

        # 按 q 键退出游戏
        if key == ord("q"):
            return "quit"

        # 防止蛇反向移动（例如正在向右走时不能直接向左）
        # 定义每个方向的反方向
        opposites = {
            curses.KEY_UP: curses.KEY_DOWN,
            curses.KEY_DOWN: curses.KEY_UP,
            curses.KEY_LEFT: curses.KEY_RIGHT,
            curses.KEY_RIGHT: curses.KEY_LEFT,
        }
        # 只有当按键不是当前方向的反方向时，才更新方向
        if key in opposites and opposites[key] != direction:
            direction = key

        # 根据当前方向计算新蛇头位置
        head_y, head_x = snake[0]  # 获取当前蛇头坐标
        if direction == curses.KEY_UP:
            head_y -= 1  # 向上：行号减小
        elif direction == curses.KEY_DOWN:
            head_y += 1  # 向下：行号增大
        elif direction == curses.KEY_LEFT:
            head_x -= 1  # 向左：列号减小
        elif direction == curses.KEY_RIGHT:
            head_x += 1  # 向右：列号增大

        new_head = (head_y, head_x)  # 新的蛇头坐标

        # 碰撞检测 1：撞墙（新蛇头超出游戏区域边界）
        if (head_y <= top or head_y >= top + height - 1 or
                head_x <= left or head_x >= left + width - 1):
            return game_over(stdscr, score)  # 游戏结束

        # 碰撞检测 2：撞自身（新蛇头与蛇身任意一节重叠）
        if new_head in snake:
            return game_over(stdscr, score)  # 游戏结束

        # 将新蛇头插入到蛇身列表的最前面（模拟蛇的前进）
        snake.insert(0, new_head)

        # 判断是否吃到食物
        if new_head == food:
            # 吃到食物：加分 +10，生成新食物，加速蛇的移动
            score += 10
            food = spawn_food(snake, top, left, height, width)  # 生成新食物
            # 加速：每吃一个食物减少 2ms 延迟，最快不超过 50ms
            speed = max(50, speed - 2)
            stdscr.timeout(speed)  # 更新刷新间隔，使蛇移动更快
        else:
            # 没吃到食物：移除蛇尾（模拟蛇的前进，长度不变）
            tail = snake.pop()  # 弹出蛇尾坐标
            stdscr.addch(tail[0], tail[1], " ")  # 在屏幕上擦除蛇尾


def spawn_food(snake, top, left, height, width):
    """在空白位置随机生成食物坐标。

    【功能说明】
    在游戏区域内随机选取一个位置，确保该位置不在蛇身上，
    然后返回该坐标作为新食物的位置。

    【参数】
        snake (list): 蛇身坐标列表，每个元素为 (y, x) 元组
        top (int): 游戏区域顶部行号
        left (int): 游戏区域左侧列号
        height (int): 游戏区域高度
        width (int): 游戏区域宽度

    【返回值】
        tuple: (y, x) 食物的坐标

    【实现说明】
    使用 while True 循环不断随机生成坐标，直到找到一个
    不与蛇身重叠的位置。当蛇身几乎占满整个区域时，
    循环次数会显著增加。
    """
    while True:
        # 在游戏区域内部随机生成坐标（避开边框）
        y = random.randint(top + 1, top + height - 2)
        x = random.randint(left + 1, left + width - 2)
        # 确保食物不会生成在蛇身上
        if (y, x) not in snake:
            return (y, x)


def game_over(stdscr, score):
    """显示游戏结束画面并等待玩家选择操作。

    【功能说明】
    清空输入缓冲，显示带有边框的游戏结束画面，
    包括最终得分和操作提示。等待玩家按 "r" 重新开始或按 "q" 退出。

    【参数】
        stdscr: curses 标准屏幕对象
        score (int): 本局游戏的最终得分

    【返回值】
        str: "restart" 表示重新开始，"quit" 表示退出游戏
    """
    sh, sw = stdscr.getmaxyx()  # 获取终端尺寸，用于居中显示
    stdscr.nodelay(False)  # 恢复阻塞输入模式，等待玩家按键

    # 游戏结束画面的内容（使用 Unicode 方框字符绘制边框）
    messages = [
        "╔══════════════════════╗",
        "║     GAME  OVER!     ║",
        f"║   Score: {score:>6}      ║",
        "║                      ║",
        "║  r = Restart         ║",
        "║  q = Quit            ║",
        "╚══════════════════════╝",
    ]

    # 计算起始行号，使画面在屏幕垂直居中
    start_y = sh // 2 - len(messages) // 2
    for i, msg in enumerate(messages):
        # 水平居中显示每一行文字
        x = (sw - len(msg)) // 2
        try:
            stdscr.addstr(start_y + i, x, msg, curses.color_pair(3) | curses.A_BOLD)
        except curses.error:
            # 防止文字超出屏幕边界导致异常（忽略即可）
            pass

    stdscr.refresh()  # 刷新屏幕显示

    # 等待玩家按键选择操作
    while True:
        key = stdscr.getch()
        if key == ord("r"):
            return "restart"  # 返回重启信号
        elif key == ord("q"):
            return "quit"  # 返回退出信号


# 程序入口：使用 curses.wrapper 包装 main 函数
# wrapper 会自动处理 curses 的初始化和清理工作（如恢复终端设置）
if __name__ == "__main__":
    curses.wrapper(main)
