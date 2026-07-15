# Python 代码阅读顺序

本文档为 `python/mini_claude/` 包的推荐阅读顺序，从底层依赖到核心引擎，逐步深入。

---

## 依赖关系图

```
frontmatter.py ←── memory.py ←── tools.py ←── agent.py (核心引擎)
                  ←── skills.py ←── prompt.py ←──┘
                  ←── subagent.py ←──┘
session.py ←──────────────────────────────────────┘
ui.py ←───────────────────────────────────────────┘
autonomy.py ←─────────────────────────────────────┘
mcp_client.py ←───────────────────────────────────┘
__main__.py ──→ Agent (创建并驱动)
```

---

## 第一层：基础工具（无内部依赖，先读）

### 1. `frontmatter.py` (102 行)

最底层的解析器，被 `memory.py` 和 `skills.py` 依赖。

**重点：**
- `parse_frontmatter()` — 将 `---` 分隔的 YAML 元数据拆分为 `meta` 字典 + `body` 正文
- `format_frontmatter()` — 反向序列化，将元数据和正文拼回带 frontmatter 的字符串
- `FrontmatterResult` 数据类结构

---

### 2. `session.py` (121 行)

纯粹的 JSON 文件持久化，4 个函数完成会话的 CRUD。

**重点：**
- 存储路径：`~/.mini-claude/sessions/`
- `save_session()` / `load_session()` — 会话数据的序列化与反序列化
- `get_latest_session_id()` — 按 `startTime` 排序获取最近会话

---

## 第二层：核心子系统（依赖第一层）

### 3. `ui.py` (280 行)

终端渲染层，基于 Rich 库实现彩色输出。

**重点：**
- `print_tool_call()` / `print_tool_result()` — 工具调用的输出格式
- spinner 的线程实现（`start_spinner()` / `stop_spinner()`）
- `print_plan_for_approval()` — 计划审批界面

---

### 4. `memory.py` (480 行) ⭐ 核心模块

基于文件的持久化记忆系统，支持 4 种记忆类型（user/feedback/project/reference）。

**重点：**
- `MemoryEntry` 数据结构（`__slots__` 优化内存）
- `get_memory_dir()` — 基于项目路径哈希的隔离机制
- `start_memory_prefetch()` — 异步预取记忆，减少用户等待
- `format_memories_for_injection()` — 将记忆注入系统提示词
- `build_memory_prompt_section()` — 构建记忆描述段落
- 语义召回（Semantic Recall）通过 `SideQueryFn` 调用模型选择相关记忆

---

### 5. `skills.py` (245 行)

技能的发现、解析和执行。

**重点：**
- `discover_skills()` — 扫描 `~/.claude/skills/` 和 `.claude/skills/*/SKILL.md`
- `_parse_skill_file()` — 解析 SKILL.md 的 frontmatter 元数据 + 提示词模板
- `execute_skill()` — 模板变量替换（如 `$ARGUMENTS`）
- 项目级技能覆盖同名用户级技能

---

### 6. `subagent.py` (282 行)

子代理系统，采用"分叉-返回"(fork-return) 模式。

**重点：**
- 三种内置代理类型的工具集限制：
  - **explore** — 只读，快速搜索代码库
  - **plan** — 只读，生成结构化计划
  - **general** — 完整工具集，可执行独立任务
- `.claude/agents/*.md` — 用户自定义代理的加载机制
- `READ_ONLY_TOOLS` 集合定义

---

## 第三层：工具与自主性（依赖第二层）

### 7. `tools.py` (913 行) ⭐ 工具系统核心

定义并执行 11 个工具，支持 5 种权限模式。

**重点：**
- `tool_definitions` — 所有工具的 Anthropic schema 定义
- `execute_tool()` — 工具名到执行函数的分发逻辑
- `check_permission()` — 五种权限模式的判断树：
  - `default` — 读取自动允许，写入需确认
  - `plan` — 只允许读取和编辑计划文件
  - `acceptEdits` — 编辑类工具自动允许
  - `bypassPermissions` — 所有工具自动允许（--yolo）
  - `dontAsk` — 需确认的操作自动拒绝
- `CONCURRENCY_SAFE_TOOLS` — 可并行执行的只读工具集合
- `get_active_tool_definitions()` — 延迟加载工具的激活机制

---

### 8. `autonomy.py` (556 行)

三大自主工作功能的提示词和逻辑。

**重点：**
- **`/goal`** — 目标评估器
  - `goal_directive()` — 首轮注入的目标指令
  - `GOAL_EVALUATOR_SYSTEM` — 评估器系统提示词
  - `parse_goal_verdict()` — 三态解析（ok/reason/impossible）
- **`/loop`** — 循环任务调度
  - `parse_loop_input()` — 定时模式 vs 动态自节奏模式
  - `dynamic_loop_directive()` — 动态循环的指令注入
- **Auto Mode** — LLM 分类器门控
  - `build_classifier_system()` — 分类器系统提示词
  - `parse_block_verdict()` — 判断是否放行
  - `AUTO_MODE_FAST_PATH_TOOLS` — 快速放行的工具白名单

---

### 9. `mcp_client.py` (366 行)

MCP (Model Context Protocol) 客户端，通过 stdio 与外部工具服务器通信。

**重点：**
- `McpConnection` — 单个 MCP 服务器的 JSON-RPC 通信
  - 子进程管理（启动/关闭）
  - 请求 ID 递增机制
- `McpManager` — 多服务器管理器
- 工具命名规则：`mcp__serverName__toolName` 前缀避免冲突
- 配置来源优先级：`~/.claude/settings.json` > `.claude/settings.json` > `.mcp.json`

---

## 第四层：系统提示词构建

### 10. `prompt.py` (340 行) ⭐ 提示词架构核心

构建发送给模型的系统提示词，复刻 Claude Code 的提示词架构。

**重点：**
- `SYSTEM_PROMPT_TEMPLATE` — 静态核心模板，理解 Claude Code 的提示词设计理念
- **静态/动态分离**：静态部分可被前缀缓存，动态部分每轮重建
  - `build_static_system_prompt()` — 静态部分
  - `build_dynamic_system_context()` — 收集环境/Git/记忆/技能等上下文
  - `build_user_context_reminder()` — CLAUDE.md + 日期注入到首条用户消息
- `load_claude_md()` — CLAUDE.md 的 `@include` 递归引用机制
- `.claude/rules/*.md` — 规则目录的加载

---

## 第五层：Agent 核心引擎（最后读，最复杂）

### 11. `agent.py` (2538 行) ⭐⭐⭐ 整个项目的核心

完整的智能体循环引擎，建议分段阅读：

#### 11.1 重试机制 (L100-165)
- `_is_retryable()` — 判断错误是否可重试（429/503/529/网络超时）
- `_with_retry()` — 指数退避重试，避免雪崩效应

#### 11.2 模型能力检测 (L168-245)
- `_get_context_window()` — 不同模型的上下文窗口大小
- `_model_supports_thinking()` — 是否支持扩展思考
- `_get_max_output_tokens()` — 最大输出 token 数

#### 11.3 Agent.__init__ (L290-475)
所有状态初始化，理解双后端（Anthropic / OpenAI 兼容）的配置差异

#### 11.4 chat() 主循环入口 (L648-685)
- one-shot 模式（`run_once()`）vs REPL 模式（`chat()`）的区别
- 记忆预取的触发时机

#### 11.5 _chat_anthropic() 流式处理 (L2014-2175) ⭐ 最核心
- stream → `tool_use` 块累积 → 并发/串行执行 → 结果回填的完整流程
- `_on_tool_block` 回调：流式生成过程中提前执行工具
- `CONCURRENCY_SAFE_TOOLS` 并发执行 vs 串行执行的判断

#### 11.6 _chat_openai() 流式处理 (L2271-2433)
- OpenAI 兼容后端的实现，与 Anthropic 版本对称
- tool_calls 的增量解析和执行

#### 11.7 四层上下文压缩 (L1380-1600)
渐进式上下文管理，从轻量到激进：
1. **budget** — 裁剪过大的工具结果
2. **snip** — 移除过时的工具调用结果
3. **microcompact** — 微压缩，移除低价值内容
4. **autocompact** — 自动压缩，调用模型总结历史

#### 11.8 计划模式 (L1738-1870)
- `_build_plan_mode_prompt()` — 计划模式的系统提示词注入
- `enter_plan_mode()` / `exit_plan_mode()` — 状态机转换
- 计划文件的生成与用户审批流程

#### 11.9 子代理执行 (L1877-1920)
- `_execute_agent_tool()` — fork-return 模式的实现
- 子代理的工具集隔离和结果返回

#### 11.10 目标/循环 (L806-1170)
- `set_goal()` / `pursue_goal()` — `/goal` 的完整生命周期
- `_evaluate_goal()` — 调用评估模型判断目标是否达成
- `run_loop()` — `/loop` 的定时/动态两种模式
- `_run_loop_dynamic()` — 自节奏循环，根据输出决定下次延迟

---

## 第六层：入口点

### 12. `__main__.py` (450 行)

CLI 入口和交互式 REPL。

**重点：**
- `parse_args()` — 命令行参数定义（--yolo, --plan, --auto, --model 等）
- `_resolve_permission_mode()` — 参数到权限模式的优先级链
- REPL 主循环的命令分发：
  - `/clear` — 清空历史
  - `/plan` — 切换计划模式
  - `/cost` — 显示费用统计
  - `/compact` — 手动触发压缩
  - `/goal` — 设置/查看目标
  - `/loop` — 启动循环任务
  - `/memory` — 管理记忆
  - `/skills` — 查看可用技能

---

## 关键设计思想

| 设计 | 说明 |
|------|------|
| **双后端** | Anthropic 原生 API + OpenAI 兼容 API，通过 `_chat_anthropic()` / `_chat_openai()` 分叉 |
| **静态/动态提示词分离** | 静态部分可被 API 前缀缓存，动态部分每轮重建 |
| **四层压缩** | budget → snip → microcompact → autocompact，渐进式上下文管理 |
| **权限门控** | 5 种模式 + Auto Mode LLM 分类器，安全与效率的平衡 |
| **记忆系统** | 文件持久化 + 语义召回 + 异步预取 |
| **子代理** | fork-return 模式，三种内置类型 + 用户自定义 |
