"""Shared YAML frontmatter parser for memory and skills files.
Handles simple `key: value` pairs between `---` delimiters."""

# ──────────────────────────────────────────────────────────────────────────────
# YAML 前置元数据解析器 (Frontmatter Parser)
# ──────────────────────────────────────────────────────────────────────────────
# 本模块为 memory（记忆）和 skills（技能）文件提供共享的 YAML 前置元数据解析功能。
# 支持解析 `---` 分隔符之间的简单 `key: value` 键值对格式。
# 典型用法：
#   ---
#   name: my-skill
#   description: 一个示例技能
#   ---
#   技能正文内容...
#
# 解析结果包含两部分：meta（元数据字典）和 body（正文字符串）。
# ──────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field


@dataclass
class FrontmatterResult:
    """前置元数据解析结果数据类

    Attributes:
        meta: 从前置元数据区域解析出的键值对字典，默认为空字典
        body: 前置元数据分隔符之后的正文内容，默认为空字符串
    """
    meta: dict[str, str] = field(default_factory=dict)
    body: str = ""


def parse_frontmatter(content: str) -> FrontmatterResult:
    """解析带有 YAML 前置元数据的文件内容

    解析逻辑：
      1. 检查第一行是否为 "---" 分隔符，不是则整个内容作为正文返回
      2. 查找结束分隔符 "---"，找不到则整个内容作为正文返回
      3. 在两个分隔符之间逐行解析 `key: value` 格式的键值对
      4. 分隔符之后的所有内容作为正文

    Args:
        content: 原始文件内容字符串

    Returns:
        FrontmatterResult: 包含 meta（元数据字典）和 body（正文）的解析结果
    """
    lines = content.split("\n")
    # 检查是否有起始分隔符
    if not lines or lines[0].strip() != "---":
        return FrontmatterResult(body=content)  # 无前置元数据，全部作为正文

    # 查找结束分隔符 "---"
    end_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx == -1:
        return FrontmatterResult(body=content)  # 未找到结束分隔符，全部作为正文

    # 解析两个分隔符之间的键值对
    meta: dict[str, str] = {}
    for i in range(1, end_idx):
        colon_idx = lines[i].find(":")  # 找到第一个冒号作为键值分隔
        if colon_idx == -1:
            continue  # 跳过不含冒号的行
        key = lines[i][:colon_idx].strip()
        value = lines[i][colon_idx + 1:].strip()
        if key:
            meta[key] = value

    # 提取结束分隔符之后的所有内容作为正文
    body = "\n".join(lines[end_idx + 1:]).strip()
    return FrontmatterResult(meta=meta, body=body)


def format_frontmatter(meta: dict[str, str], body: str) -> str:
    """将元数据字典和正文格式化为带有 YAML 前置元数据的字符串

    生成格式示例：
      ---
      name: my-skill
      description: 示例
      ---
      正文内容

    Args:
        meta: 要写入的元数据键值对字典
        body: 正文内容字符串

    Returns:
        格式化后的完整文件内容字符串
    """
    lines = ["---"]  # 起始分隔符
    for key, value in meta.items():
        lines.append(f"{key}: {value}")  # 每个键值对占一行
    lines.append("---")  # 结束分隔符
    lines.append("")  # 空行分隔元数据和正文
    lines.append(body)
    return "\n".join(lines)
