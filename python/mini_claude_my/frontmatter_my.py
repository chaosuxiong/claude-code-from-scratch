from dataclasses import dataclass, field


@dataclass
class FrontmatterResult:
    meta: dict[str, str] = field(default_factory=dict)
    body: str = ""


def parse_frontmatter(content: str) -> FrontmatterResult:
    lines = content.split("\n")
    if not lines or lines[0].strip() != "---":
        return FrontmatterResult(body=content)

    end_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx == -1:
        return FrontmatterResult(body=content)

    meta: dict[str, str] = {}
    for i in range(1, end_idx):
        colon_idx = lines[i].find(":")
        if colon_idx == -1:
            continue
        key = lines[i][:colon_idx].strip()
        value = lines[i][colon_idx + 1 :].strip()
        if key:
            meta[key] = value

    body = "\n".join(lines[end_idx + 1 :]).strip()
    return FrontmatterResult(meta, body)


def format_frontmatter(meta: dict[str, str], body: str) -> str:
    lines = ["---"]
    for key, value in meta.items():
        lines.append(f"{key}: {value}")
    lines.append("---")
    lines.append("")
    lines.append(body)
    return "\n".join(lines)


if __name__ == "__main__":
    # --- 测试 parse_frontmatter ---
    content_with_frontmatter = """\
---
name: my-skill
description: 一个示例技能
---
这是正文内容
第二行"""
    result = parse_frontmatter(content_with_frontmatter)
    assert result.meta == {"name": "my-skill", "description": "一个示例技能"}
    assert result.body == "这是正文内容\n第二行"
    print("✅ 有 frontmatter 的内容解析正确")

    # 测试无 frontmatter 的内容
    plain_content = "没有分隔符的纯文本"
    result = parse_frontmatter(plain_content)
    assert result.meta == {}
    assert result.body == plain_content
    print("✅ 无 frontmatter 的内容返回正确")

    # 测试没有结束分隔符的情况
    no_end = """\
---
name: broken
没有结束符"""
    result = parse_frontmatter(no_end)
    assert result.meta == {}
    assert result.body == no_end
    print("✅ 没有结束分隔符时全部作为正文")

    # 测试空内容
    result = parse_frontmatter("")
    assert result.meta == {}
    assert result.body == ""
    print("✅ 空内容返回正确")

    # --- 测试 format_frontmatter ---
    formatted = format_frontmatter({"name": "test", "version": "1.0"}, "正文")
    expected = "---\nname: test\nversion: 1.0\n---\n\n正文"
    assert formatted == expected
    print("✅ format_frontmatter 格式化正确")

    # 测试 round-trip: parse -> format -> parse
    original = content_with_frontmatter
    parsed = parse_frontmatter(original)
    rebuilt = format_frontmatter(parsed.meta, parsed.body)
    reparsed = parse_frontmatter(rebuilt)
    assert reparsed.meta == parsed.meta
    assert reparsed.body == parsed.body
    print("✅ round-trip (解析→格式化→再解析) 一致")

    print("\n🎉 所有测试通过！")
