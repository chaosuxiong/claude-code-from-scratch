import os  # 用于文件路径操作和文件存在性检查

# A skill is a reusable prompt template in .mini-skills/<name>.md. Typing
# "/commit ..." loads it and runs its prompt (with any extra text appended) as
# if you'd typed the whole thing — install-and-use like a shell script.
# 【技能系统模块】
# 技能(Skill)是一种可复用的提示词模板，存储在 .mini-skills/<name>.md 文件中。
# 用户输入 "/commit ..." 时，系统会加载对应的 .md 文件，读取其中的提示词模板，
# 并将用户输入的额外文本附加到模板末尾，然后像用户手动输入完整内容一样执行。
# 这种机制类似于 Shell 脚本的安装和使用方式——用户只需创建 .md 文件即可添加新技能。
SKILLS_DIR = os.path.join(os.getcwd(), ".mini-skills")
# 技能目录路径：当前工作目录下的 .mini-skills 文件夹


#region skill
def resolve_skill(text):
    """解析用户输入，如果匹配某个技能则返回组合后的完整提示词。

    【功能说明】
    检查用户输入的文本是否以 "/" 开头（即是否是技能调用语法），
    如果是，则提取技能名称，查找对应的 .md 文件，读取模板内容，
    并将用户输入的额外参数附加到模板末尾。

    【参数】
        text (str): 用户输入的原始文本，例如 "/commit fix bug" 或 "/help"

    【返回值】
        str | None:
            - 如果匹配到有效技能，返回组合后的完整提示词（模板 + 参数）
            - 如果不匹配任何技能（不以 "/" 开头或技能文件不存在），返回 None
    """
    # 检查输入是否以 "/" 开头，不以 "/" 开头则不是技能调用
    if not text.startswith("/"):
        return None
    # 从输入中提取技能名称和剩余参数
    # 例如 "/commit fix bug" -> name="commit", rest="fix bug"
    # partition(" ") 在第一个空格处分割，返回 (名称, 空格, 剩余部分)
    name, _, rest = text[1:].partition(" ")
    # 构建技能文件的完整路径：.mini-skills/<name>.md
    path = os.path.join(SKILLS_DIR, f"{name}.md")
    # 如果对应的 .md 文件不存在，则该技能未安装，返回 None
    if not os.path.exists(path):
        return None
    # 读取 .md 文件内容作为提示词模板，并去除首尾空白字符
    prompt = open(path, encoding="utf-8").read().strip()
    # 获取用户输入的额外参数，并去除首尾空白
    args = rest.strip()
    # 如果有额外参数，将其附加到提示词模板后面（用两个换行分隔）
    # 如果没有额外参数，直接返回模板内容
    return f"{prompt}\n\n{args}" if args else prompt
#endregion
