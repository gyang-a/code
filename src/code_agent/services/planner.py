from __future__ import annotations


def default_plan(user_goal: str) -> list[str]:
    return [
        "检查项目结构和相关清单文件。",
        "搜索与用户目标相关的文件。",
        "读取最小且必要的文件集合。",
        "如需修改，通过安全工具应用小范围编辑。",
        "运行可用且相关的验证命令。",
        "总结修改内容、验证结果和剩余风险。",
    ]
