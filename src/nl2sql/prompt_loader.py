# ============================================================
# Prompt 文件加载器
# ============================================================

from pathlib import Path


def load_prompt(name: str) -> str:
    """读取 prompts/<name>.prompt 文件内容"""
    prompt_path = Path(__file__).parents[2] / "prompts" / f"{name}.prompt"
    return prompt_path.read_text(encoding="utf-8")
