# ============================================================
# 图片 VL 摘要：DeepSeek 视觉模型（VL_MODEL，默认 deepseek-flash）
# 概括技术文档图片 → 描述写入 markdown alt / 公式原图转 LaTeX
# OpenAI 兼容 image_url 协议（base64 data URL；DeepSeek 只允许图片出现在 user 消息）
# ★ fail-open：任何异常 / 空响应返回空串，不阻塞入库
# ============================================================

import asyncio
import base64
import json
import re
from functools import lru_cache

from loguru import logger
from openai import AsyncOpenAI

from src.core.config import get_settings

# VL 调用并发上限：批量入库时多 worker/多图片并发受限，防上游限流 429
_VL_SEM = asyncio.Semaphore(max(1, get_settings().VL_CONCURRENCY))

SUMMARY_PROMPT = (
    "请用一句中文概括这张汽车技术文档图片的关键内容，"
    "仅输出描述本身，不要多余解释。"
)

FORMULA_PROMPT = (
    "这是一张公式原图（识别对照用）。请仔细辨认图片中的公式，"
    "用 LaTeX 语法把公式完整读出（如 R = \\frac{U_{max}}{I}）。"
    "若 LaTeX 无法表示，用中文文字说明公式含义。仅输出公式本身，不要多余解释。"
)

TABLE_PROMPT = (
    "这是一张文档中的表格原图。请仔细辨认表格内容，严格输出 JSON 对象（不要输出任何其他内容）："
    '{"summary": "一句话中文概括这张表格讲什么，保留关键数值（如扭矩、电压、参数等），供搜索用",'
    ' "markdown": "从原图转录的 Markdown 表格，合并单元格展开到每行；'
    '无法辨认的单元格填 null"}'
)


@lru_cache
def _vl_client() -> AsyncOpenAI:
    s = get_settings()
    return AsyncOpenAI(
        api_key=s.vl_api_key,
        base_url=s.VL_BASE_URL,
        timeout=s.VL_TIMEOUT,
        max_retries=1,  # 入库批量图片：快速失败优于长重试
    )


async def summarize_image(
    img_bytes: bytes,
    mime_type: str,
    model: str | None = None,
    context: str = "",
    prompt: str | None = None,
) -> str:
    """调用 VL 模型生成图片摘要；失败返回空串（fail-open）。

    context: 图片在文档中出现的上下文（如所在段落），可辅助模型理解图片
    主题，使描述带上车型/部件名等可检索的专业词。
    prompt: 自定义读取提示词（如公式原图用 FORMULA_PROMPT 读公式内容）。"""
    try:
        s = get_settings()
        model = model or s.VL_MODEL

        prompt = prompt or SUMMARY_PROMPT
        if context:
            prompt += f"\n\n该图片在文档中出现的上下文（辅助理解，请结合图片内容）：\n{context}"

        b64 = base64.b64encode(img_bytes).decode()
        data_url = f"data:{mime_type};base64,{b64}"
        async with _VL_SEM:
            response = await _vl_client().chat.completions.create(
                model=model,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )

        if not response.choices:
            logger.warning("VL 摘要返回空 choices")
            return ""
        text = (response.choices[0].message.content or "").strip()
        if not text:
            logger.warning("VL 摘要返回为空")
        return text

    except Exception as e:
        logger.warning(f"VL 摘要异常（fail-open）: {e}")
        return ""


def _strip_code_fence(raw: str) -> str:
    """剥掉 LLM 可能加的 ```json 围栏"""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    return cleaned


async def summarize_table(img_bytes: bytes, mime_type: str = "image/png") -> tuple[str, str]:
    """表格原图 → (一句话语义摘要, 净化 Markdown 表格转录)。

    复用图片摘要的 VL 客户端与并发限流；任何异常返回 ("", "")（fail-open，
    调用方保留 MinerU 原始 HTML 与原图，不受影响）。"""
    try:
        raw = await summarize_image(img_bytes, mime_type, prompt=TABLE_PROMPT)
        if not raw:
            return "", ""
        data = json.loads(_strip_code_fence(raw))
        summary = str(data.get("summary") or "").strip()
        md = str(data.get("markdown") or "").strip()
        return summary, md
    except Exception as e:
        logger.warning(f"表格 VL 校读失败（fail-open）: {e}")
        return "", ""
