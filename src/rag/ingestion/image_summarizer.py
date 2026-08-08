# ============================================================
# 图片 VL 摘要：qwen-vl-max 概括技术文档图片 → 描述写入 markdown alt
# ★ fail-open：任何异常 / 非 200 返回空串，不阻塞入库
# ============================================================

from __future__ import annotations

import asyncio
import base64

from loguru import logger

from src.core.config import get_settings

settings = get_settings()

SUMMARY_PROMPT = (
    "请用一句中文概括这张汽车技术文档图片的关键内容，"
    "仅输出描述本身，不要多余解释。"
)


async def summarize_image(
    img_bytes: bytes,
    mime_type: str,
    model: str | None = None,
) -> str:
    """调用 VL 模型生成图片摘要；失败返回空串（fail-open）。"""
    try:
        import dashscope
        from dashscope import MultiModalConversation

        dashscope.api_key = settings.DASHSCOPE_API_KEY
        model = model or settings.VL_MODEL

        b64 = base64.b64encode(img_bytes).decode()
        image = f"data:{mime_type};base64,{b64}"
        messages = [{
            "role": "user",
            "content": [{"image": image}, {"text": SUMMARY_PROMPT}],
        }]

        # MultiModalConversation.call 是同步调用，用 to_thread 避免阻塞事件循环
        response = await asyncio.to_thread(
            MultiModalConversation.call,
            model=model,
            messages=messages,
        )

        if response.status_code != 200:
            logger.warning(f"VL 摘要调用失败: {response.message}")
            return ""

        output = response.output or {}
        text = ""
        for item in output.get("choices") or []:
            content = (item.get("message") or {}).get("content")
            if isinstance(content, str):
                text += content
            elif isinstance(content, list):
                for seg in content:
                    if isinstance(seg, dict) and seg.get("text"):
                        text += seg["text"]
        return text.strip()

    except Exception as e:
        logger.warning(f"VL 摘要异常（fail-open）: {e}")
        return ""
