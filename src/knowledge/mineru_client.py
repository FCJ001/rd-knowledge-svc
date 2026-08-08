# ============================================================
# MinerU 文档解析客户端
#
# 与天宫医疗配置一致：
#   MINERU_API_URL=http://117.50.195.135:8000
#   MINERU_BACKEND=hybrid-auto-engine
#   MINERU_TIMEOUT=300
#
# ★ 异步提交流程：POST /tasks → 轮询 → GET /tasks/{id}/result
# ★ 保留页码修复：从 blocks 取真实 page_number
# ★ return_images=True 时提取 base64 图片 → bytes
# ============================================================

from __future__ import annotations

import asyncio
import base64
import re
from pathlib import Path

import httpx
from loguru import logger

from src.core.config import get_settings

settings = get_settings()


async def parse_document(
    file_path: str,
    file_name: str | None = None,
    backend: str | None = None,
    return_images: bool = True,
    formula_enable: bool = True,
) -> tuple[str, list[int], dict[str, bytes]]:
    """
    调用 MinerU API 解析文档。
    优先使用异步接口（POST /tasks -> 轮询），超大文件不会阻塞。
    Returns: (markdown_text, page_numbers_per_block, images_dict)
        images_dict: key=文件名, value=图片 bytes

    formula_enable: True=公式输出 LaTeX 文本（可检索/可引用）；
        False=公式输出为原图（保真对照用），用于双通道取公式原图。
    """
    base_url = settings.MINERU_API_URL
    backend = backend or settings.MINERU_BACKEND
    timeout = settings.MINERU_TIMEOUT

    if not file_name:
        file_name = Path(file_path).name

    file_bytes = Path(file_path).read_bytes()

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{base_url}/tasks",
            files={"files": (file_name, file_bytes)},
            data={
                "backend": backend,
                "return_md": "true",
                "return_images": str(return_images).lower(),
                "formula_enable": str(formula_enable).lower(),
                "table_enable": "true",
            },
        )
        resp.raise_for_status()
        task_data = resp.json()
        task_id = task_data.get("task_id")

        if not task_id:
            logger.warning("MinerU 未返回 task_id，尝试同步解析")
            return await _parse_sync(file_path, file_name, backend, return_images, formula_enable)

        for _ in range(timeout // 2):
            await asyncio.sleep(2)
            status_resp = await client.get(f"{base_url}/tasks/{task_id}")
            status_resp.raise_for_status()
            status_data = status_resp.json()
            status = status_data.get("status", "")

            if status == "completed":
                result_resp = await client.get(f"{base_url}/tasks/{task_id}/result")
                if result_resp.status_code == 200:
                    return _extract_result(result_resp.json())
                break
            elif status == "failed":
                error = status_data.get("error", "未知错误")
                raise RuntimeError(f"MinerU 解析失败: {error}")

        raise TimeoutError(f"MinerU 解析超时 ({timeout}s)")


async def _parse_sync(
    file_path: str, file_name: str, backend: str, return_images: bool = True,
    formula_enable: bool = True,
) -> tuple[str, list[int], dict[str, bytes]]:
    """同步解析（兜底方案）"""
    base_url = settings.MINERU_API_URL
    file_bytes = Path(file_path).read_bytes()

    async with httpx.AsyncClient(timeout=settings.MINERU_TIMEOUT) as client:
        resp = await client.post(
            f"{base_url}/file_parse",
            files={"files": (file_name, file_bytes)},
            data={
                "backend": backend,
                "return_md": "true",
                "return_images": str(return_images).lower(),
                "formula_enable": str(formula_enable).lower(),
                "table_enable": "true",
            },
        )
        resp.raise_for_status()
        return _extract_result(resp.json())


def _extract_result(result: dict) -> tuple[str, list[int], dict[str, bytes]]:
    """从 MinerU 响应中提取 Markdown + 页码 + 图片（兼容新旧 API 格式）"""
    results = result.get("results", {})
    md = ""
    pages: list[int] = []
    images: dict[str, bytes] = {}

    # v3.4.4+（protocol v2）: results 是 dict[str, dict]，key=文件名
    if isinstance(results, dict):
        for file_data in results.values():
            if isinstance(file_data, dict):
                md = file_data.get("md_content", "") or file_data.get("md", "") or ""
                if md:
                    # 提取图片（base64 data URI → bytes）
                    raw_images = file_data.get("images", {})
                    if isinstance(raw_images, dict):
                        for img_name, img_data in raw_images.items():
                            try:
                                images[img_name] = _decode_image(img_data)
                            except Exception as e:
                                logger.warning(f"图片解码失败 {img_name}: {e}")
                    return md, pages, images
        return "", [], {}

    # 旧版: results 是 list[dict]，有 md/blocks/images 字段
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, dict):
            md = first.get("md", "") or first.get("md_content", "") or ""
            blocks = first.get("blocks", [])
            pages = [b.get("page_number", 0) for b in blocks] if blocks else []
            raw_images = first.get("images", {})
            if isinstance(raw_images, dict):
                for img_name, img_data in raw_images.items():
                    try:
                        images[img_name] = _decode_image(img_data)
                    except Exception as e:
                        logger.warning(f"图片解码失败 {img_name}: {e}")
            return md, pages, images

    # 兜底: result 本身包含 md
    if "md" in result:
        return result["md"], [], {}
    if "md_content" in result:
        return result["md_content"], [], {}

    import json
    return json.dumps(result, ensure_ascii=False), [], {}


def _decode_image(img_data) -> bytes:
    """解码 MinerU 返回的图片数据（支持 base64 data URI 和纯 base64 字符串）"""
    if isinstance(img_data, bytes):
        return img_data
    if isinstance(img_data, str):
        # data:image/jpeg;base64,xxx → xxx
        match = re.match(r"data:image/\w+;base64,(.+)", img_data, re.DOTALL)
        if match:
            return base64.b64decode(match.group(1))
        return base64.b64decode(img_data)
    return img_data


async def check_mineru_health() -> dict:
    """检查 MinerU 服务健康状态"""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{settings.MINERU_API_URL}/health")
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}
