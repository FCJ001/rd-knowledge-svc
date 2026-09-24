# ============================================================
# MinerU 文档解析客户端
#
# 与天宫医疗配置一致：
#   MINERU_API_URL=http://localhost:8000
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
import json
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
    return_content_list: bool = False,
) -> tuple[str, list[int], dict[str, bytes], list[dict], list[dict]]:
    """
    调用 MinerU API 解析文档。
    优先使用异步接口（POST /tasks -> 轮询），超大文件不会阻塞。
    Returns: (markdown_text, page_numbers_per_block, images_dict, table_blocks, equation_blocks)
        images_dict: key=文件名, value=图片 bytes
        table_blocks: return_content_list=True 时从 content_list 提取的表格块，
            [{page_idx(0-based), bbox[x0,y0,x1,y1](归一化坐标系), table_body(HTML|None)}]
        equation_blocks: 公式块 [{page_idx, bbox(归一化), text(LaTeX $$..$$)}]

    formula_enable: True=公式输出 LaTeX 文本（可检索/可引用）。
    return_content_list: True 时请求 content_list，用于表格/公式原图定位（bbox+页码）。
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
                "return_content_list": str(return_content_list).lower(),
            },
        )
        resp.raise_for_status()
        task_data = resp.json()
        task_id = task_data.get("task_id")

        if not task_id:
            logger.warning("MinerU 未返回 task_id，尝试同步解析")
            return await _parse_sync(
                file_path, file_name, backend, return_images, formula_enable,
                return_content_list,
            )

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
    formula_enable: bool = True, return_content_list: bool = False,
) -> tuple[str, list[int], dict[str, bytes], list[dict], list[dict]]:
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
                "return_content_list": str(return_content_list).lower(),
            },
        )
        resp.raise_for_status()
        return _extract_result(resp.json())


def _block_anchor_text(b: dict) -> str:
    """取一个 content_list 块用于定位的文本片段（不同 type 的键不同）。"""
    for key in ("text", "table_body", "img_caption", "table_caption", "code_body"):
        v = b.get(key)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, list):  # img_caption 等可能是列表
            joined = " ".join(str(x) for x in v if x)
            if joined.strip():
                return joined
    return ""


def _page_anchors_from_content_list(raw) -> list[tuple[int, str]]:
    """从 content_list 构造页码锚点：按文档顺序的 (页码, 文本片段)。

    ★ 页码统一为 1-based（MinerU 的 page_idx 是 0-based，这里 +1）。
    0 保留给"页码未知"——若直接用 page_idx，首页的 0 会与"未知"撞值，
    下游无法区分"第 1 页"和"没记录到页码"。
    """
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    anchors: list[tuple[int, str]] = []
    for b in raw if isinstance(raw, list) else []:
        if not isinstance(b, dict):
            continue
        text = _block_anchor_text(b)
        if len(text) < 8:      # 太短的片段无法可靠定位
            continue
        page = int(b.get("page_idx", -1) or 0) + 1
        anchors.append((page, text))
    return anchors


def _extract_result(
    result: dict,
) -> tuple[str, list[int], dict[str, bytes], list[dict], list[dict], list[tuple[int, str]]]:
    """从 MinerU 响应中提取 Markdown + 页码锚点 + 图片 + 表格块 + 公式块。

    返回 (md, pages, images, table_blocks, equation_blocks, page_anchors)。

    `pages` 是历史字段（按块顺序的页码，下标空间是「块」），保留仅为兼容；
    **页码请用 page_anchors**——它带文本片段，可把 chunk 定位回页码。
    用 `pages[chunk_idx]` 取页码是错的：块下标与切片下标不是同一个空间，
    一个块可能被切成多个 chunk，一个 chunk 也可能横跨多个块。
    """
    results = result.get("results", {})
    md = ""
    pages: list[int] = []
    images: dict[str, bytes] = {}
    table_blocks: list[dict] = []
    equation_blocks: list[dict] = []

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
                    content_list = file_data.get("content_list")
                    table_blocks, equation_blocks = _parse_content_list(content_list)
                    page_anchors = _page_anchors_from_content_list(content_list)
                    if not page_anchors:
                        logger.warning(
                            "MinerU 响应缺少可用的 content_list 页码锚点：本次入库所有 chunk "
                            "的 page_number 将为 0（引用不显示页码）。请检查是否请求了 "
                            "return_content_list，以及 content_list 是否带 page_idx"
                        )
                    return md, pages, images, table_blocks, equation_blocks, page_anchors
        return "", [], {}, [], [], []

    # 旧版: results 是 list[dict]，有 md/blocks/images 字段
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, dict):
            md = first.get("md", "") or first.get("md_content", "") or ""
            blocks = first.get("blocks", [])
            pages = [int(b.get("page_number", 0) or 0) for b in blocks] if blocks else []
            page_anchors = [
                (int(b.get("page_number", 0) or 0) + 1, _block_anchor_text(b))
                for b in blocks if isinstance(b, dict)
            ]
            page_anchors = [(pg, t) for pg, t in page_anchors if len(t) >= 8]
            raw_images = first.get("images", {})
            if isinstance(raw_images, dict):
                for img_name, img_data in raw_images.items():
                    try:
                        images[img_name] = _decode_image(img_data)
                    except Exception as e:
                        logger.warning(f"图片解码失败 {img_name}: {e}")
            return md, pages, images, [], [], page_anchors

    # 兜底: result 本身包含 md
    if "md" in result:
        return result["md"], [], {}, [], [], []
    if "md_content" in result:
        return result["md_content"], [], {}, [], [], []

    return json.dumps(result, ensure_ascii=False), [], {}, [], [], []


def _parse_content_list(raw) -> tuple[list[dict], list[dict]]:
    """从 content_list（JSON 字符串或 list）提取表格块与公式块。

    返回 (table_blocks, equation_blocks)
    table_blocks: [{page_idx, bbox, table_body: str|None}]
        table_body=None 表示该块表格 HTML 解析失败——实测跨页表格的续页块
        就是 body=None（被 MinerU 合并进前一块的 <table>）。
    equation_blocks: [{page_idx, bbox, text(LaTeX $$..$$)}]
        type=equation 的块，bbox 与表格同一归一化坐标系，text 是完整公式文本。"""
    if not raw:
        return [], []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return [], []
    tables: list[dict] = []
    equations: list[dict] = []
    for b in raw if isinstance(raw, list) else []:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "table":
            tables.append(
                {
                    "page_idx": int(b.get("page_idx", 0) or 0),
                    "bbox": b.get("bbox"),
                    "table_body": b.get("table_body"),
                }
            )
        elif b.get("type") == "equation":
            equations.append(
                {
                    "page_idx": int(b.get("page_idx", 0) or 0),
                    "bbox": b.get("bbox"),
                    "text": b.get("text"),
                }
            )
    return tables, equations


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
