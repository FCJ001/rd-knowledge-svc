# ============================================================
# 纯稠密向量检索
# ★ 同步 gRPC 调用 to_thread 下放，不阻塞事件循环
# ============================================================

import asyncio

from pymilvus import MilvusClient

from src.infra.milvus_client import escape_milvus_string


async def vector_search(
    milvus: MilvusClient,
    collection_name: str,
    embedding: list[float],
    top_k: int = 20,
    filters: dict | None = None,
) -> list[dict]:
    filter_expr = _build_filter(filters) if filters else None
    results = await asyncio.to_thread(
        milvus.search,
        collection_name=collection_name,
        data=[embedding],
        anns_field="embedding",
        limit=top_k,
        output_fields=["text", "doc_name", "doc_type", "category", "chunk_index", "page_number", "parent_text", "image_urls"],
        filter=filter_expr,
        search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
    )
    hits = []
    for hit in results[0]:
        hits.append({
            "text": hit["entity"]["text"],
            "parent_text": hit["entity"].get("parent_text", ""),
            "doc_name": hit["entity"]["doc_name"],
            "doc_type": hit["entity"]["doc_type"],
            "score": hit["distance"],
            "chunk_index": hit["entity"]["chunk_index"],
            "page_number": hit["entity"].get("page_number", 0),
            "image_urls": hit["entity"].get("image_urls", ""),
        })
    return hits


def _build_filter(filters: dict) -> str:
    """构造 Milvus 布尔表达式。值一律经过转义，防表达式注入。"""
    parts = []
    for key, value in filters.items():
        if isinstance(value, list):
            values_str = ", ".join(f'"{escape_milvus_string(v)}"' for v in value)
            parts.append(f'{key} in [{values_str}]')
        else:
            parts.append(f'{key} == "{escape_milvus_string(value)}"')
    return " and ".join(parts)
