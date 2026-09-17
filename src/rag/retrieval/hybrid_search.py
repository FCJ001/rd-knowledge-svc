# ============================================================
# Dense + BM25 Sparse 混合检索，RRF(k=60) 融合
# ★ 与天宫医疗一致：sparse 向量由 Milvus 2.6 内置 BM25 Function 自动生成
# ★ pymilvus 是同步 gRPC 客户端：必须 to_thread 下放线程池，
#   否则慢查询会冻结整个事件循环（所有并发请求一起卡死）
# ============================================================

import asyncio

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from src.infra.milvus_client import escape_milvus_string


async def hybrid_search(
    milvus: MilvusClient,
    collection_name: str,
    dense_embedding: list[float],
    query_text: str,
    top_k: int = 20,
    filters: dict | None = None,
    extra_dense_queries: list[list[float]] | None = None,
) -> list[dict]:
    """Dense + BM25 混合检索（RRF 融合）。

    extra_dense_queries: 额外 dense 查询向量（如 HyDE 假设文档向量），
    每个向量独立成为一个 COSINE AnnSearchRequest，参与同一 RRF 融合，
    用于 doc 通道内部的多路并行召回（原始查询 / HyDE / BM25）。
    """
    filter_expr = _build_filter(filters) if filters else ""

    dense_req = AnnSearchRequest(
        data=[dense_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=top_k,
        expr=filter_expr,
    )
    sparse_req = AnnSearchRequest(
        data=[query_text],
        anns_field="sparse_embedding",
        param={"metric_type": "BM25"},
        limit=top_k,
        expr=filter_expr,
    )

    extra_reqs = [
        AnnSearchRequest(
            data=[vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            expr=filter_expr,
        )
        for vec in (extra_dense_queries or [])
    ]

    results = await asyncio.to_thread(
        milvus.hybrid_search,
        collection_name=collection_name,
        reqs=[dense_req, sparse_req, *extra_reqs],
        ranker=RRFRanker(k=60),
        limit=top_k,
        output_fields=["text", "doc_name", "doc_type", "category", "page_number", "chunk_index", "parent_text", "image_urls"],
    )

    hits = []
    for hit in results[0]:
        hits.append({
            "text": hit["entity"]["text"],
            "parent_text": hit["entity"].get("parent_text", ""),
            "doc_name": hit["entity"]["doc_name"],
            "doc_type": hit["entity"]["doc_type"],
            "page_number": hit["entity"].get("page_number", ""),
            "score": hit["distance"],
            "chunk_index": hit["entity"]["chunk_index"],
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
