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
    acl_expr: str | None,
    top_k: int = 20,
    filters: dict | None = None,
    extra_dense_queries: list[list[float]] | None = None,
) -> list[dict]:
    """Dense + BM25 混合检索（RRF 融合）。

    extra_dense_queries: 额外 dense 查询向量（如 HyDE 假设文档向量），
    每个向量独立成为一个 COSINE AnnSearchRequest，参与同一 RRF 融合，
    用于 doc 通道内部的多路并行召回（原始查询 / HyDE / BM25）。

    acl_expr: 权限谓词（见 knowledge/acl.py），并进每个 AnnSearchRequest 的
    布尔表达式。★ 无默认值：必须显式给出——None 的语义是"已确认本次查询
    无需 ACL 约束"（admin 或 DOC_ACL_ENABLED=false），不是"忘了传"。
    RRF 融合发生在过滤之后，所以无权内容既不占 top_k，也不会经融合进入结果。
    """
    filter_expr = _and_expr(_build_filter(filters) if filters else "", acl_expr)

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


def _and_expr(*exprs: str | None) -> str:
    """用 and 连接非空表达式，逐项加括号。

    ★ 必须加括号：ACL 谓词本身是函数调用（array_contains_any(...)），
    与 `a == "b" and c == "d"` 直接拼接时运算符优先级不可依赖——
    少一层括号就可能让 ACL 谓词被 and 的另一项吃掉（过滤静默失效）。
    """
    parts = [f"({e})" for e in exprs if e]
    return " and ".join(parts)
