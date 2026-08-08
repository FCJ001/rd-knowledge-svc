# ============================================================
# DashScope Reranker（qwen3-rerank）— 后检索精排
# 4 层降级：空→[] / len<=k→透传 / ImportError→docs[:k] / 非200→docs[:k]
# ============================================================

from __future__ import annotations

from loguru import logger

from src.core.config import get_settings

settings = get_settings()

# 断崖截断常量（与 dataset_rag node_rerank.py 一致）
RERANK_MAX_TOPK: int = 10   # 动态 TopK 硬上限
RERANK_MIN_TOPK: int = 1    # 动态 TopK 硬下限
RERANK_GAP_RATIO: float = 0.25  # 相对断崖阈值
RERANK_GAP_ABS: float = 0.5     # 绝对断崖阈值


def _cliff_topk(
    scores: list[float],
    min_topk: int = RERANK_MIN_TOPK,
    max_topk: int = RERANK_MAX_TOPK,
    gap_abs: float = RERANK_GAP_ABS,
    gap_ratio: float = RERANK_GAP_RATIO,
) -> int:
    """基于分数断崖的动态 TopK。

    输入为已按分数降序排列的分数列表，返回应保留的条数：
    - 相邻分差 gap>=gap_abs 或相对下降 gap/(abs(s1)+1e-6)>=gap_ratio 即视为断崖，截断于断崖前；
    - 无断崖取满 max_topk；下限 min_topk。
    """
    if not scores:
        return 0
    max_topk = min(max_topk, len(scores))
    topk = max_topk
    if topk > min_topk:
        for i in range(min_topk - 1, max_topk - 1):
            s1, s2 = scores[i], scores[i + 1]
            gap = s1 - s2
            rel = gap / (abs(s1) + 1e-6)
            if gap >= gap_abs or rel >= gap_ratio:
                logger.info(
                    f"Rerank 断崖 @index={i} (Score {s1:.4f} -> {s2:.4f}, Gap={gap:.4f})"
                )
                topk = i + 1
                break
    return topk


async def rerank_docs(
    query: str,
    documents: list[dict],
    top_k: int = 5,
    use_dynamic_topk: bool | None = None,
) -> list[dict]:
    """DashScope qwen3-rerank 精排。

    use_dynamic_topk: None 时读 settings.RAG_DYNAMIC_TOPK。
    True 时按 rerank_score 断崖动态截断（最多 RERANK_MAX_TOPK 条），
    需请求更多候选（top_n=10）供截断判断。
    """
    if not documents:
        return []

    if len(documents) <= top_k:
        return documents

    if use_dynamic_topk is None:
        use_dynamic_topk = settings.RAG_DYNAMIC_TOPK
    rerank_top_n = RERANK_MAX_TOPK if use_dynamic_topk else top_k

    try:
        import dashscope
        from dashscope import TextReRank

        dashscope.api_key = settings.DASHSCOPE_API_KEY
        texts = [doc.get("text", "") for doc in documents]

        response = TextReRank.call(
            model="qwen3-rerank",
            query=query,
            documents=texts,
            top_n=rerank_top_n,
            return_documents=False,
        )

        if response.status_code != 200:
            logger.warning(f"Reranker 调用失败: {response.message}")
            return documents[:top_k]

        reranked = []
        for item in response.output.results:
            idx = item.index
            doc = documents[idx].copy()
            doc["rerank_score"] = item.relevance_score
            reranked.append(doc)

        reranked.sort(key=lambda d: d["rerank_score"], reverse=True)

        if use_dynamic_topk:
            keep = _cliff_topk([d["rerank_score"] for d in reranked])
            logger.info(f"Rerank 动态TopK: 截断至 {keep} 条 (候选 {len(reranked)} 条)")
            return reranked[:keep]

        return reranked[:top_k]

    except ImportError:
        logger.warning("dashscope 未安装，回退到向量距离排序")
        return documents[:top_k]
    except Exception as e:
        logger.warning(f"Reranker 异常，回退到向量距离排序: {e}")
        return documents[:top_k]
