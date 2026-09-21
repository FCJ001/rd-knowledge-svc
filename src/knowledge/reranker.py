# ============================================================
# 后检索精排（Rerank）
# RERANK_PROVIDER=deepseek（默认）：DeepSeek LLM 清单式重排（DeepSeek 无专用 rerank 端点）
# RERANK_PROVIDER=dashscope：qwen3-rerank 专用重排模型（保留的可选后端）
# RERANK_PROVIDER=off：不做精排，直接用 RRF 融合序
# 降级链：空→[] / len<=k→透传 / 超时或异常→docs[:k]（融合序）
# ============================================================

from __future__ import annotations

import asyncio
import json

from loguru import logger

from src.api.deps import get_llm
from src.core.config import get_settings

settings = get_settings()

# 断崖截断常量（与 dataset_rag node_rerank.py 一致）
RERANK_MAX_TOPK: int = 10   # 动态 TopK 硬上限
RERANK_MIN_TOPK: int = 1    # 动态 TopK 硬下限
RERANK_GAP_RATIO: float = 0.25  # 相对断崖阈值
RERANK_GAP_ABS: float = 0.5     # 绝对断崖阈值

# LLM 清单式重排：候选文本截断长度与候选数上限（控 prompt 体积与延迟）
_LLM_RERANK_SNIPPET_CHARS: int = 600
_LLM_RERANK_MAX_CANDIDATES: int = 20

_LLM_RERANK_PROMPT = (
    "你是搜索结果重排器。给定用户查询和候选段落列表（每段以 [编号] 开头），"
    "请按与查询的相关性从高到低排序，并给每个候选一个 0 到 1 的相关性分数"
    "（1=直接回答查询，0=完全无关）。\n"
    '仅输出一个 JSON 对象，格式为 {{"results": [{{"index": 编号, "score": 分数}}, ...]}}，'
    "必须覆盖全部候选编号，不要输出任何其他内容。\n\n"
    "用户查询：{query}\n\n候选段落：\n{candidates}"
)


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


def _parse_llm_rerank(content: str, n_docs: int) -> list[tuple[int, float]]:
    """解析 LLM 重排输出为 [(index, score)]；越界/重复/非法条目直接丢弃。"""
    data = json.loads(content)
    results = data.get("results") if isinstance(data, dict) else data
    parsed: list[tuple[int, float]] = []
    seen: set[int] = set()
    for item in results or []:
        if not isinstance(item, dict) or "index" not in item:
            continue
        try:
            idx = int(item["index"])
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= n_docs or idx in seen:
            continue
        seen.add(idx)
        parsed.append((idx, max(0.0, min(1.0, score))))
    return parsed


async def _rerank_llm(
    query: str,
    documents: list[dict],
    top_k: int,
    use_dynamic_topk: bool,
) -> list[dict]:
    """DeepSeek LLM 清单式重排：一次调用对全部候选给出相关性分数并排序。"""
    texts = [doc.get("text", "")[:_LLM_RERANK_SNIPPET_CHARS] for doc in documents]
    shown = texts[:_LLM_RERANK_MAX_CANDIDATES]
    prompt = _LLM_RERANK_PROMPT.format(
        query=query,
        candidates="\n".join(f"[{i}] {t}" for i, t in enumerate(shown)),
    )

    # response_format=json_object：DeepSeek 原生支持；prompt 已含 JSON 字样
    llm = get_llm().bind(response_format={"type": "json_object"})
    message = await asyncio.wait_for(llm.ainvoke(prompt), timeout=settings.RERANK_TIMEOUT)

    parsed = _parse_llm_rerank(message.content, len(documents))
    if not parsed:
        logger.warning("LLM 重排输出解析为空，回退融合序")
        return documents[:top_k]

    reranked = []
    for idx, score in sorted(parsed, key=lambda p: p[1], reverse=True):
        doc = documents[idx].copy()
        doc["rerank_score"] = score
        reranked.append(doc)
    # LLM 漏掉的候选按原相对顺序排在末尾
    scored_idx = {idx for idx, _ in parsed}
    for idx, doc in enumerate(documents):
        if idx not in scored_idx:
            doc = doc.copy()
            doc["rerank_score"] = 0.0
            reranked.append(doc)

    # 分数断崖动态截断只在候选全部被 LLM 打分时启用（缺分候选的 0.0 会制造假断崖）
    if use_dynamic_topk and len(parsed) == len(documents):
        keep = _cliff_topk([d["rerank_score"] for d in reranked])
        logger.info(f"Rerank(LLM) 动态TopK: 截断至 {keep} 条 (候选 {len(reranked)} 条)")
        return reranked[:keep]

    return reranked[:top_k]


async def _rerank_dashscope(
    query: str,
    documents: list[dict],
    top_k: int,
    use_dynamic_topk: bool,
    rerank_top_n: int,
) -> list[dict]:
    """qwen3-rerank 专用重排模型（DashScope）。"""
    import dashscope
    from dashscope import TextReRank

    dashscope.api_key = settings.DASHSCOPE_API_KEY
    texts = [doc.get("text", "") for doc in documents]

    # ★ dashscope 是同步 HTTP 客户端：to_thread 下放 + 整体超时，
    #   否则一次精排挂起就冻结事件循环数秒~数十秒
    response = await asyncio.wait_for(
        asyncio.to_thread(
            TextReRank.call,
            model="qwen3-rerank",
            query=query,
            documents=texts,
            top_n=rerank_top_n,
            return_documents=False,
        ),
        timeout=settings.RERANK_TIMEOUT,
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


async def rerank_docs(
    query: str,
    documents: list[dict],
    top_k: int = 5,
    use_dynamic_topk: bool | None = None,
) -> list[dict]:
    """按 RERANK_PROVIDER 分发精排。

    use_dynamic_topk: None 时读 settings.RAG_DYNAMIC_TOPK。
    True 时按 rerank_score 断崖动态截断（最多 RERANK_MAX_TOPK 条）。
    任何失败回退 documents[:top_k]（RRF 融合序）。
    """
    if not documents:
        return []

    if len(documents) <= top_k:
        return documents

    if use_dynamic_topk is None:
        use_dynamic_topk = settings.RAG_DYNAMIC_TOPK
    rerank_top_n = RERANK_MAX_TOPK if use_dynamic_topk else top_k

    provider = settings.RERANK_PROVIDER.strip().lower()
    try:
        if provider == "off":
            return documents[:top_k]
        if provider == "dashscope":
            return await _rerank_dashscope(query, documents, top_k, use_dynamic_topk, rerank_top_n)
        return await _rerank_llm(query, documents, top_k, use_dynamic_topk)
    except Exception as e:
        logger.warning(f"Reranker 异常，回退到融合序排序: {e}")
        return documents[:top_k]
