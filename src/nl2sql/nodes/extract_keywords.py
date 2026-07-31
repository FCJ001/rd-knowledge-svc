# ============================================================
# Node ① — jieba 分词提取关键词
# ============================================================

import jieba.analyse

from src.nl2sql.state import DataAgentState
from src.nl2sql.context import DataAgentContext


async def extract_keywords(state: DataAgentState, ctx: DataAgentContext) -> dict:
    """jieba TF-IDF + 词性过滤提取关键词"""
    writer = ctx.get("writer")
    if writer:
        writer({"type": "progress", "step": "抽取关键字", "status": "running"})

    try:
        query = state["query"]
        keywords = jieba.analyse.extract_tags(
            query,
            topK=10,
            allowPOS=("n", "nr", "ns", "nt", "nz", "v", "vn", "a", "an", "eng", "i", "l"),
        )
        # 始终包含原始 query
        if query not in keywords:
            keywords.insert(0, query)

        from src.core.logger import logger
        logger.info(f"[extract_keywords] jieba 提取 {len(keywords)} 个关键词: {keywords}")

        if writer:
            writer({"type": "progress", "step": "抽取关键字", "status": "success"})
        return {"keywords": keywords}
    except Exception as e:
        if writer:
            writer({"type": "progress", "step": "抽取关键字", "status": "error"})
        raise
