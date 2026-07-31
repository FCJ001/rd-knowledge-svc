# ============================================================
# Node ②c — LLM 扩展关键词 + Milvus 向量检索相关指标
# ============================================================

import json

from langchain_core.messages import HumanMessage, SystemMessage

from src.nl2sql.state import DataAgentState
from src.nl2sql.context import DataAgentContext
from src.nl2sql.prompt_loader import load_prompt


async def recall_metrics(state: DataAgentState, ctx: DataAgentContext) -> dict:
    """LLM 扩展关键词 → 向量化 → Milvus 检索指标"""
    writer = ctx.get("writer")
    if writer:
        writer({"type": "progress", "step": "召回指标", "status": "running"})

    try:
        llm = ctx["llm"]
        embedding_model = ctx["embedding_model"]
        repo = ctx["milvus_metric_repo"]
        keywords = state["keywords"]

        # LLM 扩展关键词
        response = await llm.ainvoke([
            SystemMessage(content=load_prompt("extend_keywords_for_metric_recall")),
            HumanMessage(content=state["query"]),
        ])
        try:
            extra_keywords = json.loads(response.content.strip())
            if not isinstance(extra_keywords, list):
                extra_keywords = []
        except json.JSONDecodeError:
            extra_keywords = []

        all_keywords = list(dict.fromkeys(keywords + extra_keywords))

        # 向量检索
        retrieved: dict[str, any] = {}
        for kw in all_keywords:
            try:
                vec = embedding_model.embed_query(kw)
                metrics = repo.search(vec, top_k=5, threshold=0.6)
                for m in metrics:
                    if m.id not in retrieved:
                        retrieved[m.id] = m
            except Exception:
                continue

        from src.core.logger import logger
        logger.info(f"[recall_metrics] 扩展关键词={extra_keywords}, 向量检索命中 {len(retrieved)} 个指标")

        if writer:
            writer({"type": "progress", "step": "召回指标", "status": "success"})
        return {"retrieved_metrics": list(retrieved.values())}
    except Exception as e:
        if writer:
            writer({"type": "progress", "step": "召回指标", "status": "error"})
        raise
