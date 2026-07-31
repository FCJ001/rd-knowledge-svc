# ============================================================
# Node ②a — LLM 扩展关键词 + Milvus 向量检索相关列
# ============================================================

import json

from langchain_core.messages import HumanMessage, SystemMessage

from src.nl2sql.state import DataAgentState
from src.nl2sql.context import DataAgentContext
from src.nl2sql.prompt_loader import load_prompt


async def recall_columns(state: DataAgentState, ctx: DataAgentContext) -> dict:
    """LLM 扩展关键词 → 向量化 → Milvus 检索列"""
    writer = ctx.get("writer")
    if writer:
        writer({"type": "progress", "step": "召回字段", "status": "running"})

    try:
        llm = ctx["llm"]
        embedding_model = ctx["embedding_model"]
        repo = ctx["milvus_column_repo"]
        keywords = state["keywords"]

        # LLM 扩展关键词
        response = await llm.ainvoke([
            SystemMessage(content=load_prompt("extend_keywords_for_column_recall")),
            HumanMessage(content=state["query"]),
        ])
        try:
            extra_keywords = json.loads(response.content.strip())
            if not isinstance(extra_keywords, list):
                extra_keywords = []
        except json.JSONDecodeError:
            extra_keywords = []

        # 合并关键词
        all_keywords = list(dict.fromkeys(keywords + extra_keywords))

        # 向量检索
        retrieved: dict[str, any] = {}
        for kw in all_keywords:
            try:
                vec = embedding_model.embed_query(kw)
                cols = repo.search(vec, top_k=5, threshold=0.6)
                for c in cols:
                    if c.id not in retrieved:
                        retrieved[c.id] = c
            except Exception:
                continue

        from src.core.logger import logger
        logger.info(f"[recall_columns] 扩展关键词={extra_keywords}, 向量检索命中 {len(retrieved)} 列")

        if writer:
            writer({"type": "progress", "step": "召回字段", "status": "success"})
        return {"retrieved_columns": list(retrieved.values())}
    except Exception as e:
        if writer:
            writer({"type": "progress", "step": "召回字段", "status": "error"})
        raise
