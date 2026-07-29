# ============================================================
# 变更影响预检（轻量版 prescription_review）
# ============================================================

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from loguru import logger
from neo4j import AsyncDriver
from pymilvus import MilvusClient

from src.knowledge.doc_rag import format_doc_context, search_docs_raw
from src.knowledge.prompts import CHANGE_REVIEW_PROMPT


async def review_change(
    change_info: str,
    llm: BaseChatModel,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    neo4j_driver: AsyncDriver,
) -> str:
    """
    变更影响预检。
    查询相关文档 + 图谱 → LLM 分析影响范围。
    """
    # 1. 检索相关文档
    hits = await search_docs_raw(
        question=change_info,
        embedding_model=embedding_model,
        milvus_client=milvus_client,
        top_k=10,
        rerank_top_k=3,
        llm=llm,
        use_hyde=True,
    )
    context = format_doc_context(hits) if hits else "未找到相关文档"

    # 2. 查询图谱关系
    graph_data = "图谱查询跳过（开发期）"
    try:
        from src.knowledge.graph_rag import search_graph_raw
        records = await search_graph_raw(change_info, neo4j_driver, llm)
        if records:
            import json
            graph_data = json.dumps(records, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"图谱查询失败，跳过: {e}")

    prompt = CHANGE_REVIEW_PROMPT.format(
        change_info=change_info,
        context=f"{context}\n\n图谱数据：\n{graph_data}",
    )
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return response.content
