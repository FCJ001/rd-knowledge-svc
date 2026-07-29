# ============================================================
# DocRAG：文档向量检索 + HyDE + Rerank
# ★ 新增 model_code 过滤（跨车型串味是安全事故）
# ============================================================

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from loguru import logger
from pymilvus import MilvusClient

from src.knowledge.prompts import DOC_QA_PROMPT

COLLECTION_NAME = "alm_docs"


async def search_docs_raw(
    question: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    top_k: int = 20,
    rerank_top_k: int = 5,
    doc_type: str | None = None,
    model_code: str | None = None,  # ★ 新增
    llm: BaseChatModel | None = None,
    use_hyde: bool = False,
) -> list[dict]:
    """文档向量检索，返回原始结果列表"""

    if use_hyde and llm is not None:
        from src.knowledge.hyde import generate_hyde_embedding
        query_vec = await generate_hyde_embedding(question, llm, embedding_model)
    else:
        query_vec = await embedding_model.aembed_query(question)

    # 构建过滤表达式
    filter_parts = []
    if doc_type:
        filter_parts.append(f'doc_type == "{doc_type}"')
    if model_code:
        filter_parts.append(f'model_code == "{model_code}"')  # ★
    filter_expr = " and ".join(filter_parts) if filter_parts else None
    logger.info(
        f"DocRAG 检索: top_k={top_k} hyde={use_hyde} "
        f"filter={filter_expr or '无'} model_code={model_code or '不限'}"
    )

    try:
        results = milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vec],
            limit=top_k,
            output_fields=["doc_name", "doc_type", "page_number", "chunk_index", "text", "parent_text", "image_urls"],
            anns_field="embedding",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
            filter=filter_expr,
        )
    except Exception as e:
        logger.warning(f"文档检索失败: {e}")
        return []

    if not results or not results[0]:
        logger.info("DocRAG 召回: 0 条")
        return []

    hits = [
        {**hit["entity"], "score": hit.get("distance", 0.0)}
        for hit in results[0]
    ]
    logger.info(
        f"DocRAG 召回: {len(hits)} 条, top_score={hits[0]['score']:.4f}, "
        f"来源: {list(set(h['doc_name'] for h in hits[:5]))}"
    )

    from src.knowledge.reranker import rerank_docs
    reranked = await rerank_docs(question, hits, top_k=rerank_top_k)
    logger.info(
        f"DocRAG 精排后: {len(reranked)} 条 (rerank_top_k={rerank_top_k}), "
        f"top_score={reranked[0].get('rerank_score', reranked[0]['score']):.4f}"
    )
    return reranked


def format_doc_context(hits: list[dict]) -> str:
    """格式化检索结果为 LLM 可读上下文。
    parent_child 策略时优先使用 parent_text（完整父块），
    否则使用 text（子块）。"""
    if not hits:
        return ""
    parts = []
    for i, hit in enumerate(hits, 1):
        source = f"[{hit['doc_name']}, 第{hit.get('page_number', '?')}页]"
        # parent_child 策略：用父块完整内容
        text = hit.get("parent_text") or hit["text"]
        parts.append(f"片段{i} {source}:\n{text}")
    return "\n\n---\n\n".join(parts)


def extract_image_urls(hits: list[dict]) -> list[str]:
    """从检索结果中提取所有图片 URL（去重）"""
    import json
    urls = []
    seen = set()
    for hit in hits:
        raw = hit.get("image_urls", "")
        if not raw:
            continue
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            for url in parsed:
                if url not in seen:
                    seen.add(url)
                    urls.append(url)
        except (json.JSONDecodeError, TypeError):
            pass
    return urls


async def search_docs(
    question: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    llm: BaseChatModel,
    top_k: int = 20,
    rerank_top_k: int = 5,
    doc_type: str | None = None,
    model_code: str | None = None,
    role: str = "engineer",
    use_hyde: bool = True,
) -> str:
    """DocRAG 完整流程：检索 + 精排 + HyDE + 生成回答"""
    hits = await search_docs_raw(
        question, embedding_model, milvus_client,
        top_k=top_k, rerank_top_k=rerank_top_k,
        doc_type=doc_type, model_code=model_code,
        llm=llm, use_hyde=use_hyde,
    )
    if not hits:
        return "当前知识库中未找到与您问题相关的文档内容。"

    context = format_doc_context(hits)
    prompt = DOC_QA_PROMPT.format(question=question, context=context, role=role)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return response.content
