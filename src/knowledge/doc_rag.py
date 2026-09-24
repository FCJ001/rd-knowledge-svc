# ============================================================
# DocRAG：文档向量检索 + HyDE + Rerank
# ★ 新增 model_code 过滤（跨车型串味是安全事故）
# ============================================================

from __future__ import annotations

import asyncio

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from loguru import logger
from pymilvus import MilvusClient

from src.core.config import get_settings
from src.knowledge.acl import doc_acl_expr
from src.knowledge.prompts import DOC_QA_PROMPT

COLLECTION_NAME = "alm_docs"


async def search_docs_with_stages(
    question: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    role: str,
    top_k: int | None = None,
    rerank_top_k: int | None = None,
    doc_type: str | None = None,
    model_code: str | None = None,
    llm: BaseChatModel | None = None,
    use_hyde: bool = False,
    use_dynamic_topk: bool | None = None,
) -> tuple[list[dict], list[dict]]:
    """返回 (raw_hits, final_hits) 两阶段结果，供诊断用。

    raw_hits   = 融合召回后（未精排、未截断），长度 ≤ top_k
    final_hits = 精排 + 断崖截断后，即真正喂给生成的证据

    ★ 为什么需要它：只看 final 无法判断"召回不够"还是"截断过度"——
    两者都表现为证据变少，但对策完全相反。用同一批问题分别测两阶段的
    页级召回率，差值就是截断吃掉的量。生产路径用 search_docs_raw（只取 final）。
    """
    """文档多路召回：原始查询 / HyDE（可选）/ BM25 三路 → RRF 融合 → 精排。

    与单路 dense 检索相比：
    - use_hyde 不再"替代"查询向量，而是作为额外 dense 通道参与 RRF（HyDE 从"替代"变"增加"）；
    - BM25 通道提供词法召回兜底（本项目是内部知识库，用 BM25 代替外部联网检索）；
    - hybrid（RRF）失败自动降级为旧 dense-only 检索，再失败返回 []。

    doc_type / model_code：透传为 Milvus filter 表达式，防止跨文档类型/跨车型串味。
    role：★ 无默认值，必须由调用方显式声明。它决定 ACL 谓词（knowledge/acl.py），
    该谓词并进检索请求本身——不是检索完再过滤，所以两条召回路径（hybrid 与
    dense-only 降级）都必须带上它，否则降级即提权。

    top_k / rerank_top_k 为 None 时回落到 Settings.RAG_TOP_K / RAG_RERANK_TOP_K，
    让 .env 的调参对在线链路与离线消融同时生效（此前只有离线脚本读这两个配置，
    改 .env 对线上无影响——是个会误导运维的陷阱）。
    """
    settings = get_settings()
    if top_k is None:
        top_k = settings.RAG_TOP_K
    if rerank_top_k is None:
        rerank_top_k = settings.RAG_RERANK_TOP_K

    # 权限谓词：在检索请求里生效（admin/关闭开关时返回 None）
    acl_expr = doc_acl_expr(role)

    # 始终计算原始查询向量；use_hyde 时额外算 HyDE 向量作为第二路 dense
    query_vec = await embedding_model.aembed_query(question)
    extra_dense_queries = None
    if use_hyde and llm is not None:
        from src.knowledge.hyde import generate_hyde_embedding
        hyde_vec = await generate_hyde_embedding(question, llm, embedding_model)
        extra_dense_queries = [hyde_vec]

    filters = {}
    if doc_type:
        filters["doc_type"] = doc_type
    if model_code:
        filters["model_code"] = model_code  # ★

    logger.info(
        f"DocRAG 多路召回: top_k={top_k} hyde={use_hyde} role={role} "
        f"filter={filters or '无'} acl={acl_expr or '无（admin/已关闭）'}"
    )

    hits = []
    try:
        from src.rag.retrieval.hybrid_search import hybrid_search
        hits = await hybrid_search(
            milvus_client,
            COLLECTION_NAME,
            dense_embedding=query_vec,
            query_text=question,
            acl_expr=acl_expr,
            top_k=top_k,
            filters=filters or None,
            extra_dense_queries=extra_dense_queries,
        )
    except Exception as e:
        # 降级：RRF 混合检索失败 → 回退旧 dense-only 检索，再失败返回 []
        # ★ 降级路径同样必须带 ACL：漏带等于"hybrid 一挂就提权"
        logger.warning(f"混合检索失败，降级为 dense-only: {e}")
        try:
            from src.infra.milvus_client import escape_milvus_string
            from src.rag.retrieval.hybrid_search import _and_expr
            filter_parts = []
            if doc_type:
                filter_parts.append(f'doc_type == "{escape_milvus_string(doc_type)}"')
            if model_code:
                filter_parts.append(f'model_code == "{escape_milvus_string(model_code)}"')
            filter_expr = _and_expr(" and ".join(filter_parts), acl_expr) or None
            results = await asyncio.to_thread(
                milvus_client.search,
                collection_name=COLLECTION_NAME,
                data=[query_vec],
                limit=top_k,
                output_fields=["doc_name", "doc_type", "page_number", "chunk_index", "text", "parent_text", "image_urls"],
                anns_field="embedding",
                search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
                filter=filter_expr,
            )
            hits = [
                {**hit["entity"], "score": hit.get("distance", 0.0)}
                for hit in (results[0] if results and results[0] else [])
            ]
        except Exception as e2:
            logger.warning(f"dense-only 检索也失败: {e2}")
            return [], []

    if not hits:
        logger.info("DocRAG 召回: 0 条")
        return [], []

    logger.info(
        f"DocRAG 召回: {len(hits)} 条, top_score={hits[0]['score']:.4f}, "
        f"来源: {list(set(h['doc_name'] for h in hits[:5]))}"
    )

    from src.knowledge.reranker import rerank_docs
    reranked = await rerank_docs(
        question, hits, top_k=rerank_top_k, use_dynamic_topk=use_dynamic_topk
    )
    logger.info(
        f"DocRAG 精排后: {len(reranked)} 条 (rerank_top_k={rerank_top_k}), "
        f"top_score={reranked[0].get('rerank_score', reranked[0]['score']):.4f}"
    )
    return hits, reranked


async def search_docs_raw(
    question: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    role: str,
    top_k: int | None = None,
    rerank_top_k: int | None = None,
    doc_type: str | None = None,
    model_code: str | None = None,  # ★ 新增
    llm: BaseChatModel | None = None,
    use_hyde: bool = False,
    use_dynamic_topk: bool | None = None,
) -> list[dict]:
    """文档多路召回 → 精排 → 断崖截断，返回最终证据（生产路径）。

    role 无默认值：权限谓词依赖它，默认值会让调用方"忘了传"变成
    "按 engineer 口径静默检索"——安全相关的参数不该有隐含取值。
    """
    _, final_hits = await search_docs_with_stages(
        question, embedding_model, milvus_client, role,
        top_k=top_k, rerank_top_k=rerank_top_k,
        doc_type=doc_type, model_code=model_code,
        llm=llm, use_hyde=use_hyde, use_dynamic_topk=use_dynamic_topk,
    )
    return final_hits


def format_doc_context(hits: list[dict]) -> str:
    """格式化检索结果为 LLM 可读上下文。
    parent_child 策略时优先使用 parent_text（完整父块），
    否则使用 text（子块）。

    ★ 页码为 0 表示「未知」，此时不显示页码而不是显示"第0页"：
    页码来自解析期锚点定位，兜底解析或锚点缺失时取不到；
    显示一个看起来正常的错误页码，比不显示更糟。
    """
    if not hits:
        return ""
    parts = []
    for i, hit in enumerate(hits, 1):
        page = hit.get("page_number")
        if page in (0, "", None):
            source = f"[{hit['doc_name']}]"
        else:
            source = f"[{hit['doc_name']}, 第{page}页]"
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
    role: str,
    top_k: int = 20,
    rerank_top_k: int = 5,
    doc_type: str | None = None,
    model_code: str | None = None,
    use_hyde: bool = True,
    use_dynamic_topk: bool | None = None,
) -> str:
    """DocRAG 完整流程：多路检索 + 精排 + HyDE + 生成回答"""
    hits = await search_docs_raw(
        question, embedding_model, milvus_client, role,
        top_k=top_k, rerank_top_k=rerank_top_k,
        doc_type=doc_type, model_code=model_code,
        llm=llm, use_hyde=use_hyde, use_dynamic_topk=use_dynamic_topk,
    )
    if not hits:
        return "当前知识库中未找到与您问题相关的文档内容。"

    context = format_doc_context(hits)
    prompt = DOC_QA_PROMPT.format(question=question, context=context, role=role)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return response.content
