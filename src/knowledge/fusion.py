# ============================================================
# 多通道融合检索：doc + graph [+ nl2sql] 并行 → 融合 → 幻觉检测
# ★ return_exceptions=True：单通道失败不拖垮全局
# ============================================================

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from loguru import logger
from neo4j import AsyncDriver
from pymilvus import MilvusClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.knowledge.doc_rag import format_doc_context, search_docs_raw
from src.knowledge.graph_rag import search_graph_raw
from src.knowledge.hallucination_check import check_hallucination
from src.knowledge.prompts import FUSION_PROMPT


def _emit(event_sink: Callable[[dict], None] | None, msg: dict) -> None:
    if event_sink is not None:
        event_sink(msg)


async def _generate_answer(
    llm: BaseChatModel,
    prompt: str,
    event_sink: Callable[[dict], None] | None = None,
) -> str:
    """生成回答。

    event_sink 存在时用 astream 逐 token 推送 {"type":"delta","content":...}，
    供 SSE 流式消费；否则 ainvoke 一次性返回（保持原有行为）。
    """
    messages = [SystemMessage(content=prompt)]
    if event_sink is None:
        response = await llm.ainvoke(messages)
        return response.content

    answer_parts = []
    async for chunk in llm.astream(messages):
        content = getattr(chunk, "content", None)
        if content:
            answer_parts.append(content)
            event_sink({"type": "delta", "content": content})
    return "".join(answer_parts)


async def multi_channel_search(
    question: str,
    llm: BaseChatModel,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    neo4j_driver: AsyncDriver,
    db_session: AsyncSession | None = None,
    channels: list[str] | None = None,
    role: str = "engineer",
    use_hyde: bool = False,
    event_sink: Callable[[dict], None] | None = None,
) -> dict:
    """
    多通道并行检索 → 结果融合 → 幻觉检测 → 返回 {"answer": str, "contexts": list[str]}。

    event_sink: 可选回调，推流式事件：检索进度 {"type":"progress",...}、
    答案 token {"type":"delta",...}、结束 {"type":"done",...}。
    """
    if channels is None:
        channels = ["doc_rag", "graph_rag"]

    tasks = {}
    if "doc_rag" in channels:
        tasks["doc_rag"] = search_docs_raw(
            question, embedding_model, milvus_client,
            llm=llm, use_hyde=use_hyde,
        )
    if "graph_rag" in channels:
        tasks["graph_rag"] = search_graph_raw(question, neo4j_driver, llm)
    if "nl2sql" in channels and db_session:
        from src.nl2sql.engine import search_sql
        tasks["nl2sql"] = search_sql(question, llm, db_session)

    logger.info(f"多通道检索开始: channels={list(tasks.keys())}")

    # asyncio.wait(FIRST_COMPLETED)：每完成一个通道立即推进度事件，
    # 比 gather 一次性返回更贴近流式体验
    results = {}
    pending = {asyncio.create_task(coro, name=key) for key, coro in tasks.items()}
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            key = task.get_name()
            try:
                result = task.result()
            except Exception as e:
                logger.warning(f"通道 {key} 检索失败: {e}")
                results[key] = None
                _emit(event_sink, {"type": "progress", "channel": key, "status": "failed", "count": 0})
                continue
            results[key] = result
            if isinstance(result, list):
                count, status = len(result), ("ok" if result else "empty")
            elif isinstance(result, str) and result:
                count, status = 1, "ok"
            else:
                count, status = 0, "skipped"
            _emit(event_sink, {"type": "progress", "channel": key, "status": status, "count": count})

    # ── 汇总命中情况 ──
    summary_parts = []
    doc_hits = results.get("doc_rag")
    graph_records = results.get("graph_rag")
    sql_answer = results.get("nl2sql")

    doc_count = len(doc_hits) if isinstance(doc_hits, list) else 0
    graph_count = len(graph_records) if isinstance(graph_records, list) else 0
    sql_ok = bool(sql_answer) if sql_answer is not None else None

    summary_parts.append(f"doc_rag={doc_count}条")
    if "graph_rag" in channels:
        summary_parts.append(f"graph_rag={graph_count}条")
    if "nl2sql" in channels:
        summary_parts.append(f"nl2sql={'✓' if sql_ok else '✗' if sql_ok is not None else '跳过'}")
    logger.info(f"多通道检索完成: {', '.join(summary_parts)}")

    source_parts = []
    evidence_parts = []
    retrieved_chunks = []  # 收集所有检索 chunks，供评测使用

    if doc_hits:
        ctx = format_doc_context(doc_hits)
        source_parts.append(f"### 文档检索结果\n{ctx}")
        evidence_parts.append(ctx[:1000])
        for hit in doc_hits:
            # hit 是 dict，text 字段存储子块内容，parent_text 是完整父块
            text = hit.get("parent_text") or hit.get("text", "")
            if text:
                retrieved_chunks.append(text[:500])

    graph_records = results.get("graph_rag")
    if graph_records:
        graph_str = json.dumps(graph_records, ensure_ascii=False, indent=2)
        source_parts.append(f"### 知识图谱检索结果\n{graph_str}")
        evidence_parts.append(graph_str[:1000])
        retrieved_chunks.append(graph_str[:2000])

    sql_answer = results.get("nl2sql")
    if sql_answer and isinstance(sql_answer, str):
        source_parts.append(f"### 运营数据查询结果\n{sql_answer}")
        evidence_parts.append(sql_answer[:1000])

    if not source_parts:
        answer = "所有检索通道均未找到与您问题相关的信息。"
        _emit(event_sink, {"type": "done", "answer": answer, "contexts": []})
        return {"answer": answer, "contexts": []}

    sources = "\n\n".join(source_parts)
    prompt = FUSION_PROMPT.format(question=question, sources=sources, role=role)
    answer = await _generate_answer(llm, prompt, event_sink)
    _emit(event_sink, {"type": "done", "answer": answer, "contexts": retrieved_chunks})

    evidence = "\n".join(evidence_parts)
    hal_result = await check_hallucination(question, evidence, answer, llm)
    if not hal_result["is_grounded"]:
        claims = "、".join(hal_result.get("unsupported_claims", []))
        answer += f"\n\n⚠️ 提示：以下内容未在手册中完全印证：{claims}"

    return {
        "answer": answer,
        "contexts": retrieved_chunks,
    }
