# ============================================================
# 多通道融合检索：doc + graph [+ nl2sql] 并行 → 融合 → 幻觉检测
# ★ return_exceptions=True：单通道失败不拖垮全局
# ============================================================

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from loguru import logger
from neo4j import AsyncDriver
from pymilvus import MilvusClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.core.metrics import RETRIEVAL_REQUESTS
from src.core.resilience import with_retry, get_channel_breaker
from src.knowledge.doc_rag import extract_image_urls, format_doc_context, search_docs_raw
from src.knowledge.graph_rag import search_graph_raw
from src.knowledge.hallucination_check import check_hallucination
from src.knowledge.prompts import FUSION_PROMPT

_settings = get_settings()


def _emit(event_sink: Callable[[dict], None] | None, msg: dict) -> None:
    if event_sink is not None:
        event_sink(msg)


async def _run_channel(key: str, factory: Callable[[], object]) -> object:
    """单个检索通道：熔断 → 重试（退避）→ 超时，三层保护。

    - 熔断器 open 时直接抛错（快速失败），不再发起调用；
    - 临时失败按 RETRIEVAL_CHANNEL_RETRIES 次指数退避重试；
    - 单次执行受 RETRIEVAL_CHANNEL_TIMEOUT 限制，超时按失败降级。
    """
    breaker = get_channel_breaker(key)

    # 顺序：熔断(外) → 退避重试(中) → 单次超时(内)
    async def attempt() -> object:
        return await asyncio.wait_for(factory(), timeout=_settings.RETRIEVAL_CHANNEL_TIMEOUT)

    return await breaker.call(
        lambda: with_retry(
            attempt,
            attempts=_settings.RETRIEVAL_CHANNEL_RETRIES,
            base_delay=0.3,
            retry_on=(Exception, asyncio.TimeoutError),
            task=f"channel:{key}",
        )
    )


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


def _sanitize_answer_images(answer: str, valid_urls: set[str]) -> str:
    """只保留答案中真实存在于检索源里的图片引用，防 LLM 编造/抄占位符。

    1) 地址含省略号（...）的引用是占位符编造，直接剥成描述文字；
    2) 逐条校验剩余 ![描述](url)：url 在检索命中的图片集合内才保留完整引用，
       否则去掉引用只剩描述文字（valid_urls 为空时全部不可信，一并剥掉）；
    3) 裸占位 URL（非 markdown 形式）直接删除，避免展示 "http://...url.../"。"""
    # ① 占位符引用（URL 含省略号）：剥成描述文字
    answer = re.sub(
        r"!\[([^\]]*)\]\((https?://[^\s)]*\.\.[^\s)]*)\)",
        lambda m: m.group(1).strip(), answer,
    )
    # ② 校验剩余引用：URL 必须真实存在于检索源
    def _repl(m: re.Match) -> str:
        alt, url = m.group(1), m.group(2)
        return m.group(0) if url in valid_urls else alt.strip()

    answer = re.sub(r"!\[([^\]]*)\]\((https?://[^\s)]+)\)", _repl, answer)
    # ③ 裸占位 URL（http://... 开头，真实地址不会以省略号开头）连前一空格一并删除
    answer = re.sub(r" ?https?://\.\.[^\s)\]\"']*", "", answer)
    return answer


def _extract_md_image_urls(text: str) -> list[str]:
    """从 markdown 文本中提取图片 URL（按出现顺序，去重）。

    用于把"答案引用的图片"作为 image_urls 返回，保证图库与正文一一对应。"""
    urls, seen = [], set()
    for url in re.findall(r"!\[[^\]]*\]\((https?://[^\s)]+)\)", text):
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


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

    # 通道用"工厂函数"而非协程：失败重试时可以重新执行
    tasks: dict[str, Callable[[], object]] = {}
    if "doc_rag" in channels:
        tasks["doc_rag"] = lambda: search_docs_raw(
            question, embedding_model, milvus_client,
            llm=llm, use_hyde=use_hyde,
        )
    if "graph_rag" in channels:
        tasks["graph_rag"] = lambda: search_graph_raw(question, neo4j_driver, llm)
    if "nl2sql" in channels and db_session:
        from src.nl2sql.engine import search_sql
        tasks["nl2sql"] = lambda: search_sql(question, llm, db_session)

    logger.info(f"多通道检索开始: channels={list(tasks.keys())}")

    # asyncio.wait(FIRST_COMPLETED)：每完成一个通道立即推进度事件，
    # 比 gather 一次性返回更贴近流式体验
    results = {}
    pending = {
        asyncio.create_task(_run_channel(key, factory), name=key)
        for key, factory in tasks.items()
    }
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            key = task.get_name()
            try:
                result = task.result()
            except Exception as e:
                logger.warning(f"通道 {key} 检索失败: {e}")
                results[key] = None
                RETRIEVAL_REQUESTS.labels(channel=key, status="failed").inc()
                _emit(event_sink, {"type": "progress", "channel": key, "status": "failed", "count": 0})
                continue
            results[key] = result
            if isinstance(result, list):
                count, status = len(result), ("ok" if result else "empty")
            elif isinstance(result, str) and result:
                count, status = 1, "ok"
            else:
                count, status = 0, "skipped"
            RETRIEVAL_REQUESTS.labels(channel=key, status=status).inc()
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
    image_urls = []        # 文档检索命中的图片 URL，供响应层展示

    if doc_hits:
        ctx = format_doc_context(doc_hits)
        source_parts.append(f"### 文档检索结果\n{ctx}")
        evidence_parts.append(ctx[:1000])
        for hit in doc_hits:
            # hit 是 dict，text 字段存储子块内容，parent_text 是完整父块
            text = hit.get("parent_text") or hit.get("text", "")
            if text:
                retrieved_chunks.append(text[:500])
        image_urls = extract_image_urls(doc_hits)

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
        _emit(event_sink, {"type": "done", "answer": answer, "contexts": [], "image_urls": []})
        return {"answer": answer, "contexts": [], "image_urls": []}

    sources = "\n\n".join(source_parts)
    prompt = FUSION_PROMPT.format(question=question, sources=sources, role=role)
    answer = await _generate_answer(llm, prompt, event_sink)
    # ★ 图片防伪：答案里的图片 URL 必须真实存在于检索源，防 LLM 编造
    answer = _sanitize_answer_images(answer, set(image_urls))
    # ★ 图库只返回答案真正引用的图片，与正文一一对应（不再返回全部命中图）
    image_urls = _extract_md_image_urls(answer)
    _emit(event_sink, {"type": "done", "answer": answer, "contexts": retrieved_chunks, "image_urls": image_urls})

    evidence = "\n".join(evidence_parts)
    hal_result = await check_hallucination(question, evidence, answer, llm)
    if not hal_result["is_grounded"]:
        claims = "、".join(hal_result.get("unsupported_claims", []))
        answer += f"\n\n⚠️ 提示：以下内容未在手册中完全印证：{claims}"

    return {
        "answer": answer,
        "contexts": retrieved_chunks,
        "image_urls": image_urls,
    }
