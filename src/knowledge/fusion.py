# ============================================================
# 多通道融合检索：doc + graph [+ nl2sql] 并行 → 融合 → 幻觉检测
# ★ return_exceptions=True：单通道失败不拖垮全局
# ★ nl2sql 通道已迁移到 rd-chatBI 服务，本服务通过 HTTP 调用，
#   通道返回数据摘要文本（与旧 engine.search_sql 同构），融合逻辑不变
# ============================================================

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from loguru import logger
from neo4j import AsyncDriver
from pymilvus import MilvusClient

from src.core.config import get_settings
from src.core.metrics import RETRIEVAL_REQUESTS
from src.core.resilience import with_retry, get_channel_breaker
from src.knowledge.doc_rag import extract_image_urls, format_doc_context, search_docs_raw
from src.knowledge.graph_rag import search_graph_raw
from src.knowledge.hallucination_check import check_hallucination
from src.knowledge.prompts import FUSION_PROMPT
from src.knowledge.query_rewriter import rewrite_query

_settings = get_settings()


def _chatbi_headers(
    role: str, owner_domain_id: int | None, business_line: str | None
) -> dict:
    """ChatBI 请求头：身份 + 行级过滤参数 + 数据源路由"""
    headers = {
        "X-User-Id": "knowledge-svc",
        "X-User-Role": role,
        "X-Project-Id": _settings.CHATBI_PROJECT_ID,
    }
    if owner_domain_id is not None:
        headers["X-Owner-Domain-Id"] = str(owner_domain_id)
    if business_line:
        headers["X-Business-Line"] = business_line
    return headers


async def _search_chatbi(
    question: str,
    role: str = "admin",
    session_id: str = "default",
    owner_domain_id: int | None = None,
    business_line: str | None = None,
) -> str:
    """ChatBI 通道：HTTP 调 rd-chatBI，返回数据摘要文本。

    失败抛异常 → 交给 _run_channel 的重试/熔断/超时三层保护处理。
    只取 summary（图表/表格由前端直连 rd-chatBI，不经过 fusion）。"""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{_settings.CHATBI_URL}/api/v1/bi/query",
            json={"question": question, "session_id": session_id, "with_chart": False},
            headers=_chatbi_headers(role, owner_domain_id, business_line),
        )
        resp.raise_for_status()
        data = resp.json().get("data") or {}
        if not data.get("success"):
            raise RuntimeError(data.get("error") or "chatbi 查询失败")
        return data.get("summary") or "（无数据）"


def _emit(event_sink: Callable[[dict], None] | None, msg: dict) -> None:
    if event_sink is not None:
        event_sink(msg)


async def _rewrite_question(
    question: str,
    llm: BaseChatModel,
    role: str,
    event_sink: Callable[[dict], None] | None,
) -> dict:
    """主链路第一步：Query 改写（口语→专业术语 + 子查询拆分）。

    - 受 QUERY_REWRITE_ENABLED 开关与 QUERY_REWRITE_TIMEOUT 超时双重保护，
      失败/超时一律降级为原问题 —— 改写是增强项，不能成为新的故障点；
    - 改写结果通过 event_sink 推给前端（进度事件 channel=query_rewrite），
      并写入日志供离线评测对照改写前后召回差异。"""
    if not _settings.QUERY_REWRITE_ENABLED:
        return {"queries": [question], "intent": "knowledge_qa"}

    try:
        rewritten = await asyncio.wait_for(
            rewrite_query(question, llm, role=role),
            timeout=_settings.QUERY_REWRITE_TIMEOUT,
        )
        queries = [q for q in rewritten.get("queries", []) if q][: _settings.QUERY_REWRITE_MAX_SUB_QUERIES]
        if not queries:
            raise ValueError("改写结果为空")
        rewritten["queries"] = queries
        logger.info(f"Query 改写完成: {question[:40]} → {queries} intent={rewritten.get('intent')}")
        _emit(event_sink, {
            "type": "progress", "channel": "query_rewrite", "status": "ok",
            "queries": queries, "intent": rewritten.get("intent", ""),
        })
        return rewritten
    except Exception as e:
        logger.warning(f"Query 改写失败，使用原始问题: {e}")
        _emit(event_sink, {"type": "progress", "channel": "query_rewrite", "status": "failed"})
        return {"queries": [question], "intent": "knowledge_qa"}


def _merge_doc_hits(hit_lists: list[list[dict]], limit: int) -> list[dict]:
    """多个子查询的文档命中按 id 去重（保留高分那条），按分数降序截断。"""
    best: dict[str, dict] = {}
    for hits in hit_lists:
        for hit in hits or []:
            key = str(hit.get("id") or hit.get("chunk_id") or hit.get("text", ""))[:256]
            if key not in best or (hit.get("score") or 0) > (best[key].get("score") or 0):
                best[key] = hit
    merged = sorted(best.values(), key=lambda h: h.get("score") or 0, reverse=True)
    return merged[:limit]


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
    channels: list[str] | None = None,
    role: str = "admin",
    use_hyde: bool = False,
    session_id: str = "default",
    owner_domain_id: int | None = None,
    business_line: str | None = None,
    event_sink: Callable[[dict], None] | None = None,
) -> dict:
    """
    多通道并行检索 → 结果融合 → 幻觉检测 → 返回 {"answer": str, "contexts": list[str]}。

    主链路第一步做 Query 改写（_rewrite_question）：主查询喂给全部通道，
    拆分出的子查询额外走 doc_rag 并合并命中（子查询只增强文档召回，
    图谱/nl2sql 通道保持单查询 —— 多子查询会让通道成本翻倍）。
    nl2sql 通道通过 HTTP 调 rd-chatBI（CHATBI_URL），失败自动降级。
    event_sink: 可选回调，推流式事件：检索进度 {"type":"progress",...}、
    答案 token {"type":"delta",...}、结束 {"type":"done",...}。
    """
    if channels is None:
        channels = ["doc_rag", "graph_rag"]

    # ── Query 改写（口语→术语 + 子查询拆分，失败降级原问题）──
    rewritten = await _rewrite_question(question, llm, role, event_sink)
    queries = rewritten["queries"]
    query = queries[0]

    # 通道用"工厂函数"而非协程：失败重试时可以重新执行
    tasks: dict[str, Callable[[], object]] = {}
    if "doc_rag" in channels:
        async def _doc_channel():
            # 主查询 + 其余子查询并行召回，合并去重（增强子问题的文档覆盖）
            hit_lists = await asyncio.gather(*[
                search_docs_raw(
                    q, embedding_model, milvus_client,
                    llm=llm, use_hyde=use_hyde,
                )
                for q in queries
            ])
            return _merge_doc_hits(hit_lists, limit=20)

        tasks["doc_rag"] = _doc_channel
    if "graph_rag" in channels:
        tasks["graph_rag"] = lambda: search_graph_raw(query, neo4j_driver, llm)
    if "nl2sql" in channels:
        tasks["nl2sql"] = lambda: _search_chatbi(
            query, role=role, session_id=session_id,
            owner_domain_id=owner_domain_id, business_line=business_line,
        )

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
