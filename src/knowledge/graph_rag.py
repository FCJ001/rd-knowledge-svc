# ============================================================
# GraphRAG：实体提取 → NL2Cypher（最多 3 次重试）→ LLM 整合
# 读项目一的 Neo4j。
# ★ 只读强制（双保险）：
#   1. 关键词拦截 —— LLM 生成的 Cypher 含任何写子句直接拒绝；
#   2. READ_ACCESS 会话 —— 即使拦截被绕过（如 CALL 存储过程），数据库侧也拒绝写。
# ============================================================

from __future__ import annotations

import asyncio
import json
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from loguru import logger
from neo4j import READ_ACCESS, AsyncDriver

from src.core.config import get_settings
from src.knowledge.prompts import ENTITY_EXTRACT_PROMPT, GRAPH_QA_PROMPT, NL2CYPHER_PROMPT

MAX_CYPHER_RETRIES = 2

# 写子句黑名单：LLM 输出出现任意一个即拒绝执行（问题文本可注入 prompt，
# 所以不能信任 LLM "只会生成读查询"这一假设）
_CYPHER_WRITE_RE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV)\b"
    r"|\bCALL\s+(dbms|db)\.",
    re.IGNORECASE,
)


async def _extract_entities(question: str, llm: BaseChatModel) -> dict:
    prompt = ENTITY_EXTRACT_PROMPT.format(question=question)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        return json.loads(content)
    except Exception as e:
        logger.warning(f"实体提取失败: {e}")
        return {"phenomena": [], "root_causes": [], "config_items": [],
                "baselines": [], "requirements": []}


async def _generate_cypher(
    question: str, entities: dict, llm: BaseChatModel, error_hint: str = "",
) -> str:
    extra = ""
    if error_hint:
        extra = f"\n\n上一次生成的 Cypher 执行报错：{error_hint}\n请修正后重新生成。"
    prompt = NL2CYPHER_PROMPT.format(
        question=question,
        entities=json.dumps(entities, ensure_ascii=False),
    ) + extra
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    cypher = response.content.strip()
    if "```" in cypher:
        cypher = cypher.split("```")[1].lstrip("cypher").strip()
    return cypher


async def _execute_cypher(cypher: str, neo4j_driver: AsyncDriver) -> list[dict]:
    if not cypher:
        return []
    if _CYPHER_WRITE_RE.search(cypher):
        raise ValueError("拒绝执行包含写操作的 Cypher（只读通道）")
    settings = get_settings()

    async def _run() -> list[dict]:
        # READ_ACCESS：数据库侧强制只读，写语句直接被 Neo4j 拒绝
        async with neo4j_driver.session(default_access_mode=READ_ACCESS) as session:
            result = await session.run(cypher)
            return await result.data()

    return await asyncio.wait_for(_run(), timeout=settings.NEO4J_QUERY_TIMEOUT)


async def search_graph_raw(
    question: str,
    neo4j_driver: AsyncDriver,
    llm: BaseChatModel,
) -> list[dict]:
    """GraphRAG 检索，返回原始图谱结果"""
    entities = await _extract_entities(question, llm)
    entity_counts = {k: len(v) for k, v in entities.items()}
    logger.info(f"GraphRAG 实体提取: {entity_counts}")

    error_hint = ""
    for attempt in range(MAX_CYPHER_RETRIES + 1):
        cypher = await _generate_cypher(question, entities, llm, error_hint)
        logger.info(f"GraphRAG Cypher (attempt {attempt + 1}): {cypher[:150]}")
        try:
            records = await _execute_cypher(cypher, neo4j_driver)
            logger.info(f"GraphRAG 命中: {len(records)} 条记录 (attempt {attempt + 1})")
            return records[:20]
        except Exception as e:
            error_hint = str(e)
            logger.warning(f"Cypher 执行失败 (attempt {attempt + 1}): {e}")
            if attempt == MAX_CYPHER_RETRIES:
                return []
    return []


async def search_graph(
    question: str,
    neo4j_driver: AsyncDriver,
    llm: BaseChatModel,
    role: str = "engineer",
) -> str:
    """GraphRAG 完整流程"""
    records = await search_graph_raw(question, neo4j_driver, llm)

    if not records:
        return "知识图谱中未找到与您问题相关的信息。"

    graph_result = json.dumps(records, ensure_ascii=False, indent=2)
    prompt = GRAPH_QA_PROMPT.format(question=question, graph_result=graph_result, role=role)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return response.content
