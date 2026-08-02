# ============================================================
# NL2SQL 核心引擎
# ★ 修复医疗版 validated NameError (search_sql_raw 返回二元组)
# ★ 新增 last_sql 支持下钻追问
# ============================================================

from __future__ import annotations

import json
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from src.nl2sql.prompts import (
    FOLLOWUP_PROMPT,
    NL2SQL_SYSTEM_PROMPT,
    SCHEMA_PROMPT,
    SUMMARY_PROMPT,
)
from src.nl2sql.security import apply_role_filter, validate_sql

MAX_RETRIES = 2
SQL_TIMEOUT = 10


@dataclass
class QueryResult:
    question: str
    sql: str = ""
    data: list[dict] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    summary: str = ""
    error: str = ""
    success: bool = True


@dataclass
class ConversationContext:
    history: list[QueryResult] = field(default_factory=list)

    @property
    def last_result(self) -> QueryResult | None:
        return self.history[-1] if self.history else None

    def add(self, result: QueryResult):
        self.history.append(result)
        if len(self.history) > 10:
            self.history = self.history[-10:]


async def generate_sql(
    question: str,
    llm: BaseChatModel,
    role: str = "engineer",
    owner_domain_id: int | None = None,
    business_line: str | None = None,
    context: ConversationContext | None = None,
    error_hint: str = "",
) -> str:
    """LLM 生成 SQL"""
    if context and context.last_result and context.last_result.success:
        prompt = FOLLOWUP_PROMPT.format(
            previous_sql=context.last_result.sql,
            previous_summary=context.last_result.summary[:500],
            question=question,
            schema=SCHEMA_PROMPT,
        )
    else:
        prompt = NL2SQL_SYSTEM_PROMPT.format(schema=SCHEMA_PROMPT)

    messages = [SystemMessage(content=prompt)]
    if error_hint:
        messages.append(HumanMessage(
            content=f"上一轮 SQL 报错：{error_hint}\n请修正。\n\n{question}"
        ))
    else:
        messages.append(HumanMessage(content=question))

    response = await llm.ainvoke(messages)
    sql = response.content.strip()
    if "```" in sql:
        sql = sql.split("```")[1].lstrip("sql").strip()
    return sql


async def setup_readonly_session(db: AsyncSession) -> None:
    """执行前设置会话安全属性：超时 + 只读（第四层防线）。

    供旧引擎与流水线 execute_sql 节点共用，保证两条路径行为一致。"""
    await db.execute(text(f"SET LOCAL statement_timeout = '{SQL_TIMEOUT * 1000}'"))
    await db.execute(text("SET LOCAL default_transaction_read_only = on"))


async def execute_sql(sql: str, db: AsyncSession) -> tuple[list[dict], list[str]]:
    """执行 SQL，返回 (rows, columns)"""
    await setup_readonly_session(db)
    result = await db.execute(text(sql))
    columns = list(result.keys())
    rows = [dict(row) for row in result.mappings().all()]
    return rows, columns


async def generate_summary(question: str, data: list[dict], llm: BaseChatModel) -> str:
    """LLM 生成数据摘要"""
    result_str = json.dumps(data[:20], ensure_ascii=False, default=str)
    prompt = SUMMARY_PROMPT.format(question=question, result=result_str)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return response.content


async def run_query(
    question: str,
    llm: BaseChatModel,
    db: AsyncSession,
    role: str = "engineer",
    owner_domain_id: int | None = None,
    business_line: str | None = None,
    context: ConversationContext | None = None,
) -> QueryResult:
    """完整 NL2SQL 流程：生成 SQL → 安全校验 → 行过滤 → 执行 → 摘要"""
    error_hint = ""

    for attempt in range(MAX_RETRIES + 1):
        raw_sql = await generate_sql(
            question, llm, role, owner_domain_id, business_line, context, error_hint,
        )
        logger.info(f"NL2SQL (attempt {attempt + 1}): {raw_sql}")

        valid, validated = validate_sql(raw_sql)
        if not valid:
            result = QueryResult(question=question, sql=raw_sql,
                                 error=f"安全校验失败: {validated}", success=False)
            if context:
                context.add(result)
            return result

        allowed, filtered_sql = apply_role_filter(
            validated, role, business_line, owner_domain_id,
        )
        if not allowed:
            result = QueryResult(question=question, sql=raw_sql,
                                 error=filtered_sql, success=False)
            if context:
                context.add(result)
            return result

        try:
            data, columns = await execute_sql(filtered_sql, db)
            summary = await generate_summary(question, data, llm)

            result = QueryResult(
                question=question, sql=filtered_sql,
                data=data, columns=columns,
                row_count=len(data), summary=summary,
            )
            if context:
                context.add(result)
            return result

        except DBAPIError as e:
            await db.rollback()  # ★ 重置 abort 状态，否则后续重试全在坏事务里
            if "canceling statement" in str(e) or "timeout" in str(e).lower():
                result = QueryResult(question=question, sql=filtered_sql,
                                     error=f"查询超时（{SQL_TIMEOUT}秒）", success=False)
                if context:
                    context.add(result)
                return result
            error_hint = str(e)
            logger.warning(f"SQL 执行失败 (attempt {attempt + 1}): {e}")
            if attempt == MAX_RETRIES:
                result = QueryResult(question=question, sql=filtered_sql,
                                     error=f"执行失败: {error_hint}", success=False)
                if context:
                    context.add(result)
                return result

        except Exception as e:
            await db.rollback()
            error_hint = str(e)
            logger.warning(f"异常 (attempt {attempt + 1}): {e}")
            if attempt == MAX_RETRIES:
                result = QueryResult(question=question, sql=raw_sql,
                                     error=f"执行失败: {error_hint}", success=False)
                if context:
                    context.add(result)
                return result

    result = QueryResult(question=question, sql="", error="未知错误", success=False)
    if context:
        context.add(result)
    return result


# ★ 修复 NameError：search_sql_raw 返回 (data, executed_sql) 二元组
async def search_sql_raw(
    question: str,
    llm: BaseChatModel,
    db: AsyncSession,
    role: str = "engineer",
    owner_domain_id: int | None = None,
    business_line: str | None = None,
) -> tuple[list[dict], str]:
    """检索 NL2SQL 原始结果，返回 (rows, executed_sql)"""
    result = await run_query(question, llm, db, role, owner_domain_id, business_line)
    if result.success:
        return result.data, result.sql
    return [], result.sql


async def search_sql(
    question: str,
    llm: BaseChatModel,
    db: AsyncSession,
    role: str = "engineer",
    owner_domain_id: int | None = None,
    business_line: str | None = None,
) -> str:
    """检索 NL2SQL 结果，返回 LLM 摘要"""
    result = await run_query(question, llm, db, role, owner_domain_id, business_line)
    if result.success:
        return result.summary
    return result.error
