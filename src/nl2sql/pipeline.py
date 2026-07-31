# ============================================================
# NL2SQL 流水线入口 — 手动编排节点，通过 async generator 流式返回
#
# 不使用 LangGraph astream（在 FastAPI StreamingResponse 中会 hang），
# 改用 asyncio.gather 实现并行，直接调用节点函数。
# ============================================================

import asyncio

from src.nl2sql.context import DataAgentContext

# ── 原始节点函数 ────────────────────────────────────────────────
from src.nl2sql.nodes.add_context import add_context as _add_context
from src.nl2sql.nodes.correct_sql import correct_sql as _correct_sql
from src.nl2sql.nodes.execute_sql import execute_sql as _execute_sql
from src.nl2sql.nodes.extract_keywords import extract_keywords as _extract_keywords
from src.nl2sql.nodes.filter_metrics import filter_metrics as _filter_metrics
from src.nl2sql.nodes.filter_tables import filter_tables as _filter_tables
from src.nl2sql.nodes.generate_sql import generate_sql as _generate_sql
from src.nl2sql.nodes.merge_info import merge_info as _merge_info
from src.nl2sql.nodes.recall_columns import recall_columns as _recall_columns
from src.nl2sql.nodes.recall_metrics import recall_metrics as _recall_metrics
from src.nl2sql.nodes.recall_values import recall_values as _recall_values
from src.nl2sql.nodes.validate_sql import validate_sql as _validate_sql


async def run_pipeline(query: str, ctx: DataAgentContext):
    """手动编排 NL2SQL 流水线，通过 async generator 流式返回状态事件。

    图结构（与 graph.py 定义一致）：
      ① extract_keywords
         → ② 三路并行召回 (recall_columns | recall_values | recall_metrics)
         → ③ merge_info
         → ④ 两路并行过滤 (filter_tables | filter_metrics)
         → ⑤ add_context → ⑥ generate_sql → ⑦ validate_sql
         → ⑧ error ? correct_sql : skip
         → ⑨ execute_sql → END
    """
    from src.core.logger import logger

    logger.info(f"[pipeline] starting for query: {query[:80]}")
    state: dict = {"query": query}
    event_count = 0

    def emit(node_name: str, result: dict) -> dict:
        nonlocal event_count
        event_count += 1
        logger.info(f"[pipeline] event #{event_count}: {node_name}")
        return {node_name: result}

    # ① 关键词提取
    result = await _extract_keywords(state, ctx)
    state.update(result)
    yield emit("extract_keywords", result)

    # ② 三路并行召回
    recall_results = await asyncio.gather(
        _recall_columns(state, ctx),
        _recall_values(state, ctx),
        _recall_metrics(state, ctx),
    )
    for name, result in [
        ("recall_columns", recall_results[0]),
        ("recall_values", recall_results[1]),
        ("recall_metrics", recall_results[2]),
    ]:
        state.update(result)
        yield emit(name, result)

    # ③ 合并检索结果
    result = await _merge_info(state, ctx)
    state.update(result)
    yield emit("merge_info", result)

    # ④ 两路并行过滤
    filter_results = await asyncio.gather(
        _filter_tables(state, ctx),
        _filter_metrics(state, ctx),
    )
    for name, result in [
        ("filter_tables", filter_results[0]),
        ("filter_metrics", filter_results[1]),
    ]:
        state.update(result)
        yield emit(name, result)

    # ⑤ 注入日期/DB 上下文
    result = await _add_context(state, ctx)
    state.update(result)
    yield emit("add_context", result)

    # ⑥ LLM 生成 SQL
    result = await _generate_sql(state, ctx)
    state.update(result)
    yield emit("generate_sql", result)

    # ⑦ EXPLAIN 校验
    result = await _validate_sql(state, ctx)
    state.update(result)
    yield emit("validate_sql", result)

    # ⑧ 条件纠错
    if state.get("error"):
        result = await _correct_sql(state, ctx)
        state.update(result)
        yield emit("correct_sql", result)

    # ⑨ 执行 SQL
    result = await _execute_sql(state, ctx)
    state.update(result)
    yield emit("execute_sql", result)

    logger.info(f"[pipeline] complete, {event_count} events")
