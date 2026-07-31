# ============================================================
# Node ⑨ — 执行 SQL + 返回结果
# ============================================================

from sqlalchemy import text

from src.nl2sql.engine import generate_summary
from src.nl2sql.state import DataAgentState
from src.nl2sql.context import DataAgentContext


async def execute_sql(state: DataAgentState, ctx: DataAgentContext) -> dict:
    """执行 SQL 并返回结果"""
    writer = ctx.get("writer")
    if writer:
        writer({"type": "progress", "step": "执行SQL", "status": "running"})

    sql = state["sql"]
    db = ctx["dw_db_session"]
    llm = ctx["llm"]

    # 角色行级过滤
    from src.nl2sql.security import apply_role_filter
    _, filtered_sql = apply_role_filter(sql, "engineer", None, None)

    try:
        await db.execute(text("SET LOCAL statement_timeout = '10000'"))
        result = await db.execute(text(filtered_sql))
        columns = list(result.keys())
        rows = [dict(row) for row in result.mappings().all()]
        summary = await generate_summary(state["query"], rows, llm)
        error = None
    except Exception as e:
        columns, rows, summary, error = [], [], "", str(e)

    from src.core.logger import logger

    if error:
        logger.error(f"[execute_sql] 执行失败: {error}")
        if writer:
            writer({"type": "progress", "step": "执行SQL", "status": "error"})
    else:
        logger.info(f"[execute_sql] 返回 {len(rows)} 行, {len(columns)} 列")
        if writer:
            writer({"type": "progress", "step": "执行SQL", "status": "success"})
            writer({"type": "result", "data": {
                "sql": filtered_sql,
                "columns": columns,
                "data": rows,
                "row_count": len(rows),
                "summary": summary,
            }})

    return {
        "error": error,
        "result_sql": filtered_sql,
        "result_columns": columns,
        "result_data": rows,
        "result_row_count": len(rows),
        "result_summary": summary,
        "result_error": error or "",
    }
