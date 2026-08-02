# ============================================================
# Node ⑨ — 执行 SQL + 返回结果
# ============================================================

from sqlalchemy import text

from src.nl2sql.engine import generate_summary, setup_readonly_session
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

    # 角色行级过滤 —— 使用认证用户角色/域/业务线（由 bi.py 注入 ctx）
    from src.nl2sql.security import apply_role_filter
    role = ctx.get("role", "engineer")
    allowed, filtered_sql = apply_role_filter(
        sql, role,
        business_line=ctx.get("business_line"),
        owner_domain_id=ctx.get("owner_domain_id"),
    )

    columns, rows, summary, error = [], [], "", ""
    if not allowed:
        error = filtered_sql  # 如 customer 角色直接拒绝
    else:
        try:
            await setup_readonly_session(db)
            result = await db.execute(text(filtered_sql))
            columns = list(result.keys())
            rows = [dict(row) for row in result.mappings().all()]
            summary = await generate_summary(state["query"], rows, llm)
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
