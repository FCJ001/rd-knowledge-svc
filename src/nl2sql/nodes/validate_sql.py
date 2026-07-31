# ============================================================
# Node ⑦ — EXPLAIN 校验 SQL
# ============================================================

from sqlalchemy import text

from src.nl2sql.state import DataAgentState
from src.nl2sql.context import DataAgentContext


async def validate_sql(state: DataAgentState, ctx: DataAgentContext) -> dict:
    """在数据库上执行 EXPLAIN 校验 SQL"""
    writer = ctx.get("writer")
    if writer:
        writer({"type": "progress", "step": "验证SQL", "status": "running"})

    sql = state["sql"]
    db = ctx["dw_db_session"]

    # 安全规则校验
    from src.nl2sql.security import validate_sql as security_check
    valid, result = security_check(sql)
    if not valid:
        if writer:
            writer({"type": "progress", "step": "验证SQL", "status": "error"})
        return {"error": result}

    try:
        await db.execute(text(f"EXPLAIN {sql}"))

        from src.core.logger import logger
        logger.info("[validate_sql] EXPLAIN 校验通过")

        if writer:
            writer({"type": "progress", "step": "验证SQL", "status": "success"})
        return {"error": None}
    except Exception as e:
        from src.core.logger import logger
        logger.warning(f"[validate_sql] EXPLAIN 失败: {e}")

        if writer:
            writer({"type": "progress", "step": "验证SQL", "status": "error"})
        return {"error": str(e)}
