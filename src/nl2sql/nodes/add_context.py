# ============================================================
# Node ⑤ — 注入日期/DB 上下文
# ============================================================

from datetime import datetime

from src.nl2sql.state import DataAgentState
from src.nl2sql.context import DataAgentContext


async def add_context(state: DataAgentState, ctx: DataAgentContext) -> dict:
    """注入当前日期信息和数据库方言/版本"""
    writer = ctx.get("writer")
    if writer:
        writer({"type": "progress", "step": "添加额外上下文信息", "status": "running"})

    try:
        now = datetime.now()
        quarter = f"Q{(now.month - 1) // 3 + 1}"
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        date_info = {
            "date": now.strftime("%Y-%m-%d"),
            "weekday": weekdays[now.weekday()],
            "quarter": quarter,
        }

        db_info = {
            "dialect": "postgresql",
            "version": "16",
        }

        from src.core.logger import logger
        logger.info(f"[add_context] 日期={date_info['date']} {date_info['weekday']} {date_info['quarter']}, 数据库={db_info['dialect']} {db_info['version']}")

        if writer:
            writer({"type": "progress", "step": "添加额外上下文信息", "status": "success"})
        return {"date_info": date_info, "db_info": db_info}
    except Exception as e:
        if writer:
            writer({"type": "progress", "step": "添加额外上下文信息", "status": "error"})
        raise
