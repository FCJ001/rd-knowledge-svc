# ============================================================
# Node ⑥ — LLM 生成 SQL
# ============================================================

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from src.nl2sql.state import DataAgentState
from src.nl2sql.context import DataAgentContext
from src.nl2sql.prompt_loader import load_prompt


async def generate_sql(state: DataAgentState, ctx: DataAgentContext) -> dict:
    """LLM 根据筛选后的表/列/指标生成 SQL"""
    writer = ctx.get("writer")
    if writer:
        writer({"type": "progress", "step": "生成SQL", "status": "running"})

    try:
        llm = ctx["llm"]

        # 序列化表信息
        tables_list = []
        for t in state["table_infos"]:
            cols = [{"name": c.name, "type": c.type, "role": c.role,
                     "description": c.description, "examples": c.examples[:5] if c.examples else None}
                    for c in t.columns]
            tables_list.append({"name": t.name, "role": t.role,
                               "description": t.description, "columns": cols})
        table_infos_str = yaml.dump(tables_list, allow_unicode=True, default_flow_style=False)

        # 序列化指标
        metrics_list = [{"name": m.name, "description": m.description} for m in state["metric_infos"]]
        metric_infos_str = yaml.dump(metrics_list, allow_unicode=True, default_flow_style=False)

        date_info = state.get("date_info", {})
        db_info = state.get("db_info", {})

        # 加载并填充 prompt
        system_prompt = load_prompt("generate_sql")
        system_prompt = system_prompt.replace("{table_infos}", table_infos_str)
        system_prompt = system_prompt.replace("{metric_infos}", metric_infos_str)
        system_prompt = system_prompt.replace("{date}", date_info.get("date", ""))
        system_prompt = system_prompt.replace("{quarter}", date_info.get("quarter", ""))
        system_prompt = system_prompt.replace("{version}", db_info.get("version", ""))

        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["query"]),
        ])

        sql = response.content.strip()
        if "```" in sql:
            sql = sql.split("```")[1]
            if sql.startswith("sql"):
                sql = sql[3:]
            sql = sql.strip()

        from src.core.logger import logger
        logger.info(f"[generate_sql] 生成 SQL ({len(sql)} 字符): {sql[:200]}")

        if writer:
            writer({"type": "progress", "step": "生成SQL", "status": "success"})
        return {"sql": sql}
    except Exception as e:
        if writer:
            writer({"type": "progress", "step": "生成SQL", "status": "error"})
        raise
