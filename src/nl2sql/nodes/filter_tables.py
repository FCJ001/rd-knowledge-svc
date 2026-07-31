# ============================================================
# Node ④a — LLM 筛选必要表/列
# ============================================================

import json

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from src.nl2sql.state import DataAgentState
from src.nl2sql.context import DataAgentContext
from src.nl2sql.prompt_loader import load_prompt


async def filter_tables(state: DataAgentState, ctx: DataAgentContext) -> dict:
    """LLM 从候选表/列中筛选查询所需的最小集合"""
    writer = ctx.get("writer")
    if writer:
        writer({"type": "progress", "step": "过滤表格", "status": "running"})

    try:
        llm = ctx["llm"]
        table_infos = state["table_infos"]

        # 序列化为 YAML
        tables_dict = {}
        for t in table_infos:
            cols = []
            for c in t.columns:
                cols.append({
                    "name": c.name,
                    "type": c.type,
                    "role": c.role,
                    "description": c.description,
                    "examples": c.examples if c.examples else None,
                })
            tables_dict[t.name] = {
                "role": t.role,
                "description": t.description,
                "columns": cols,
            }
        tables_yaml = yaml.dump(tables_dict, allow_unicode=True, default_flow_style=False)

        response = await llm.ainvoke([
            SystemMessage(content=load_prompt("filter_table_info").replace("{tables_yaml}", tables_yaml)),
            HumanMessage(content=state["query"]),
        ])

        try:
            selection = json.loads(response.content.strip())
            if not isinstance(selection, dict):
                selection = {}
        except json.JSONDecodeError:
            selection = {}

        from src.core.logger import logger

        if not selection:
            logger.info("[filter_tables] LLM 返回无效，保留全部表")
            if writer:
                writer({"type": "progress", "step": "过滤表格", "status": "success"})
            return {}

        filtered = []
        for t in table_infos:
            if t.name not in selection:
                continue
            selected_cols = set(selection[t.name])
            # 始终保留主键和外键
            for c in t.columns:
                if c.role in ("primary_key", "foreign_key"):
                    selected_cols.add(c.name)
            t.columns = [c for c in t.columns if c.name in selected_cols]
            if t.columns:
                filtered.append(t)

        total_cols = sum(len(t.columns) for t in filtered)
        logger.info(f"[filter_tables] 筛选后 {len(filtered)} 张表, {total_cols} 列")

        if writer:
            writer({"type": "progress", "step": "过滤表格", "status": "success"})
        return {"table_infos": filtered}
    except Exception as e:
        if writer:
            writer({"type": "progress", "step": "过滤表格", "status": "error"})
        raise
