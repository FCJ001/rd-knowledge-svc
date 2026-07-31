# ============================================================
# Node ④b — LLM 筛选必要指标
# ============================================================

import json

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from src.nl2sql.state import DataAgentState
from src.nl2sql.context import DataAgentContext
from src.nl2sql.prompt_loader import load_prompt


async def filter_metrics(state: DataAgentState, ctx: DataAgentContext) -> dict:
    """LLM 从候选指标中筛选计算所需的最小集合"""
    writer = ctx.get("writer")
    if writer:
        writer({"type": "progress", "step": "过滤指标", "status": "running"})

    try:
        llm = ctx["llm"]
        metric_infos = state["metric_infos"]

        # 序列化为 YAML
        metrics_list = [
            {"name": m.name, "description": m.description, "aliases": m.alias}
            for m in metric_infos
        ]
        metrics_yaml = yaml.dump(metrics_list, allow_unicode=True, default_flow_style=False)

        response = await llm.ainvoke([
            SystemMessage(content=load_prompt("filter_metric_info").replace("{metrics_yaml}", metrics_yaml)),
            HumanMessage(content=state["query"]),
        ])

        try:
            selection = json.loads(response.content.strip())
            if not isinstance(selection, list):
                selection = []
        except json.JSONDecodeError:
            selection = []

        from src.core.logger import logger

        if not selection:
            logger.info("[filter_metrics] LLM 返回无效，保留全部指标")
            if writer:
                writer({"type": "progress", "step": "过滤指标", "status": "success"})
            return {}

        keep = set(selection)
        filtered = [m for m in metric_infos if m.name in keep]
        logger.info(f"[filter_metrics] 筛选后 {len(filtered)} 个指标: {[m.name for m in filtered]}")

        if writer:
            writer({"type": "progress", "step": "过滤指标", "status": "success"})
        return {"metric_infos": filtered}
    except Exception as e:
        if writer:
            writer({"type": "progress", "step": "过滤指标", "status": "error"})
        raise
