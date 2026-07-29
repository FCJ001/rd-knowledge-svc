# ============================================================
# 图表推荐 + Plotly 图表生成
# LLM 根据查询结果推荐图表类型，然后用 Plotly 渲染。
# ============================================================

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from loguru import logger

from src.nl2sql.prompts import CHART_ADVISOR_PROMPT


async def recommend_chart(
    question: str,
    data: list[dict],
    columns: list[str],
    llm: BaseChatModel,
) -> dict:
    """LLM 推荐图表类型和配置，返回 {chart_type, title, x_column, y_column, ...}"""
    if not data:
        return {"chart_type": "table", "title": "无数据", "description": "查询结果为空"}

    df = pd.DataFrame(data)
    preview = df.head(5).to_string(index=False)

    prompt = CHART_ADVISOR_PROMPT.format(
        question=question,
        preview=preview,
        columns=columns,
        row_count=len(data),
    )

    response = await llm.ainvoke([SystemMessage(content=prompt)])
    content = response.content.strip()

    try:
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        return json.loads(content)
    except Exception as e:
        logger.warning(f"图表推荐解析失败: {e}")
        return {"chart_type": "table", "title": "查询结果", "description": ""}


def render_chart(data: list[dict], config: dict) -> go.Figure | None:
    """根据推荐配置生成 Plotly 图表"""
    if not data or config.get("chart_type") == "table":
        return None

    df = pd.DataFrame(data)
    chart_type = config.get("chart_type", "bar")
    title = config.get("title", "")
    x_col = config.get("x_column")
    y_col = config.get("y_column")
    color_col = config.get("color_column")

    if x_col and x_col not in df.columns:
        x_col = df.columns[0]
    if y_col and y_col not in df.columns:
        y_col = df.columns[-1] if len(df.columns) > 1 else df.columns[0]

    try:
        if chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title)
        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title)
        elif chart_type == "pie":
            fig = px.pie(df, names=x_col, values=y_col, title=title)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title)
        elif chart_type == "heatmap":
            if x_col and y_col and color_col:
                pivot = df.pivot_table(index=y_col, columns=x_col, values=color_col, aggfunc="sum")
                fig = px.imshow(pivot, title=title, aspect="auto")
            else:
                numeric_cols = df.select_dtypes(include="number")
                fig = px.imshow(numeric_cols.corr(), title=title or "相关性热力图", aspect="auto")
        else:
            return None

        fig.update_layout(
            template="plotly_white",
            font=dict(family="Microsoft YaHei, sans-serif"),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return fig

    except Exception as e:
        logger.warning(f"图表渲染失败: {e}")
        return None
