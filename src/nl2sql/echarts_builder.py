# ============================================================
# ECharts option JSON 构建器（纯后端产出，不做渲染）
# 原文照搬 tiangong-agent
# ============================================================

from __future__ import annotations

import pandas as pd


def to_echarts_option(data: list[dict], config: dict) -> dict:
    """根据推荐 config + 查询数据生成 ECharts option JSON"""
    if not data or config.get("chart_type") == "table":
        return _table_option(data, config)

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

    builders = {
        "bar": _bar_option,
        "line": _line_option,
        "pie": _pie_option,
        "scatter": _scatter_option,
        "heatmap": _heatmap_option,
    }

    builder = builders.get(chart_type, _table_option)
    return builder(df, title, x_col, y_col, color_col)


def _base_option(title: str) -> dict:
    return {
        "title": {"text": title, "left": "center"},
        "tooltip": {"trigger": "axis"},
        "toolbox": {
            "feature": {"saveAsImage": {}, "dataZoom": {}, "restore": {}}
        },
    }


def _bar_option(df, title, x_col, y_col, color_col) -> dict:
    option = _base_option(title)
    if color_col and color_col in df.columns:
        groups = df.groupby(color_col)
        categories = df[x_col].unique().tolist()
        series = []
        for name, group in groups:
            values = []
            for cat in categories:
                row = group[group[x_col] == cat]
                values.append(float(row[y_col].iloc[0]) if len(row) > 0 else 0)
            series.append({"name": str(name), "type": "bar", "data": values})
        option["legend"] = {"top": "bottom"}
        option["xAxis"] = {"type": "category", "data": [str(c) for c in categories]}
        option["yAxis"] = {"type": "value"}
        option["series"] = series
    else:
        option["xAxis"] = {"type": "category", "data": df[x_col].astype(str).tolist(), "axisLabel": {"rotate": 30}}
        option["yAxis"] = {"type": "value"}
        option["series"] = [{"type": "bar", "data": [_to_number(v) for v in df[y_col].tolist()]}]
    return option


def _line_option(df, title, x_col, y_col, color_col) -> dict:
    option = _base_option(title)
    if color_col and color_col in df.columns:
        groups = df.groupby(color_col)
        categories = df[x_col].unique().tolist()
        series = []
        for name, group in groups:
            values = []
            for cat in categories:
                row = group[group[x_col] == cat]
                values.append(float(row[y_col].iloc[0]) if len(row) > 0 else 0)
            series.append({"name": str(name), "type": "line", "data": values, "smooth": True})
        option["legend"] = {"top": "bottom"}
        option["xAxis"] = {"type": "category", "data": [str(c) for c in categories]}
        option["yAxis"] = {"type": "value"}
        option["series"] = series
    else:
        option["xAxis"] = {"type": "category", "data": df[x_col].astype(str).tolist()}
        option["yAxis"] = {"type": "value"}
        option["series"] = [{"type": "line", "data": [_to_number(v) for v in df[y_col].tolist()], "smooth": True}]
    return option


def _pie_option(df, title, x_col, y_col, color_col) -> dict:
    option = _base_option(title)
    option["tooltip"] = {"trigger": "item", "formatter": "{b}: {c} ({d}%)"}
    option["legend"] = {"orient": "vertical", "left": "left", "top": "middle"}
    pie_data = [{"name": str(row[x_col]), "value": _to_number(row[y_col])} for _, row in df.iterrows()]
    option["series"] = [{
        "type": "pie", "radius": ["40%", "70%"], "center": ["60%", "50%"],
        "data": pie_data, "emphasis": {"itemStyle": {"shadowBlur": 10}},
    }]
    return option


def _scatter_option(df, title, x_col, y_col, color_col) -> dict:
    option = _base_option(title)
    option["tooltip"]["trigger"] = "item"
    option["xAxis"] = {"type": "value", "name": x_col}
    option["yAxis"] = {"type": "value", "name": y_col}
    if color_col and color_col in df.columns:
        groups = df.groupby(color_col)
        series = []
        for name, group in groups:
            series.append({"name": str(name), "type": "scatter",
                           "data": group[[x_col, y_col]].values.tolist()})
        option["legend"] = {"top": "bottom"}
        option["series"] = series
    else:
        option["series"] = [{"type": "scatter", "data": df[[x_col, y_col]].values.tolist()}]
    return option


def _heatmap_option(df, title, x_col, y_col, color_col) -> dict:
    option = _base_option(title)
    option["tooltip"]["trigger"] = "item"
    if x_col and y_col and color_col and color_col in df.columns:
        x_cats = df[x_col].unique().tolist()
        y_cats = df[y_col].unique().tolist()
        heat_data = []
        for _, row in df.iterrows():
            xi = x_cats.index(row[x_col])
            yi = y_cats.index(row[y_col])
            heat_data.append([xi, yi, _to_number(row[color_col])])
        option["xAxis"] = {"type": "category", "data": [str(c) for c in x_cats]}
        option["yAxis"] = {"type": "category", "data": [str(c) for c in y_cats]}
        option["visualMap"] = {"min": 0, "max": max(d[2] for d in heat_data) if heat_data else 1, "calculable": True}
        option["series"] = [{"type": "heatmap", "data": heat_data}]
    else:
        return _table_option(df.to_dict("records"), {"title": title})
    return option


def _table_option(data, config: dict) -> dict:
    return {
        "chart_type": "table",
        "title": config.get("title", "查询结果"),
        "data": _serialize_data(data if isinstance(data, list) else []),
    }


def _serialize_data(data: list[dict]) -> list[dict]:
    """递归转换 datetime/Decimal 等不可序列化类型为字符串"""
    import datetime as _dt
    import decimal as _dec

    result = []
    for row in data:
        converted = {}
        for k, v in row.items():
            if isinstance(v, (_dt.datetime, _dt.date, _dt.time)):
                converted[k] = str(v)
            elif isinstance(v, _dec.Decimal):
                converted[k] = float(v)
            elif isinstance(v, bytes):
                converted[k] = v.decode("utf-8", errors="replace")
            else:
                converted[k] = v
        result.append(converted)
    return result


def _to_number(val) -> float | int:
    try:
        f = float(val)
        return int(f) if f == int(f) else f
    except (ValueError, TypeError):
        return 0
