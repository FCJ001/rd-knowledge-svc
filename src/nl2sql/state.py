# ============================================================
# LangGraph State — 流水线节点间传递的数据
# ============================================================

from typing import TypedDict

from src.nl2sql.entities import ColumnInfo, TableInfo, MetricInfo, ValueInfo


class DateInfoState(TypedDict):
    date: str      # YYYY-MM-DD
    weekday: str   # e.g. "Thursday"
    quarter: str   # e.g. "Q3"


class DBInfoState(TypedDict):
    dialect: str   # e.g. "postgresql"
    version: str   # e.g. "16.0"


class DataAgentState(TypedDict, total=False):
    """NL2SQL 流水线状态"""
    query: str
    keywords: list[str]
    retrieved_columns: list[ColumnInfo]
    retrieved_values: list[ValueInfo]
    retrieved_metrics: list[MetricInfo]
    table_infos: list[TableInfo]
    metric_infos: list[MetricInfo]
    date_info: DateInfoState
    db_info: DBInfoState
    sql: str
    error: str  # None 表示 SQL 校验通过，非空为错误信息
    # 执行结果（由 execute_sql 节点填充，供 SSE consumer 读取）
    result_sql: str
    result_columns: list[str]
    result_data: list[dict]
    result_row_count: int
    result_summary: str
    result_error: str
