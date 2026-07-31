# ============================================================
# LangGraph 图定义 — NL2SQL 流水线编排
#
# 三路并行召回 → merge → 两路并行过滤 → context → SQL → validate
#       → 条件路由 (error ? correct : execute)
#
# 通过 _with_ctx 适配器将 (state, ctx) 签名的节点函数
# 转换为 LangGraph 标准的 (state, config) 签名。
# ============================================================

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from src.nl2sql.context import DataAgentContext
from src.nl2sql.state import DataAgentState

# ── 原始节点函数 ────────────────────────────────────────────────
from src.nl2sql.nodes.add_context import add_context as _add_context
from src.nl2sql.nodes.correct_sql import correct_sql as _correct_sql
from src.nl2sql.nodes.execute_sql import execute_sql as _execute_sql
from src.nl2sql.nodes.extract_keywords import extract_keywords as _extract_keywords
from src.nl2sql.nodes.filter_metrics import filter_metrics as _filter_metrics
from src.nl2sql.nodes.filter_tables import filter_tables as _filter_tables
from src.nl2sql.nodes.generate_sql import generate_sql as _generate_sql
from src.nl2sql.nodes.merge_info import merge_info as _merge_info
from src.nl2sql.nodes.recall_columns import recall_columns as _recall_columns
from src.nl2sql.nodes.recall_metrics import recall_metrics as _recall_metrics
from src.nl2sql.nodes.recall_values import recall_values as _recall_values
from src.nl2sql.nodes.validate_sql import validate_sql as _validate_sql


def _with_ctx(fn):
    """适配器：将 (state, ctx) -> dict 节点函数转为 LangGraph 的 (state, config) -> dict"""
    async def wrapper(state: DataAgentState, config: RunnableConfig) -> dict:
        ctx: DataAgentContext = config.get("configurable", {})
        return await fn(state, ctx)
    # 保留原函数名以便 LangGraph 调试
    wrapper.__name__ = fn.__name__
    return wrapper


# ── 包装后的节点 ────────────────────────────────────────────────
extract_keywords = _with_ctx(_extract_keywords)
recall_columns = _with_ctx(_recall_columns)
recall_values = _with_ctx(_recall_values)
recall_metrics = _with_ctx(_recall_metrics)
merge_info = _with_ctx(_merge_info)
filter_tables = _with_ctx(_filter_tables)
filter_metrics = _with_ctx(_filter_metrics)
add_context = _with_ctx(_add_context)
generate_sql = _with_ctx(_generate_sql)
validate_sql = _with_ctx(_validate_sql)
correct_sql = _with_ctx(_correct_sql)
execute_sql = _with_ctx(_execute_sql)


# ── 图构建 ─────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    """构建并编译 NL2SQL 处理图"""

    graph = StateGraph(DataAgentState)

    # 添加节点
    graph.add_node("extract_keywords", extract_keywords)
    graph.add_node("recall_columns", recall_columns)
    graph.add_node("recall_values", recall_values)
    graph.add_node("recall_metrics", recall_metrics)
    graph.add_node("merge_info", merge_info)
    graph.add_node("filter_tables", filter_tables)
    graph.add_node("filter_metrics", filter_metrics)
    graph.add_node("add_context", add_context)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("validate_sql", validate_sql)
    graph.add_node("correct_sql", correct_sql)
    graph.add_node("execute_sql", execute_sql)

    # START → 关键词提取
    graph.add_edge(START, "extract_keywords")

    # 三路并行召回（fan-out）
    graph.add_edge("extract_keywords", "recall_columns")
    graph.add_edge("extract_keywords", "recall_values")
    graph.add_edge("extract_keywords", "recall_metrics")

    # 三路汇总（fan-in） — merge_info 会运行 3 次，
    # LangGraph 的 TypedDict last-write-wins reducer 确保最终状态正确
    graph.add_edge("recall_columns", "merge_info")
    graph.add_edge("recall_values", "merge_info")
    graph.add_edge("recall_metrics", "merge_info")

    # 两路并行过滤（fan-out）
    graph.add_edge("merge_info", "filter_tables")
    graph.add_edge("merge_info", "filter_metrics")

    # 两路汇总（fan-in）
    graph.add_edge("filter_tables", "add_context")
    graph.add_edge("filter_metrics", "add_context")

    # 串行：context → SQL → 校验
    graph.add_edge("add_context", "generate_sql")
    graph.add_edge("generate_sql", "validate_sql")

    # 条件路由：error 为空则执行，否则纠错
    def route_after_validate(state: DataAgentState) -> str:
        return "correct_sql" if state.get("error") else "execute_sql"

    graph.add_conditional_edges(
        "validate_sql",
        route_after_validate,
        {"correct_sql": "correct_sql", "execute_sql": "execute_sql"},
    )
    graph.add_edge("correct_sql", "execute_sql")
    graph.add_edge("execute_sql", END)

    return graph.compile()
