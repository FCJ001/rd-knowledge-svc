# ============================================================
# Node ③ — 合并检索结果：补 PK/FK、按表分组、值挂列
# ============================================================

from src.nl2sql.entities import ColumnInfo, TableInfo, MetricInfo
from src.nl2sql.state import DataAgentState
from src.nl2sql.context import DataAgentContext


async def merge_info(state: DataAgentState, ctx: DataAgentContext) -> dict:
    """将三路检索结果合并为结构化的 table_infos 和 metric_infos"""
    writer = ctx.get("writer")
    if writer:
        writer({"type": "progress", "step": "合并召回信息", "status": "running"})

    try:
        pg = ctx["pg_meta_repo"]
        cols = state.get("retrieved_columns", [])
        vals = state.get("retrieved_values", [])
        metrics = state.get("retrieved_metrics", [])

        # 1. 从指标的相关列扩展列信息
        retrieved_map: dict[str, ColumnInfo] = {}
        for col in cols:
            retrieved_map[col.id] = col
        for metric in metrics:
            for relevant_col_id in metric.relevant_columns:
                if relevant_col_id not in retrieved_map:
                    try:
                        tname, cname = relevant_col_id.rsplit(".", 1)
                        t = await pg.get_table_by_name(tname)
                        if t:
                            for tc in t.columns:
                                if tc.id == relevant_col_id:
                                    retrieved_map[relevant_col_id] = tc
                                    break
                    except Exception:
                        continue

        # 2. 获取所有候选表的全量列（从 PG 元数据）
        seen_tables: set[str] = set()
        for col in retrieved_map.values():
            table_name = col.table_name or col.id.split(".", 1)[0]
            if table_name not in seen_tables:
                seen_tables.add(table_name)

        # 冷启动：如果没有从 Milvus 召回任何表，回退到全量表
        if not seen_tables:
            all_tables = await pg.get_all_tables()
            for t in all_tables:
                seen_tables.add(t.name)

        all_columns: list[ColumnInfo] = []
        for tname in seen_tables:
            t = await pg.get_table_by_name(tname)
            if t:
                all_columns.extend(t.columns)
                key_cols = await pg.get_key_columns(t.id)
                for kc in key_cols:
                    if kc.id not in {c.id for c in all_columns}:
                        all_columns.append(kc)

        # 3. 将值挂载到对应列的 examples
        col_map = {c.id: c for c in all_columns}
        for v in vals:
            if v.column_id in col_map:
                col = col_map[v.column_id]
                if v.value not in col.examples:
                    col.examples.append(v.value)

        # 4. 按表分组
        table_groups: dict[str, list[ColumnInfo]] = {}
        for c in all_columns:
            table_name = c.table_name or c.id.split(".", 1)[0]
            table_groups.setdefault(table_name, []).append(c)

        table_infos: list[TableInfo] = []
        for tname, tcols in table_groups.items():
            t = await pg.get_table_by_name(tname)
            table_infos.append(TableInfo(
                id=t.id if t else tname,
                name=tname,
                role=t.role if t else "",
                description=t.description if t else "",
                columns=tcols,
            ))

        # 5. 指标 — 如果有召回，用召回的；否则全量
        metric_infos: list[MetricInfo] = list(metrics) if metrics else await pg.get_all_metrics()

        from src.core.logger import logger
        total_cols = sum(len(t.columns) for t in table_infos)
        logger.info(f"[merge_info] {len(table_infos)} 张表, {total_cols} 列, {len(metric_infos)} 个指标")

        if writer:
            writer({"type": "progress", "step": "合并召回信息", "status": "success"})
        return {"table_infos": table_infos, "metric_infos": metric_infos}
    except Exception as e:
        if writer:
            writer({"type": "progress", "step": "合并召回信息", "status": "error"})
        raise
