# ============================================================
# ColumnMetric entity — 列-指标多对多关系
# ============================================================

from dataclasses import dataclass


@dataclass
class ColumnMetric:
    column_id: str
    metric_id: str
