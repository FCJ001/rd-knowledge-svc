# ============================================================
# NL2SQL entity dataclasses — 节点间传递的数据对象
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ColumnInfo:
    id: str  # "{table}.{column}"
    name: str
    type: str
    role: str  # pk / fk / dimension / measure / date
    description: str = ""
    alias: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)  # ES 召回或 DB 枚举值
    table_id: str = ""

    @property
    def table_name(self) -> str:
        return self.id.split(".", 1)[0]


@dataclass
class TableInfo:
    id: str
    name: str
    role: str  # fact / dim
    description: str = ""
    columns: list[ColumnInfo] = field(default_factory=list)

    def get_pk(self) -> ColumnInfo | None:
        for c in self.columns:
            if c.role == "primary_key":
                return c
        return None

    def get_fks(self) -> list[ColumnInfo]:
        return [c for c in self.columns if c.role == "foreign_key"]

    def get_key_columns(self) -> list[ColumnInfo]:
        return [c for c in self.columns if c.role in ("primary_key", "foreign_key")]


@dataclass
class MetricInfo:
    id: str
    name: str
    description: str = ""
    relevant_columns: list[str] = field(default_factory=list)
    alias: list[str] = field(default_factory=list)


@dataclass
class ValueInfo:
    id: str  # "{column_id}.{value}"
    value: str
    column_id: str


# 重新导出 ColumnMetric
from src.nl2sql.entities.column_metric import ColumnMetric  # noqa: E402, F401
