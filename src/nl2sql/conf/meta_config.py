# ============================================================
# NL2SQL 元数据配置 dataclass — 解析 conf/nl2sql_meta.yaml
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ColumnMeta:
    name: str
    type: str
    role: str  # primary_key / foreign_key / dimension / measure / date
    description: str = ""
    alias: list[str] = field(default_factory=list)
    sync: bool = False  # 是否需要从 DB 拉取枚举值写入 ES


@dataclass
class TableMeta:
    name: str
    role: str  # fact / dim
    description: str = ""
    columns: list[ColumnMeta] = field(default_factory=list)


@dataclass
class MetricMeta:
    name: str
    description: str
    relevant_columns: list[str] = field(default_factory=list)
    alias: list[str] = field(default_factory=list)


@dataclass
class MetaConfig:
    tables: list[TableMeta] = field(default_factory=list)
    metrics: list[MetricMeta] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "MetaConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        tables = []
        for t in raw.get("tables", []):
            cols = []
            for c in t.get("columns", []):
                cols.append(ColumnMeta(
                    name=c["name"],
                    type=c.get("type", ""),
                    role=c.get("role", ""),
                    description=c.get("description", ""),
                    alias=c.get("alias", []),
                    sync=c.get("sync", False),
                ))
            tables.append(TableMeta(
                name=t["name"],
                role=t.get("role", ""),
                description=t.get("description", ""),
                columns=cols,
            ))

        metrics = []
        for m in raw.get("metrics", []):
            metrics.append(MetricMeta(
                name=m["name"],
                description=m.get("description", ""),
                relevant_columns=m.get("relevant_columns", []),
                alias=m.get("alias", []),
            ))

        return cls(tables=tables, metrics=metrics)

    @property
    def all_columns(self) -> list[tuple[TableMeta, ColumnMeta]]:
        """展开所有列，返回 (table, column) 对"""
        result = []
        for t in self.tables:
            for c in t.columns:
                result.append((t, c))
        return result
