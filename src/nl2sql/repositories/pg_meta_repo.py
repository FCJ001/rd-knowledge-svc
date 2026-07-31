# ============================================================
# PostgreSQL 元数据仓库 — 查询 NL2SQL 表/列/指标
# ============================================================

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.nl2sql.entities import ColumnInfo, TableInfo, MetricInfo
from src.nl2sql.models import Nl2sqlTable, Nl2sqlColumn, Nl2sqlMetric


class PgMetaRepository:
    """从 rd_knowledge 库读取 NL2SQL 元数据"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all_tables(self) -> list[TableInfo]:
        result = await self.db.execute(select(Nl2sqlTable))
        tables = []
        for t in result.scalars().all():
            tables.append(TableInfo(
                id=str(t.id),
                name=t.table_name,
                role=t.role,
                description=t.description or "",
            ))
        return tables

    async def get_columns_by_table(self, table_id: str) -> list[ColumnInfo]:
        result = await self.db.execute(
            select(Nl2sqlColumn, Nl2sqlTable.table_name).join(
                Nl2sqlTable, Nl2sqlColumn.table_id == Nl2sqlTable.id
            ).where(Nl2sqlColumn.table_id == int(table_id))
        )
        return [_to_column_info(c, tname) for c, tname in result.all()]

    async def get_all_columns(self) -> list[ColumnInfo]:
        result = await self.db.execute(
            select(Nl2sqlColumn, Nl2sqlTable.table_name).join(
                Nl2sqlTable, Nl2sqlColumn.table_id == Nl2sqlTable.id
            )
        )
        return [_to_column_info(c, tname) for c, tname in result.all()]

    async def get_key_columns(self, table_id: str) -> list[ColumnInfo]:
        """获取主键和外键列"""
        result = await self.db.execute(
            select(Nl2sqlColumn, Nl2sqlTable.table_name).join(
                Nl2sqlTable, Nl2sqlColumn.table_id == Nl2sqlTable.id
            ).where(
                Nl2sqlColumn.table_id == int(table_id),
                Nl2sqlColumn.role.in_(["primary_key", "foreign_key"]),
            )
        )
        return [_to_column_info(c, tname) for c, tname in result.all()]

    async def get_table_by_name(self, name: str) -> TableInfo | None:
        result = await self.db.execute(
            select(Nl2sqlTable).where(Nl2sqlTable.table_name == name)
        )
        t = result.scalar_one_or_none()
        if not t:
            return None
        cols = await self.get_columns_by_table(str(t.id))
        return TableInfo(
            id=str(t.id), name=t.table_name, role=t.role,
            description=t.description or "", columns=cols,
        )

    async def get_all_metrics(self) -> list[MetricInfo]:
        result = await self.db.execute(select(Nl2sqlMetric))
        return [
            MetricInfo(
                id=str(m.id), name=m.metric_name,
                description=m.description or "",
                relevant_columns=m.relevant_columns or [],
                alias=m.aliases or [],
            )
            for m in result.scalars().all()
        ]

    async def get_metric_by_name(self, name: str) -> MetricInfo | None:
        result = await self.db.execute(
            select(Nl2sqlMetric).where(Nl2sqlMetric.metric_name == name)
        )
        m = result.scalar_one_or_none()
        if not m:
            return None
        return MetricInfo(
            id=str(m.id), name=m.metric_name,
            description=m.description or "",
            relevant_columns=m.relevant_columns or [],
            alias=m.aliases or [],
        )


def _to_column_info(c: Nl2sqlColumn, table_name: str = "") -> ColumnInfo:
    return ColumnInfo(
        id=f"{table_name}.{c.column_name}",
        name=c.column_name,
        type=c.column_type or "",
        role=c.role,
        description=c.description or "",
        alias=c.aliases or [],
        examples=c.examples or [],
        table_id=str(c.table_id),
    )
