# ============================================================
# TableInfo mapper — entity ↔ ORM model
# ============================================================

from dataclasses import asdict

from src.nl2sql.entities import TableInfo
from src.nl2sql.models.nl2sql_table import Nl2sqlTable


class TableInfoMapper:
    @staticmethod
    def to_entity(model: Nl2sqlTable) -> TableInfo:
        return TableInfo(
            id=str(model.id),
            name=model.table_name,
            role=model.role,
            description=model.description or "",
        )

    @staticmethod
    def to_model(entity: TableInfo) -> Nl2sqlTable:
        return Nl2sqlTable(**asdict(entity))
