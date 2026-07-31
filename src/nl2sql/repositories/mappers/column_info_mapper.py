# ============================================================
# ColumnInfo mapper — entity ↔ ORM model
# ============================================================

from dataclasses import asdict

from src.nl2sql.entities import ColumnInfo
from src.nl2sql.models.nl2sql_column import Nl2sqlColumn


class ColumnInfoMapper:
    @staticmethod
    def to_entity(model: Nl2sqlColumn, table_name: str = "") -> ColumnInfo:
        return ColumnInfo(
            id=f"{table_name}.{model.column_name}" if table_name else str(model.id),
            name=model.column_name,
            type=model.column_type or "",
            role=model.role,
            description=model.description or "",
            alias=model.aliases or [],
            examples=model.examples or [],
            table_id=str(model.table_id),
        )

    @staticmethod
    def to_model(entity: ColumnInfo) -> Nl2sqlColumn:
        return Nl2sqlColumn(**asdict(entity))
