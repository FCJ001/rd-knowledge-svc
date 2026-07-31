from sqlalchemy import Column, String, BigInteger, Text, JSON, ForeignKey
from src.core.base_model import BaseModel


class Nl2sqlColumn(BaseModel):
    __tablename__ = "nl2sql_columns"

    table_id = Column(BigInteger, ForeignKey("nl2sql_tables.id"), nullable=False, comment="关联表ID")
    column_name = Column(String(200), nullable=False, comment="列名")
    column_type = Column(String(50), default="", comment="列类型")
    role = Column(String(20), nullable=False, comment="pk / fk / dimension / measure / date")
    description = Column(Text, default="", comment="列说明")
    aliases = Column(JSON, default=list, comment="别名列表")
    examples = Column(JSON, default=list, comment="枚举值示例")
