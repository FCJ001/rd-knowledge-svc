from sqlalchemy import Column, String, BigInteger, Text
from src.core.base_model import BaseModel


class Nl2sqlTable(BaseModel):
    __tablename__ = "nl2sql_tables"

    table_name = Column(String(100), unique=True, nullable=False, comment="表名")
    role = Column(String(20), nullable=False, comment="fact / dim")
    description = Column(Text, default="", comment="表说明")
