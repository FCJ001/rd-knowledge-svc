from sqlalchemy import Column, String
from src.core.base_model import BaseModel


class ColumnMetricModel(BaseModel):
    __tablename__ = "nl2sql_column_metrics"

    column_id = Column(String(256), primary_key=True, comment="列编号")
    metric_id = Column(String(256), primary_key=True, comment="指标编号")
