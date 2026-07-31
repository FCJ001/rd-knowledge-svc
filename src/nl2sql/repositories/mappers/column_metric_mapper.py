# ============================================================
# ColumnMetric mapper — entity ↔ ORM model
# ============================================================

from dataclasses import asdict

from src.nl2sql.entities.column_metric import ColumnMetric
from src.nl2sql.models.column_metric import ColumnMetricModel


class ColumnMetricMapper:
    @staticmethod
    def to_entity(model: ColumnMetricModel) -> ColumnMetric:
        return ColumnMetric(
            column_id=model.column_id,
            metric_id=model.metric_id,
        )

    @staticmethod
    def to_model(entity: ColumnMetric) -> ColumnMetricModel:
        return ColumnMetricModel(**asdict(entity))
