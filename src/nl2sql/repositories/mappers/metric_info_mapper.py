# ============================================================
# MetricInfo mapper — entity ↔ ORM model
# ============================================================

from dataclasses import asdict

from src.nl2sql.entities import MetricInfo
from src.nl2sql.models.nl2sql_metric import Nl2sqlMetric


class MetricInfoMapper:
    @staticmethod
    def to_entity(model: Nl2sqlMetric) -> MetricInfo:
        return MetricInfo(
            id=str(model.id),
            name=model.metric_name,
            description=model.description or "",
            relevant_columns=model.relevant_columns or [],
            alias=model.aliases or [],
        )

    @staticmethod
    def to_model(entity: MetricInfo) -> Nl2sqlMetric:
        return Nl2sqlMetric(**asdict(entity))
