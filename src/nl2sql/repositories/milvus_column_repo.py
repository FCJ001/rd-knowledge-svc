# ============================================================
# Milvus 列向量检索 — 语义搜索相关列
# ============================================================

from pymilvus import MilvusClient

from src.nl2sql.entities import ColumnInfo

COLUMN_COLLECTION = "nl2sql_columns"


class MilvusColumnRepository:
    """Milvus 向量检索 column 元数据"""

    def __init__(self, client: MilvusClient):
        self.client = client

    def search(
        self, query_vector: list[float], top_k: int = 5, threshold: float = 0.6
    ) -> list[ColumnInfo]:
        """向量相似度检索候选列，threshold 以下的结果被丢弃"""
        if not self._collection_exists():
            return []

        results = self.client.search(
            collection_name=COLUMN_COLLECTION,
            data=[query_vector],
            limit=top_k,
            output_fields=["column_id", "column_name", "column_type", "role",
                           "description", "aliases", "table_name"],
        )

        columns: dict[str, ColumnInfo] = {}
        for hit in results[0]:
            if hit["distance"] < threshold:
                continue
            entity = hit["entity"]
            col_id = entity.get("column_id", "")
            if col_id not in columns:
                columns[col_id] = ColumnInfo(
                    id=col_id,
                    name=entity.get("column_name", ""),
                    type=entity.get("column_type", ""),
                    role=entity.get("role", ""),
                    description=entity.get("description", ""),
                    alias=entity.get("aliases", []),
                )
        return list(columns.values())

    def _collection_exists(self) -> bool:
        try:
            collections = self.client.list_collections()
            return COLUMN_COLLECTION in collections
        except Exception:
            return False
