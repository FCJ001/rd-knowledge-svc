# ============================================================
# Elasticsearch 列值全文检索 — IK 中文分词
# ============================================================

from elasticsearch import AsyncElasticsearch

from src.nl2sql.entities import ValueInfo

VALUE_INDEX = "nl2sql_values"


class ESValueRepository:
    """ES 全文检索列值枚举"""

    def __init__(self, client: AsyncElasticsearch):
        self.client = client

    async def search(self, keyword: str, size: int = 10) -> list[ValueInfo]:
        if not await self._index_exists():
            return []

        result = await self.client.search(
            index=VALUE_INDEX,
            body={
                "query": {
                    "match": {
                        "value": {
                            "query": keyword,
                            "operator": "or",
                        }
                    }
                },
                "size": size,
            },
        )

        values: dict[str, ValueInfo] = {}
        for hit in result["hits"]["hits"]:
            src = hit["_source"]
            vid = src.get("id", "")
            if vid not in values:
                values[vid] = ValueInfo(
                    id=vid,
                    value=src.get("value", ""),
                    column_id=src.get("column_id", ""),
                )
        return list(values.values())

    async def _index_exists(self) -> bool:
        try:
            return await self.client.indices.exists(index=VALUE_INDEX)
        except Exception:
            return False
