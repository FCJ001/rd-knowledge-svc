# ============================================================
# Neo4j 连接 — 读项目一的追溯图谱，只读账号
# ============================================================

from functools import lru_cache

from neo4j import AsyncGraphDatabase
from neo4j import AsyncDriver as Neo4jAsyncDriver

from src.core.config import get_settings


@lru_cache(maxsize=1)
def get_neo4j_driver() -> Neo4jAsyncDriver:
    """返回 Neo4j 异步 driver 单例（只读账户）"""
    settings = get_settings()
    driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    return driver
