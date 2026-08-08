# ============================================================
# Neo4j 连接 — 读项目一的追溯图谱，只读账号
# ★ 不用单例/lru_cache：异步 driver 绑定创建它的事件循环，
#   跨循环复用会报 "Future attached to different loop"
# ============================================================

from neo4j import AsyncGraphDatabase
from neo4j import AsyncDriver as Neo4jAsyncDriver

from src.core.config import get_settings


def get_neo4j_driver() -> Neo4jAsyncDriver:
    """返回 Neo4j 异步 driver。每次调用新建，避免跨事件循环复用"""
    settings = get_settings()
    return AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
