# ============================================================
# Neo4j 连接 — 读项目一的追溯图谱
#
# 进程级单例：lifespan 启动时惰性创建、关闭时统一 close。
# 旧实现"每请求新建 driver"会泄漏连接池（driver 自带连接池且从不关闭）。
# 异步 driver 绑定创建它的事件循环 —— FastAPI 单循环运行没问题；
# 若在别的循环里用到（如 pytest 多循环），先 close_neo4j_driver() 再重建。
# ============================================================

from neo4j import AsyncDriver as Neo4jAsyncDriver
from neo4j import AsyncGraphDatabase

from src.core.config import get_settings

_driver: Neo4jAsyncDriver | None = None


def get_neo4j_driver() -> Neo4jAsyncDriver:
    """返回进程级单例 Neo4j 异步 driver（带连接池上限与建连超时）"""
    global _driver
    if _driver is None:
        settings = get_settings()
        _driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            connection_timeout=settings.NEO4J_CONNECTION_TIMEOUT,
            max_connection_pool_size=settings.NEO4J_MAX_POOL_SIZE,
        )
    return _driver


async def close_neo4j_driver() -> None:
    """关闭 driver（lifespan shutdown 调用）"""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
