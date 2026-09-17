# ============================================================
# PostgreSQL 异步连接 — 本服务自有库 rd_knowledge
#
# 业务代码全走这里的 get_db()。
# ★ pool_pre_ping=True 必须开：容器重启后连接池里的旧连接是死的。
#
# 连接池：DB_POOL_ENABLED=true（默认）用 QueuePool —— 生产 QPS 下
# 每请求新建 PG 连接会造成连接风暴。FastAPI 单事件循环运行没问题；
# 设 DB_POOL_ENABLED=false 回退 NullPool（pytest 多循环等场景）。
# ============================================================

from collections.abc import AsyncGenerator

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.core.config import get_settings

settings = get_settings()

_engine_kwargs: dict = {"echo": settings.APP_DEBUG}

if settings.DB_POOL_ENABLED:
    _engine_kwargs.update(
        pool_pre_ping=True,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_recycle=1800,  # 30 分钟回收，防中间件静默掐空闲连接
    )
else:
    _engine_kwargs.update(poolclass=NullPool, pool_pre_ping=True)

engine = create_async_engine(settings.DATABASE_URL, **_engine_kwargs)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：一个请求一个 session，正常结束提交，异常回滚。

    ★ 已知权衡：yield 依赖的 commit 在响应发送之后执行，commit 失败时
      客户端已收到 200 —— 这里把失败显式打 ERROR 日志（监控可告警），
      不允许静默丢数据。写操作想"先落库再返回 200"的端点用 commit_or_log。"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def commit_or_log(session: AsyncSession) -> bool:
    """端点内主动提交。成功返回 True；失败回滚、打 ERROR 日志并返回 False。"""
    try:
        await session.commit()
        return True
    except Exception:
        await session.rollback()
        logger.exception("数据库提交失败（已回滚）")
        return False
