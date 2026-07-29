# ============================================================
# PostgreSQL 异步连接 — 本服务自有库 rd_knowledge
#
# 业务代码全走这里的 get_db()。
# ★ pool_pre_ping=True 必须开：容器重启后连接池里的旧连接是死的。
# ============================================================

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.core.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.APP_DEBUG,
    poolclass=NullPool,  # ★ Gradio event loop 不兼容连接池
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：一个请求一个 session，正常结束提交，异常回滚"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
