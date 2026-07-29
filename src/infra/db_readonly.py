# ============================================================
# ALM 业务库只读连接 — NL2SQL 查询目标
#
# 连共享 PG 的 rd_agent 库（项目一的业务库）。
# 开发期用同一用户，生产环境换成只读用户 + default_transaction_read_only=on。
# ★ NullPool 解决 Gradio event loop 不兼容问题
# ============================================================

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.core.config import get_settings

settings = get_settings()

readonly_engine = create_async_engine(
    settings.ALM_DATABASE_URL,
    echo=settings.APP_DEBUG,
    poolclass=NullPool,  # ★ 不跨 event loop 复用连接
    connect_args={},
)

ReadOnlySessionLocal = async_sessionmaker(
    readonly_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db_readonly() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：只读 session，select 直接过，写操作报错"""
    async with ReadOnlySessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
