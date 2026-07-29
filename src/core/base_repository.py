# ============================================================
# 通用 CRUD Repository（仿 MyBatis BaseMapper）
# ============================================================

from typing import TypeVar, Generic, Type, Sequence

from sqlalchemy import select, delete, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.base_model import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseRepository(Generic[T]):
    def __init__(self, model: Type[T], db: AsyncSession):
        self.model = model
        self.db = db

    async def get_by_id(self, id: int) -> T | None:
        return await self.db.get(self.model, id)

    async def get_all(self, offset: int = 0, limit: int = 100) -> Sequence[T]:
        stmt = select(self.model).offset(offset).limit(limit)
        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def create(self, obj: T) -> T:
        self.db.add(obj)
        await self.db.flush()
        await self.db.refresh(obj)
        return obj

    async def update(self, obj: T) -> T:
        await self.db.flush()
        await self.db.refresh(obj)
        return obj

    async def delete(self, obj: T) -> None:
        await self.db.delete(obj)
        await self.db.flush()

    async def delete_by_id(self, id: int) -> None:
        stmt = delete(self.model).where(self.model.id == id)
        await self.db.execute(stmt)

    async def get_page(
        self,
        offset: int = 0,
        limit: int = 20,
        keyword: str | None = None,
        search_fields: list[str] | None = None,
    ) -> tuple[list[T], int]:
        stmt = select(self.model)

        if keyword and search_fields:
            conditions = []
            for field_name in search_fields:
                column = getattr(self.model, field_name, None)
                if column is not None:
                    conditions.append(column.like(f"%{keyword}%"))
            if conditions:
                stmt = stmt.where(or_(*conditions))

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_result = await self.db.execute(count_stmt)
        total = total_result.scalar_one()

        stmt = stmt.offset(offset).limit(limit).order_by(self.model.id.asc())
        result = await self.db.execute(stmt)
        items = list(result.scalars().all())

        return items, total
