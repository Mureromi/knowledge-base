"""数据库会话与初始化。

- engine：全局异步引擎（阶段 0 用 create_all 建表，阶段 1 换 alembic）
- get_db：FastAPI 依赖，请求级 session，自动 commit / rollback
- init_models / dispose_engine：由 main.py 的 lifespan 调用
"""

import json
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.models.base import Base


def _make_engine(url: str) -> AsyncEngine:
    """按 URL 分派驱动参数。

    SQLite：需要 check_same_thread=False。
    PostgreSQL：需要显式 json_serializer —— asyncpg **不像 psycopg2 那样自动序列化
        JSON 列**，不配置的话插入 list/dict 会报：
            descriptor 'encode' for 'str' objects doesn't apply to a 'list' object
        这个坑在阶段 4 切 PostgreSQL 时实测踩到，且**只在真正写入 JSON 列时暴露**。
    """
    kwargs: dict = {"echo": False, "future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["json_serializer"] = lambda obj: json.dumps(obj, ensure_ascii=False)
        kwargs["json_deserializer"] = json.loads
    return create_async_engine(url, **kwargs)


engine: AsyncEngine = _make_engine(settings.database_url)

SessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖：每个请求独享一个 session。

    正常结束 → commit；抛异常 → rollback 并向上冒泡。
    """
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_models() -> None:
    """阶段 0 简化：直接 create_all。阶段 1 引入 alembic 后移除。"""
    # 触发所有模型的导入，保证 metadata 完整
    from app import models  # noqa: F401  isort: skip

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    await engine.dispose()
