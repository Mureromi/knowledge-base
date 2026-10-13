"""User 数据访问。"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User


async def get_by_id(db: AsyncSession, user_id: int) -> User | None:
    return await db.get(User, user_id)


async def get_by_username(db: AsyncSession, username: str) -> User | None:
    stmt = select(User).where(User.username == username)
    return (await db.execute(stmt)).scalar_one_or_none()


async def create(db: AsyncSession, *, username: str, password_hash: str) -> User:
    user = User(username=username, password_hash=password_hash)
    db.add(user)
    await db.flush()      # 拿到自增 id 但不提交事务
    await db.refresh(user)
    return user
