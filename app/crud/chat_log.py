"""ChatLog 数据访问。"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_log import ChatLog


async def create(db: AsyncSession, log: ChatLog) -> ChatLog:
    db.add(log)
    await db.flush()
    await db.refresh(log)
    return log
