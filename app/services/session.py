"""会话历史管理（阶段 2 §3）。

★ 滑动窗口 + 缓存键指纹必须用同一个窗口函数：
  指纹要覆盖的是"实际送进 LLM 的那部分上下文"，不多不少——
  多了（整会话历史）缓存永不命中；少了（只用问题）会返回错误答案。
"""

import hashlib
import logging
from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.chat_message import ChatMessage
from app.models.chat_session import ChatSession

logger = logging.getLogger(__name__)


async def get_or_create_session(
    db: AsyncSession, *, owner_id: int, session_id: int | None, first_question: str
) -> ChatSession:
    """取会话；session_id 为 None 则新建。越权检查在这里做。"""
    if session_id is not None:
        stmt = select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.owner_id == owner_id,      # ★ 越权检查：绝不允许访问他人会话
        )
        session = (await db.scalars(stmt)).one_or_none()
        if session is None:
            # 不区分"不存在"和"不属于你"，避免探测他人会话 id 是否存在（404 而非 403）
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": 40402, "message": "会话不存在", "detail": None},
            )
        return session

    session = ChatSession(
        owner_id=owner_id,
        title=first_question.strip()[: settings.session_title_max_chars],
    )
    db.add(session)
    await db.flush()
    await db.refresh(session)
    return session


async def load_window_messages(
    db: AsyncSession, *, session_id: int, limit_turns: int | None = None
) -> list[dict[str, str]]:
    """★ 加载最近 N 轮历史，转成 LLM messages 格式（从旧到新）。

    为什么按"轮"而不是"条"取：一轮 = 一次 user + 一次 assistant。
    只取最近 2N 条，保证不会出现"有 assistant 回答但缺 user 提问"的错位。
    """
    turns = limit_turns or settings.session_window_turns
    max_messages = turns * 2

    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.desc())
        .limit(max_messages)
    )
    rows = list((await db.scalars(stmt)).all())
    rows.reverse()          # 倒序取、正序用
    return [{"role": m.role, "content": m.content} for m in rows]


def history_fingerprint(window_messages: list[dict[str, str]]) -> str:
    """★ 窗口内历史的指纹，用于缓存键（§4.2：只哈希窗口内、不哈希整会话）。"""
    h = hashlib.sha256()
    for m in window_messages:
        h.update(m["role"].encode())
        h.update(b"\x00")
        h.update(m["content"].encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


async def append_turn(
    db: AsyncSession,
    *,
    session: ChatSession,
    question: str,
    answer: str,
    cited_chunk_ids: list[int],
    cache_hit: bool,
) -> None:
    """写入本轮 user + assistant 两条消息并刷新 last_active_at。

    ★ 调用时机：必须在"读窗口 + 算缓存键"之后（§3.2 的写入顺序坑）——
      先写本轮问题再读窗口，历史指纹会包含本轮问题本身，缓存永远不命中。
    """
    db.add(
        ChatMessage(
            session_id=session.id,
            role="user",
            content=question,
            cited_chunk_ids=[],
            cache_hit=False,
        )
    )
    db.add(
        ChatMessage(
            session_id=session.id,
            role="assistant",
            content=answer,
            cited_chunk_ids=cited_chunk_ids,
            cache_hit=cache_hit,
        )
    )
    # 会话列表按 last_active_at 排序（§3.3），不要用 created_at
    session.last_active_at = datetime.now(timezone.utc)
