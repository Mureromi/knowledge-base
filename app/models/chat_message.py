"""单条消息（阶段 2 §2.1）。

role 只有 user / assistant 两种（system 是共享常量，不入库——它变了应全局失效，
而不是让每个会话的历史指纹跟着变）。
"""

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    # 这一轮引用了哪些切片，用 JSON 存 id 列表（和 chat_logs 一致）
    cited_chunk_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    session: Mapped["ChatSession"] = relationship(back_populates="messages")
