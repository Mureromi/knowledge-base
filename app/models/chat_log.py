"""问答日志：阶段 0 就要有，阶段 6 的评估价值全靠它。

每次问答都落库：token 数 / 缓存命中数 / 成本 / 延迟 / 引用切片。
"""

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    # 阶段 2：会话溯源。★ 用 SET NULL 而不是 CASCADE：
    # 会话被删时成本记录不能跟着消失——那是长期成本分析的数据，
    # "审计数据"与"业务数据"生命周期不同（§2.3）。
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_yuan: Mapped[float] = mapped_column(Float, default=0.0)
    # 阶段 1：命中时本次真实 cost_yuan=0，用 saved_cost_yuan 记录因命中而省下的钱
    saved_cost_yuan: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    model: Mapped[str] = mapped_column(String(50), default="")
    is_off_peak: Mapped[bool] = mapped_column(Boolean, default=False)
    cited_chunk_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    # 阶段 1：缓存命中标记（§9 的命中率 SQL 全靠它）
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
