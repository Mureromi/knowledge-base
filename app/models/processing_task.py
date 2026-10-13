"""阶段 3：文档处理任务表。

为什么要单独建表（§4.2），而不是只复用 documents.status：
- 失败原因有地方放（扫描件？格式损坏？超时？），塞进 documents 会污染业务表
- 可追溯重试历史：同一文档重新解析会产生多条记录
- 统计与告警：失败率、平均解析耗时都需要任务级数据

两者职责不同：documents.status 是**当前状态的投影**，本表是**事件流水**。
"""

from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

# 任务状态机（§4.3）
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"

TERMINAL_STATUSES = frozenset({STATUS_SUCCEEDED, STATUS_FAILED})
ACTIVE_STATUSES = frozenset({STATUS_PENDING, STATUS_RUNNING})


class ProcessingTask(Base):
    """一次文档处理任务。与 documents 一对多：重新解析会新建一条记录。"""

    __tablename__ = "processing_tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    # ★ 验收修复：CASCADE → SET NULL。
    #   原设计在删文档时把任务记录一并删掉，与 §4.2 声称的"可追溯重试历史、统计失败率"
    #   直接矛盾——文档一删，历史就出现缺口。改为置空后任务流水得以保留。
    #   （与 chat_logs.session_id 的策略一致：审计数据与业务数据的生命周期本就不同。）
    document_id: Mapped[int | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"), nullable=True, index=True
    )
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )

    # pending / running / succeeded / failed
    status: Mapped[str] = mapped_column(String(16), default=STATUS_PENDING, index=True)
    # ARQ 的 job_id，便于从日志反查、以及做幂等去重
    job_id: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    attempt: Mapped[int] = mapped_column(Integer, default=0)

    # 失败原因（给用户看的，不要塞堆栈）
    error: Mapped[str | None] = mapped_column(Text, default=None)
    # 解析出的字符数、切片数，便于排查"为什么检索不到"
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )

    document: Mapped["Document | None"] = relationship(back_populates="tasks")

    @property
    def is_terminal(self) -> bool:
        """前端轮询到 True 就可以停止（§5.4）。"""
        return self.status in TERMINAL_STATUSES

    def is_stale(self, stale_seconds: int, now: datetime | None = None) -> bool:
        """僵死判定：worker 被 kill -9 / 机器重启 → 永远停在 running（§11.1）。

        统一用 UTC aware 比较：SQLite 取回的是 naive，PostgreSQL 是 aware，
        混用会抛 "can't compare offset-naive and offset-aware datetimes"。
        """
        if self.status != STATUS_RUNNING or self.started_at is None:
            return False
        now = now or datetime.now(UTC)
        started = self.started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=UTC)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        return (now - started).total_seconds() > stale_seconds
