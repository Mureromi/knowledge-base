from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, LargeBinary, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(200))
    source_type: Mapped[str] = mapped_column(String(20), default="txt")
    # 阶段 3：pdf/docx 走异步路径时，上传这一刻还没解析，全文为空
    raw_text: Mapped[str | None] = mapped_column(Text, default=None)
    # ★ 阶段 3：原始上传字节落库。worker 是**另一个进程**，不能靠内存传 bytes，
    #   也不该靠临时文件（多实例部署不共享磁盘）。重新解析也因此成为可能（§11）。
    raw_content: Mapped[bytes | None] = mapped_column(LargeBinary, default=None)
    status: Mapped[str] = mapped_column(String(20), default="ready")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", passive_deletes=True
    )
    # 任务流水：★ 验收修复后**不随文档删除**（document_id 置 NULL 而非级联删除）。
    #   passive_deletes=True 让数据库去做 SET NULL，ORM 不去逐个加载/删除子行。
    #   这样任务历史（含失败原因）得以保留，可用于统计失败率与排查（§4.2 的初衷）。
    tasks: Mapped[list["ProcessingTask"]] = relationship(
        back_populates="document", passive_deletes=True
    )
