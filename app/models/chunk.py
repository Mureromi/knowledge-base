from sqlalchemy import ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    # 冗余存 owner_id：检索时免 join 就能做权限过滤（阶段 1 多用户隔离的基础）
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    seq: Mapped[int] = mapped_column(Integer)            # 文档内序号，用于返回出处
    content: Mapped[str] = mapped_column(Text)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    # 阶段 4 在此加：embedding: Mapped[list[float]] = mapped_column(Vector(1024))

    document: Mapped["Document"] = relationship(back_populates="chunks")
