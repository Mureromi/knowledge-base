from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
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
    # 阶段 4：pgvector 向量列，可空（"加列→回填"两步分开做，无向量仍可被关键词召回）。
    # ★ 维度必须与 embedding 模型一致（bge-small-zh-v1.5 = 512），
    #   写错不会在写入时报错，只在查询时报 "different vector dimensions"，难排查。
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding_dim), default=None
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")
