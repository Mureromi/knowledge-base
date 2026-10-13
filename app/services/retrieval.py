"""检索层。阶段 0 = 关键词匹配；阶段 4 换向量+混合检索，签名保持不变。

对应 §7.2。
"""

from dataclasses import dataclass

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.chunk import Chunk
from app.models.document import Document


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    document_title: str
    seq: int
    content: str
    score: float = 0.0


def _keywords(question: str) -> list[str]:
    """极简分词：英文按空格，中文按 2-gram。够用，阶段 4 会被向量检索取代。"""
    tokens: list[str] = [w.strip() for w in question.split() if w.strip()]
    chinese = "".join(ch for ch in question if "\u4e00" <= ch <= "\u9fff")
    tokens += [chinese[i:i + 2] for i in range(len(chinese) - 1)]
    return list({t for t in tokens if len(t) >= 2})[:20]


async def search(
    db: AsyncSession, *, owner_id: int, question: str, top_k: int | None = None
) -> list[RetrievedChunk]:
    top_k = top_k or settings.retrieval_top_k
    tokens = _keywords(question)
    if not tokens:
        return []

    conditions = [Chunk.content.ilike(f"%{t}%") for t in tokens]
    stmt = (
        select(Chunk, Document.title)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.owner_id == owner_id)      # ★ 权限过滤：绝不返回别人的文档
        # ★ 阶段 3 §4.3：只检索 status='ready' 的文档。
        #   pending/processing 天然无切片，但 failed 文档可能残留"先写一半再失败"的切片，
        #   不显式过滤会让半成品内容出现在回答里。
        .where(Document.status == "ready")
        .where(or_(*conditions))
        .limit(top_k * 4)                       # 多取一些，再按命中数重排
    )
    rows = (await db.execute(stmt)).all()

    results = [
        RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_title=title,
            seq=chunk.seq,
            content=chunk.content,
            score=float(sum(1 for t in tokens if t in chunk.content)),
        )
        for chunk, title in rows
    ]
    results.sort(key=lambda r: (-r.score, r.chunk_id))
    return results[:top_k]
