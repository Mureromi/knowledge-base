"""Document 与 Chunk 的数据访问。"""

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.chunk import Chunk
from app.models.document import Document
from app.models.processing_task import ProcessingTask
from app.services.ingest import ChunkDraft


async def create_with_chunks(
    db: AsyncSession,
    *,
    owner_id: int,
    title: str,
    source_type: str,
    raw_text: str,
    drafts: list[ChunkDraft],
    raw_content: bytes | None = None,
    embeddings: list[list[float]] | None = None,
) -> tuple[Document, int]:
    """一次事务写入文档 + 切片（txt/md 快路径用）。返回 (document, chunk_count)。

    embeddings 与 drafts 对齐（阶段 4）：传入则新切片入库即带向量，为 None / 长度不够时该行留空。
    """
    doc = Document(
        owner_id=owner_id,
        title=title,
        source_type=source_type,
        raw_text=raw_text,
        raw_content=raw_content,
        status="ready",
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)

    chunks = [
        Chunk(
            document_id=doc.id,
            owner_id=owner_id,
            seq=d.seq,
            content=d.content,
            char_count=len(d.content),
            embedding=(
                embeddings[i] if embeddings is not None and i < len(embeddings) else None
            ),
        )
        for i, d in enumerate(drafts)
    ]
    for c in chunks:
        db.add(c)
    await db.flush()
    return doc, len(chunks)


async def create_pending(
    db: AsyncSession,
    *,
    owner_id: int,
    title: str,
    source_type: str,
    raw_content: bytes,
) -> Document:
    """pdf/docx 异步路径：先建 status=pending 的文档，全文与切片由 worker 回填。

    ★ raw_content 必须落库：worker 是另一个进程，只能从数据库拿到上传的 bytes。
    只 flush 不 commit，提交时机交给端点控制（必须在 enqueue 之前 commit，§5.3）。
    """
    doc = Document(
        owner_id=owner_id,
        title=title,
        source_type=source_type,
        raw_text=None,
        raw_content=raw_content,
        status="pending",
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)
    return doc


async def get(db: AsyncSession, doc_id: int) -> Document | None:
    """worker 侧按 id 取文档（需要 raw_content / source_type 做解析）。"""
    return await db.get(Document, doc_id)


async def set_status(db: AsyncSession, doc_id: int, status: str) -> None:
    """更新 documents.status（当前状态投影）。不 flush，由调用方统一提交。"""
    await db.execute(
        update(Document).where(Document.id == doc_id).values(status=status)
    )


async def replace_chunks(
    db: AsyncSession,
    *,
    document_id: int,
    owner_id: int,
    raw_text: str,
    drafts: list[ChunkDraft],
    embeddings: list[list[float]] | None = None,
) -> int:
    """先删后写切片（§6.3 ③）。重试场景上次可能写了半截，必须清理避免重复切片。

    同时回填 raw_text。返回新写入的切片数。
    embeddings 与 drafts 对齐（阶段 4）：worker 入库时顺带写向量，缺则留空。
    """
    await db.execute(delete(Chunk).where(Chunk.document_id == document_id))
    await db.execute(
        update(Document).where(Document.id == document_id).values(raw_text=raw_text)
    )
    chunks = [
        Chunk(
            document_id=document_id,
            owner_id=owner_id,
            seq=d.seq,
            content=d.content,
            char_count=len(d.content),
            embedding=(
                embeddings[i] if embeddings is not None and i < len(embeddings) else None
            ),
        )
        for i, d in enumerate(drafts)
    ]
    db.add_all(chunks)
    await db.flush()
    return len(chunks)


async def get_with_chunk_count(db: AsyncSession, doc_id: int) -> tuple[Document, int] | None:
    """按 id 取文档 + 切片数；找不到返回 None（越权由调用方过滤 owner_id）。"""
    stmt = (
        select(Document, func.count(Chunk.id))
        .outerjoin(Chunk, Chunk.document_id == Document.id)
        .where(Document.id == doc_id)
        .group_by(Document.id)
    )
    row = (await db.execute(stmt)).one_or_none()
    if row is None:
        return None
    doc, cnt = row
    return doc, int(cnt or 0)


async def list_by_owner(
    db: AsyncSession,
    *,
    owner_id: int,
    page: int = 1,
    page_size: int = 20,
    title_like: str | None = None,
) -> tuple[list[Document], int]:
    """分页 + 标题模糊。返回 (items, total)。"""
    conditions = [Document.owner_id == owner_id]
    if title_like:
        conditions.append(Document.title.ilike(f"%{title_like}%"))

    count_stmt = select(func.count(Document.id)).where(*conditions)
    total = int((await db.execute(count_stmt)).scalar_one() or 0)

    stmt = (
        select(Document)
        .where(*conditions)
        .order_by(Document.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .options(selectinload(Document.chunks))
    )
    items = list((await db.execute(stmt)).scalars().all())
    return items, total


async def delete_document(db: AsyncSession, doc: Document) -> int:
    """删除文档：切片级联删除，**任务历史保留**（§4.2 + 验收修复）。

    两种子记录的处理方式不同，是刻意的：
      - `chunks`：随文档删除（内容没了，切片留着毫无意义 → cascade delete-orphan）
      - `processing_tasks`：**保留**，只把 document_id 置 NULL
        （任务流水是审计数据，用于统计失败率与排查；文档删了历史不该消失）
    """
    stmt = select(func.count(Chunk.id)).where(Chunk.document_id == doc.id)
    cnt = int((await db.execute(stmt)).scalar_one() or 0)

    # 显式置空，不依赖数据库的 ON DELETE SET NULL：
    #   ORM 已经加载过这些对象时，直接删父行会让内存里的对象与库不一致（stale），
    #   显式 UPDATE 能保证无论走哪条路径，行为都确定。
    await db.execute(
        update(ProcessingTask)
        .where(ProcessingTask.document_id == doc.id)
        .values(document_id=None)
    )
    await db.delete(doc)
    await db.flush()
    return cnt
