"""回填幂等性。docs/项目实现文档-阶段4.md §7.1 / §9.2 test_backfill。

★ scripts/backfill_embeddings.py 直接用真实 fastembed 模型，测试环境不加载模型；
  这里用与脚本**完全相同的查询模式**（`WHERE embedding IS NULL` + 每批一事务）复刻一遍，
  用假 embedder 提供向量，验证的正是脚本赖以"重跑安全"的那条幂等不变量。
"""

import pytest
from sqlalchemy import func, select, update

from app.models.chunk import Chunk
from app.tests.conftest import FakeEmbedder


async def _backfill_pass(db, embedder, batch_size: int = 2) -> int:
    """复刻脚本主循环：只挑 embedding IS NULL 的行，编码后写回，每批一事务。返回本次处理行数。"""
    processed = 0
    while True:
        stmt = (
            select(Chunk.id, Chunk.content)
            .where(Chunk.embedding.is_(None))
            .limit(batch_size)
        )
        rows = (await db.execute(stmt)).all()
        if not rows:
            break
        ids = [int(r[0]) for r in rows]
        texts = [r[1] for r in rows]
        vectors = await embedder.embed_documents(texts)
        for cid, vec in zip(ids, vectors, strict=True):
            await db.execute(update(Chunk).where(Chunk.id == cid).values(embedding=vec))
        await db.commit()
        processed += len(ids)
    return processed


async def _pending(db) -> int:
    return int(
        await db.scalar(select(func.count(Chunk.id)).where(Chunk.embedding.is_(None))) or 0
    )


@pytest.mark.asyncio
async def test_backfill_is_idempotent(db_session, seed_document):
    """幂等：跑两次结果一致，第二次处理 0 行。"""
    await seed_document(
        chunks=[("内容一", None), ("内容二", None), ("内容三", None)]
    )
    assert await _pending(db_session) == 3

    first = await _backfill_pass(db_session, FakeEmbedder())
    assert first == 3
    assert await _pending(db_session) == 0

    second = await _backfill_pass(db_session, FakeEmbedder())
    assert second == 0, "重跑不重算已成功的行"


@pytest.mark.asyncio
async def test_backfill_only_touches_null_rows(db_session, seed_document):
    """部分失败续跑：已成功的行不被重算，第二次只补剩余的。"""
    _doc, chunks = await seed_document(
        chunks=[("已完成", None), ("待补一", None), ("待补二", None)]
    )
    # 模拟上一批成功后中断：手工把第一条标为已回填
    await db_session.execute(
        update(Chunk).where(Chunk.id == chunks[0].id).values(embedding=[0.5, 0.5])
    )
    await db_session.commit()
    assert await _pending(db_session) == 2

    embedder = FakeEmbedder()
    processed = await _backfill_pass(db_session, embedder)
    assert processed == 2
    # 编码只应发生在两条 NULL 行上，绝不重算已完成那条
    encoded_texts = [t for batch in embedder.doc_calls for t in batch]
    assert "已完成" not in encoded_texts
    assert set(encoded_texts) == {"待补一", "待补二"}
    assert await _pending(db_session) == 0
