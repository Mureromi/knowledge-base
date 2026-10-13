"""向量召回。docs/项目实现文档-阶段4.md §9.2 test_vector_recall。

★ SQLite 上 `<#>` 无法执行，retrieval._vector_candidates 会走应用层余弦分支（§9.1），
  所以这里能用假 embedder + 手工向量精确控制召回顺序，无需真实 pgvector。
"""

import pytest
from app.services.retrieval import _vector_candidates
from app.tests.conftest import FakeEmbedder


@pytest.mark.asyncio
async def test_vector_recall_orders_by_similarity(db_session, seed_document):
    """假 embedder 返回可控向量，Top-N 顺序符合预期。"""
    _doc, chunks = await seed_document(
        chunks=[
            ("A", [1.0, 0.0, 0.0]),   # 与查询同向 → 余弦 1.0
            ("B", [0.0, 1.0, 0.0]),   # 正交 → 0.0
            ("C", [-1.0, 0.0, 0.0]),  # 反向 → -1.0
        ]
    )
    embedder = FakeEmbedder(vectors={"退款": [1.0, 0.0, 0.0]}, dim=3)

    pairs = await _vector_candidates(
        db_session, embedder, owner_id=1, query="退款", limit=10
    )
    ids = [cid for cid, _ in pairs]
    assert ids == [chunks[0].id, chunks[1].id, chunks[2].id], "按余弦降序 A>B>C"
    assert embedder.query_calls == ["退款"]


@pytest.mark.asyncio
async def test_vector_recall_skips_null_embeddings(db_session, seed_document):
    """只召回非空 embedding：回填中途（部分 NULL）不报错、不返回 NULL 行。"""
    _doc, chunks = await seed_document(
        chunks=[
            ("有向量", [1.0, 0.0, 0.0]),
            ("待回填", None),
        ]
    )
    embedder = FakeEmbedder(vectors={"q": [1.0, 0.0, 0.0]}, dim=3)
    pairs = await _vector_candidates(
        db_session, embedder, owner_id=1, query="q", limit=10
    )
    assert [cid for cid, _ in pairs] == [chunks[0].id]


@pytest.mark.asyncio
async def test_vector_recall_only_ready_documents(db_session, seed_document):
    """只召回 status=ready：failed 文档的切片不出现（阶段 3 约定在向量路径同样生效）。"""
    _ready_doc, ready_chunks = await seed_document(
        status="ready", chunks=[("就绪内容", [1.0, 0.0, 0.0])]
    )
    _failed_doc, failed_chunks = await seed_document(
        status="failed", chunks=[("失败残留", [1.0, 0.0, 0.0])]
    )
    embedder = FakeEmbedder(vectors={"q": [1.0, 0.0, 0.0]}, dim=3)
    pairs = await _vector_candidates(
        db_session, embedder, owner_id=1, query="q", limit=10
    )
    ids = {cid for cid, _ in pairs}
    assert ready_chunks[0].id in ids
    assert failed_chunks[0].id not in ids, "failed 文档不得被向量召回"


@pytest.mark.asyncio
async def test_vector_recall_respects_owner_isolation(db_session, seed_document):
    """权限过滤：绝不召回别人的切片。"""
    _mine, mine = await seed_document(
        owner_id=1, chunks=[("我的", [1.0, 0.0, 0.0])]
    )
    _other, other = await seed_document(
        owner_id=2, chunks=[("别人的", [1.0, 0.0, 0.0])]
    )
    embedder = FakeEmbedder(vectors={"q": [1.0, 0.0, 0.0]}, dim=3)
    pairs = await _vector_candidates(
        db_session, embedder, owner_id=1, query="q", limit=10
    )
    ids = {cid for cid, _ in pairs}
    assert mine[0].id in ids
    assert other[0].id not in ids
