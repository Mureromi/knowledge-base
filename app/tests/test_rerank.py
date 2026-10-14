"""rerank 精排。docs/项目实现文档-阶段4.md §9.2 test_rerank。

种子切片都无向量（embedding=None）→ 向量路为空，退化为"关键词单路 + RRF"，
候选顺序确定，便于断言 rerank 是否改变了顺序、是否降级。
"""

import pytest

from app.core.config import settings
from app.services.retrieval import search
from app.tests.conftest import FakeEmbedder, FakeReranker


@pytest.mark.asyncio
async def test_rerank_reorders_by_scores(db_session, seed_document, monkeypatch):
    """精排改变顺序：假 reranker 返回指定分数 → 顺序按分数排。

    ★ 显式打开 rerank：默认值是 False（§15.6 实测决策），依赖默认值会让测试变脆。
    """
    monkeypatch.setattr(settings, "rerank_enabled", True)
    _doc, chunks = await seed_document(
        chunks=[("退款一条", None), ("退款二条", None), ("退款三条", None)]
    )
    q = "退款"
    reranker = FakeReranker(
        scores={
            (q, "退款一条"): 1.0,
            (q, "退款二条"): 3.0,
            (q, "退款三条"): 5.0,
        }
    )
    results = await search(
        db_session, owner_id=1, question=q,
        embedder=FakeEmbedder(), reranker=reranker,
    )
    assert [r.chunk_id for r in results] == [
        chunks[2].id, chunks[1].id, chunks[0].id
    ], "分数 5>3>1 → 三、二、一"


@pytest.mark.asyncio
async def test_rerank_disabled_skips_reranker(db_session, seed_document, monkeypatch):
    """RERANK_ENABLED=false：跳过精排，顺序 = RRF 顺序，且根本不调用 reranker。"""
    monkeypatch.setattr(settings, "rerank_enabled", False)
    _doc, chunks = await seed_document(
        chunks=[("退款一条", None), ("退款二条", None)]
    )
    reranker = FakeReranker()
    results = await search(
        db_session, owner_id=1, question="退款",
        embedder=FakeEmbedder(), reranker=reranker,
    )
    assert reranker.calls == [], "关闭后不应调用 reranker"
    assert [r.chunk_id for r in results] == [chunks[0].id, chunks[1].id]


@pytest.mark.asyncio
async def test_rerank_skipped_when_too_few_candidates(
    db_session, seed_document, monkeypatch
):
    """候选少于 rerank_min_candidates：跳过精排（不白跑模型）。

    ★ 必须显式打开 rerank：默认值是 False（§15.6 实测决策）。
      否则这条会**因为"未启用"而通过**——断言 reranker.calls == [] 照样成立，
      但它已经不再验证"候选不足时跳过"这个行为了（假通过）。
    """
    monkeypatch.setattr(settings, "rerank_enabled", True)
    _doc, chunks = await seed_document(chunks=[("退款唯一", None)])
    reranker = FakeReranker()
    results = await search(
        db_session, owner_id=1, question="退款",
        embedder=FakeEmbedder(), reranker=reranker,
    )
    assert reranker.calls == [], "候选不足 rerank_min_candidates，跳过精排"
    assert [r.chunk_id for r in results] == [chunks[0].id]


@pytest.mark.asyncio
async def test_rerank_failure_degrades_to_rrf_order(
    db_session, seed_document, monkeypatch
):
    """★ rerank 抛异常 → 降级为 RRF 顺序，问答不受影响（§9.2，文档伪代码缺的这层保护）。

    ★ 必须显式打开 rerank：默认值是 False（§15.6 实测决策）。
      否则**降级分支根本不会被触发**——这条保护层就没人看着了。
    """
    monkeypatch.setattr(settings, "rerank_enabled", True)
    _doc, chunks = await seed_document(
        chunks=[("退款一条", None), ("退款二条", None), ("退款三条", None)]
    )
    reranker = FakeReranker(raises=True)
    results = await search(
        db_session, owner_id=1, question="退款",
        embedder=FakeEmbedder(), reranker=reranker,
    )
    # 不抛异常，且保持关键词单路的 RRF 顺序（chunk_id 升序）
    assert [r.chunk_id for r in results] == [
        chunks[0].id, chunks[1].id, chunks[2].id
    ]
