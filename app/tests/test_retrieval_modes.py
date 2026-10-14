"""检索模式（retrieval_mode 契约）与渐进降级。docs/项目实现文档-阶段4.md §8 / §9.2。"""

import pytest

from app.core.config import settings
from app.services.retrieval import resolve_retrieval_mode, search
from app.tests.conftest import FakeEmbedder, FakeReranker


class _RaisingEmbedder:
    """模拟向量路整体报错（模型未加载 / 编码失败）。"""

    is_ready = True

    async def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("向量模型挂了")

    async def embed_documents(self, texts):
        raise RuntimeError("向量模型挂了")


def test_resolve_mode_with_rerank(monkeypatch):
    """★ 显式打开 rerank：默认值是 False（§15.6 实测决策），依赖默认值会让测试变脆。"""
    monkeypatch.setattr(settings, "rerank_enabled", True)
    assert resolve_retrieval_mode(FakeEmbedder(), FakeReranker()) == "hybrid+rerank"


def test_resolve_mode_without_rerank(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", False)
    assert resolve_retrieval_mode(FakeEmbedder(), FakeReranker()) == "hybrid"


def test_resolve_mode_without_embedder():
    """没注入 embedder（或关掉 embedding）→ 退回 keyword。"""
    assert resolve_retrieval_mode(None, FakeReranker()) == "keyword"


@pytest.mark.asyncio
async def test_embedding_disabled_skips_vector_and_is_keyword(
    db_session, seed_document, monkeypatch
):
    """★ 关掉 embedding：不做向量召回（query 根本不编码），行为退回纯关键词（渐进降级）。"""
    monkeypatch.setattr(settings, "embedding_enabled", False)
    await seed_document(chunks=[("退款政策说明", [1.0, 0.0, 0.0])])
    embedder = FakeEmbedder()

    results = await search(
        db_session, owner_id=1, question="退款",
        embedder=embedder, reranker=None,
    )
    assert embedder.query_calls == [], "embedding 关闭 → 不编码 query"
    assert len(results) == 1, "仍靠关键词召回"
    assert resolve_retrieval_mode(embedder, None) == "keyword"


@pytest.mark.asyncio
async def test_endpoint_reports_hybrid_rerank(client, auth_headers, monkeypatch):
    """显式打开 rerank → 接口 retrieval_mode = hybrid+rerank。

    ★ 这里必须显式 monkeypatch：rerank 的**默认值是 False**（依据 §15.6 的自有语料实测，
      Top-1 +6% / Top-5 +0%，不值得常驻 801MB）。原先这条测试依赖"默认开启"，
      改默认值后它会误红——这类"依赖默认配置"的测试是脆弱的，改成显式声明意图。
    """
    monkeypatch.setattr(settings, "rerank_enabled", True)
    r = await client.post(
        "/api/v1/chat/sync", json={"question": "随便问一个没有的问题"}, headers=auth_headers
    )
    assert r.status_code == 200
    body = r.json()
    assert body["retrieval_mode"] == "hybrid+rerank"


@pytest.mark.asyncio
async def test_endpoint_falls_back_to_keyword_when_embedding_off(
    client, auth_headers, monkeypatch
):
    monkeypatch.setattr(settings, "embedding_enabled", False)
    r = await client.post(
        "/api/v1/chat/sync", json={"question": "随便问"}, headers=auth_headers
    )
    assert r.status_code == 200
    assert r.json()["retrieval_mode"] == "keyword"


@pytest.mark.asyncio
async def test_vector_failure_degrades_to_keyword(db_session, seed_document):
    """★ 假 embedder 报错 → 降级为关键词，问答不崩（§10）。"""
    await seed_document(chunks=[("退款政策说明", None)])
    results = await search(
        db_session, owner_id=1, question="退款",
        embedder=_RaisingEmbedder(), reranker=None,
    )
    assert len(results) == 1, "向量路崩了，关键词路仍命中"


@pytest.mark.asyncio
async def test_endpoint_reports_hybrid_by_default(client, auth_headers):
    """★ 默认配置（embedding 开、rerank 关）→ 接口 retrieval_mode = hybrid。

    这条守住 §15.6 的决策：默认不加载 reranker（省 801MB RSS），
    且接口必须**如实报告**实际生效的检索方式，不能谎报 hybrid+rerank。
    """
    assert settings.rerank_enabled is False, (
        "rerank 默认值被改回 True 了？若是有意为之（语料变大后复测通过），"
        "请同时更新本条测试与 docs/项目实现文档-阶段4.md §15.6 的结论。"
    )
    r = await client.post(
        "/api/v1/chat/sync", json={"question": "随便问一个没有的问题"}, headers=auth_headers
    )
    assert r.status_code == 200, r.text
    assert r.json()["retrieval_mode"] == "hybrid"
