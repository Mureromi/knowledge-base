"""多轮查询改写。docs/项目实现文档-阶段4.md §4 / §9.2 test_query_rewrite。

直接调用 rewrite_query（不经端点），用可控假 LLM + 真缓存夹具逐例验证。
"""

import pytest

from app.core.config import settings
from app.services.query_rewrite import rewrite_query
from app.tests.conftest import FakeLLM


class _RaisingLLM:
    async def chat(self, messages, **kwargs):
        raise RuntimeError("LLM 挂了")


@pytest.mark.asyncio
async def test_first_turn_skips_llm(cache_backend, monkeypatch):
    """首轮（无历史）：不调用 LLM，直接返回原问题。"""
    monkeypatch.setattr(settings, "query_rewrite_enabled", True)
    llm = FakeLLM()
    out = await rewrite_query(
        None, llm, cache_backend,
        owner_id=1, session_id=1, question="退款要多久到账？", window=[],
    )
    assert out == "退款要多久到账？"
    assert llm.calls == [], "首轮绝不应调用 LLM"


@pytest.mark.asyncio
async def test_rewrite_returns_rephrased_and_keeps_original_question(cache_backend, monkeypatch):
    """多轮追问：返回改写句；断言发给 LLM 的最后一条仍是原始问题（§4.3 的坑）。"""
    monkeypatch.setattr(settings, "query_rewrite_enabled", True)
    llm = FakeLLM(content="退货的条件和流程是什么")
    window = [
        {"role": "user", "content": "退款要多久到账？"},
        {"role": "assistant", "content": "审核通过后 5-7 个工作日。"},
    ]
    out = await rewrite_query(
        None, llm, cache_backend,
        owner_id=1, session_id=1, question="那退货呢？", window=window,
    )
    assert out == "退货的条件和流程是什么"
    sent = llm.calls[0]["messages"]
    assert sent[-1] == {"role": "user", "content": "那退货呢？"}, "喂给改写器的原句必须是用户原话"


@pytest.mark.asyncio
async def test_rewrite_result_is_cached(cache_backend, monkeypatch):
    """改写结果被缓存：同一窗口问两次只调一次 LLM。"""
    monkeypatch.setattr(settings, "query_rewrite_enabled", True)
    llm = FakeLLM(content="退货流程是什么")
    window = [{"role": "user", "content": "退款要多久"}, {"role": "assistant", "content": "5-7 天"}]
    args = dict(owner_id=1, session_id=1, question="那退货呢？", window=window)

    first = await rewrite_query(None, llm, cache_backend, **args)
    second = await rewrite_query(None, llm, cache_backend, **args)
    assert first == second == "退货流程是什么"
    assert len(llm.calls) == 1, "命中缓存后不再调 LLM"


@pytest.mark.asyncio
async def test_different_window_different_cache_key(cache_backend, monkeypatch):
    """不同窗口同问题：缓存键不同（恢复不同上下文），各自都调 LLM。"""
    monkeypatch.setattr(settings, "query_rewrite_enabled", True)
    llm = FakeLLM(content="改写结果")
    w1 = [{"role": "user", "content": "退款要多久"}]
    w2 = [{"role": "user", "content": "保修多久"}]
    await rewrite_query(None, llm, cache_backend, owner_id=1, session_id=1, question="那它呢？", window=w1)
    await rewrite_query(None, llm, cache_backend, owner_id=1, session_id=1, question="那它呢？", window=w2)
    assert len(llm.calls) == 2, "窗口不同 → 指纹不同 → 不共用缓存"


@pytest.mark.asyncio
async def test_rewrite_failure_degrades_to_original(cache_backend, monkeypatch):
    """★ 改写失败：退化为原问题，不抛异常。"""
    monkeypatch.setattr(settings, "query_rewrite_enabled", True)
    out = await rewrite_query(
        None, _RaisingLLM(), cache_backend,
        owner_id=1, session_id=1, question="那退货呢？",
        window=[{"role": "user", "content": "退款要多久"}],
    )
    assert out == "那退货呢？"


@pytest.mark.asyncio
async def test_disabled_returns_original_without_llm(cache_backend, monkeypatch):
    """query_rewrite_enabled=false（默认，测试隔离）：原样返回、不调 LLM。"""
    monkeypatch.setattr(settings, "query_rewrite_enabled", False)
    llm = FakeLLM()
    out = await rewrite_query(
        None, llm, cache_backend,
        owner_id=1, session_id=1, question="那退货呢？",
        window=[{"role": "user", "content": "退款要多久"}],
    )
    assert out == "那退货呢？"
    assert llm.calls == []
