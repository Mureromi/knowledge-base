"""端到端答案缓存测试（阶段 1 §10.2）。

覆盖 §11 验收清单：
- 同一问题连问 2 次：第 2 次 fake_llm 不再被调用，cache_hit=true
- 上传相关新文档后再问 → 缓存未命中（答案随资料更新）
- 删除被引用文档后再问 → 不返回失效引用
- CACHE_ENABLED=false → 每次都调 LLM
"""

import pytest
from sqlalchemy import select

from app.models.chat_log import ChatLog
from app.tests.conftest import FakeLLM


async def _upload(client, headers, filename: str, content: str) -> int:
    r = await client.post(
        "/api/v1/documents",
        headers=headers,
        files={"file": (filename, content.encode("utf-8"), "text/plain")},
    )
    # 阶段 3 §5.1：上传改为 202 Accepted（异步）。txt/md 快路径历史上是 201，
    # 两者都在语义上表示"文档已创建"，故都接受；真正的错误码仍会被这条断言挡下。
    assert r.status_code in (201, 202), r.text
    return r.json()["id"]


async def _ask(client, headers, question: str):
    # 阶段 2：/chat 已改 SSE，业务逻辑断言走非流式端点（§8.1 的测试分工）
    r = await client.post(
        "/api/v1/chat/sync", headers=headers, json={"question": question}
    )
    assert r.status_code == 200, r.text
    return r.json()


def _citation_keys(citations: list[dict]) -> list[tuple]:
    """只取能定位原文的字段。缓存命中时 _hydrate_citations 不重算 score（恒为 0.0），
    直接比字典会假阳性，需放宽到主身份字段。"""
    return [
        (c["chunk_id"], c["document_id"], c["document_title"], c["seq"])
        for c in citations
    ]


@pytest.mark.asyncio
async def test_repeated_question_hits_cache_and_skips_llm(
    client, auth_headers, fake_llm: FakeLLM, session_factory
):
    """★ 阶段 1 的核心收益：相同问题第二次不再调 LLM。"""
    await _upload(client, auth_headers, "policy.md", "退款流程：先提交申请，再等待审核")

    first = await _ask(client, auth_headers, "退款流程")
    assert first["usage"]["cache_hit"] is False
    assert fake_llm.calls and len(fake_llm.calls) == 1
    llm_cost = first["usage"]["cost_yuan"]

    second = await _ask(client, auth_headers, "退款流程")
    assert second["usage"]["cache_hit"] is True, "第二次应命中缓存"
    assert len(fake_llm.calls) == 1, "命中缓存时绝不能再调 LLM"
    assert second["usage"]["cost_yuan"] == 0.0
    assert second["usage"]["saved_cost_yuan"] == llm_cost
    assert second["answer"] == first["answer"]
    assert _citation_keys(second["citations"]) == _citation_keys(first["citations"]), (
        "命中缓存后引用应指向同一 chunk（score 因不重新检索而归 0，不比）"
    )

    # chat_logs 应落两条：第一条 cache_hit=False，第二条 cache_hit=True
    async with session_factory() as s:
        rows = (await s.execute(select(ChatLog).order_by(ChatLog.id))).scalars().all()
        hits = [r.cache_hit for r in rows]
        assert hits == [False, True]
        # 命中的那条真实成本为 0，省下的成本另存
        hit_log = rows[-1]
        assert hit_log.cost_yuan == 0.0
        assert hit_log.saved_cost_yuan == llm_cost


@pytest.mark.asyncio
async def test_normalization_lets_punctuation_versions_share_cache(
    client, auth_headers, fake_llm: FakeLLM
):
    """"退款流程" 与 "退款流程。" 归一化后应命中同一条缓存。"""
    await _upload(client, auth_headers, "policy.md", "退款流程：先提交申请")

    await _ask(client, auth_headers, "退款流程")
    second = await _ask(client, auth_headers, "退款流程。")
    assert second["usage"]["cache_hit"] is True
    assert len(fake_llm.calls) == 1


@pytest.mark.asyncio
async def test_uploading_relevant_doc_invalidates_cache(
    client, auth_headers, fake_llm: FakeLLM
):
    """上传进入 Top-K 的新文档 → 检索结果变 → 指纹变 → 缓存未命中。"""
    await _upload(client, auth_headers, "a.md", "退款流程：提交申请")
    await _ask(client, auth_headers, "退款流程")
    assert len(fake_llm.calls) == 1

    # 再上传一份"退款"更相关的文档，让它进 Top-K
    await _upload(
        client, auth_headers, "b.md", "退款补充：审核 3 日到账（退款流程详见此）"
    )
    third = await _ask(client, auth_headers, "退款流程")
    # 检索结果变了 → 键变 → 缓存未命中 → 又调了一次 LLM
    assert third["usage"]["cache_hit"] is False, (
        "新文档进入 Top-K 后应视为未命中（不能返回旧答案）"
    )
    assert len(fake_llm.calls) == 2


@pytest.mark.asyncio
async def test_deleting_cited_doc_makes_cache_stale_and_refetches(
    client, auth_headers, fake_llm: FakeLLM
):
    """引用被删的文档不能出现在 citations 里；此时应视为未命中重新生成。"""
    doc_id = await _upload(client, auth_headers, "temp.md", "退款流程：先提交申请")
    await _ask(client, auth_headers, "退款流程")
    assert len(fake_llm.calls) == 1

    # 直接删掉这个文档（chunk 会级联删）
    r = await client.delete(f"/api/v1/documents/{doc_id}", headers=auth_headers)
    assert r.status_code == 200

    again = await _ask(client, auth_headers, "退款流程")
    # 检索已无该 chunk → 走"无相关资料"分支（不调 LLM 也不命中缓存）
    assert again["citations"] == []
    assert "没有找到" in again["answer"] or "无法回答" in again["answer"]


@pytest.mark.asyncio
async def test_cache_hit_does_not_consume_daily_quota(
    client, auth_headers, fake_llm: FakeLLM, cache_backend
):
    """§10.2 第三条防回归：命中缓存不吃 L3 配额，否则缓存白做。"""
    from app.core.config import settings

    await _upload(client, auth_headers, "quota.md", "退款流程：提交申请")

    # 问 3 次，第 1 次调 LLM 扣 1 次配额，第 2、3 次命中缓存不扣
    for _ in range(3):
        await _ask(client, auth_headers, "退款流程")

    from app.services.ratelimit import _extract_redis

    redis = _extract_redis(cache_backend)
    assert redis is not None
    # 找当天配额键
    from datetime import datetime

    from app.clients.cache import CST

    today = datetime.now(CST).strftime("%Y%m%d")
    key = f"{settings.cache_key_prefix}:v1:quota:user:{1}:{today}"
    # 拿当前登录用户的 id：从 auth_headers 反查
    # 简单起见扫所有 quota 键
    async for k in redis.scan_iter(match=f"{settings.cache_key_prefix}:v1:quota:user:*"):
        used = int(await redis.get(k))
        assert used == 1, (
            f"3 次问答只有 1 次真正调 LLM，配额只应扣 1，实际 {used}（键 {k}）"
        )


@pytest.mark.asyncio
async def test_cache_disabled_calls_llm_every_time(
    client, auth_headers, fake_llm: FakeLLM, monkeypatch
):
    """CACHE_ENABLED=false 时每次都调 LLM（用于对比测试）。"""
    from app.core.config import settings
    from app.services import qa as qa_mod

    monkeypatch.setattr(settings, "cache_enabled", False)
    monkeypatch.setattr(qa_mod.settings, "cache_enabled", False, raising=False)

    await _upload(client, auth_headers, "policy.md", "退款流程：先提交申请")
    await _ask(client, auth_headers, "退款流程")
    second = await _ask(client, auth_headers, "退款流程")

    assert second["usage"]["cache_hit"] is False
    assert len(fake_llm.calls) == 2
