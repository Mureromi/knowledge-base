"""检索层测试（§9）：越权是最严重的 bug。"""

import pytest
from httpx import AsyncClient

from app.crud import document as doc_crud
from app.tests.conftest import register_and_login


async def _upload(client: AsyncClient, headers: dict, filename: str, content: str) -> int:
    r = await client.post(
        "/api/v1/documents",
        headers=headers,
        files={"file": (filename, content.encode("utf-8"), "text/plain")},
    )
    # 阶段 3 §5.1：上传改为 202 Accepted（异步）。txt/md 快路径历史上是 201，
    # 两者都在语义上表示"文档已创建"，故都接受；真正的错误码仍会被这条断言挡下。
    assert r.status_code in (201, 202), r.text
    return r.json()["id"]


@pytest.mark.asyncio
async def test_search_is_isolated_between_users(client: AsyncClient):
    """A 上传“量子隧穿”，B 上传“红烧肉做法”；B 问“量子隧穿”→ 应命中"无相关资料"。"""
    token_a = await register_and_login(client, "alice_a", "secret123")
    token_b = await register_and_login(client, "bob_b", "secret123")
    ha = {"Authorization": f"Bearer {token_a}"}
    hb = {"Authorization": f"Bearer {token_b}"}

    await _upload(client, ha, "physics.md", "量子隧穿是粒子穿越势垒的现象")
    await _upload(client, hb, "recipe.md", "红烧肉需要焯水、上色、慢炖")

    r = await client.post(
        "/api/v1/chat/sync",
        headers=hb,
        json={"question": "什么是量子隧穿？", "top_k": 5},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # B 检索不到 A 的文档 → 走"无相关资料"分支
    assert body["citations"] == []
    assert "没有找到" in body["answer"] or "无法回答" in body["answer"]


@pytest.mark.asyncio
async def test_keyword_hit_returns_citations(client: AsyncClient):
    token_a = await register_and_login(client, "carl_a", "secret123")
    ha = {"Authorization": f"Bearer {token_a}"}
    doc_id = await _upload(client, ha, "policy.md", "退款流程：先提交申请，再等待审核")

    r = await client.post(
        "/api/v1/chat/sync",
        headers=ha,
        json={"question": "退款流程", "top_k": 3},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["citations"], "至少应命中一条引用"
    assert {c["document_id"] for c in body["citations"]} == {doc_id}


@pytest.mark.asyncio
async def test_failed_document_chunks_never_retrieved(
    client: AsyncClient, auth_headers, session_factory
):
    """★ 阶段 3 §4.3 回归：status=failed 的文档，其切片永不出现在检索结果。

    模拟"先写了一半切片再失败"的残留场景：文档有匹配切片，但状态是 failed，
    检索必须显式过滤掉它。
    """
    doc_id = await _upload(client, auth_headers, "refund.md", "退款流程：先提交申请，再等待审核")
    # 把已成功入库的文档状态改为 failed（残留切片仍在 chunks 表）
    async with session_factory() as db:
        await doc_crud.set_status(db, doc_id, "failed")
        await db.commit()

    r = await client.post(
        "/api/v1/chat/sync",
        headers=auth_headers,
        json={"question": "退款流程", "top_k": 5},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # failed 文档被过滤 → 检索不到 → 走"无相关资料"分支
    assert body["citations"] == []
    assert doc_id not in {c["document_id"] for c in body["citations"]}
