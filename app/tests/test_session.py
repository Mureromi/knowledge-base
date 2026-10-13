"""会话与滑动窗口测试（阶段 2 §8.3）。

覆盖：
- 新建会话返回 session_id + title 来自首问
- 访问他人会话 → 404（不是 403，避免探测存在性）
- 聊 8 轮后发给 LLM 的 messages 只含最近 6 轮（12 条 + 本轮 + system）
- 窗口对齐：消息按轮取，不出现"有答无问"
- 会话列表按 last_active_at 倒序 / 重命名 / 删除
"""

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.models.chat_log import ChatLog
from app.tests.conftest import FakeLLM, register_and_login


async def _ask(client: AsyncClient, headers: dict, question: str, session_id=None):
    body = {"question": question}
    if session_id is not None:
        body["session_id"] = session_id
    r = await client.post("/api/v1/chat/sync", headers=headers, json=body)
    assert r.status_code == 200, r.text
    return r.json()


async def _upload(client: AsyncClient, headers: dict, content: str) -> int:
    r = await client.post(
        "/api/v1/documents",
        headers=headers,
        files={"file": ("policy.md", content.encode("utf-8"), "text/plain")},
    )
    # 阶段 3 §5.1：上传改为 202 Accepted（异步）。txt/md 快路径历史上是 201，
    # 两者都在语义上表示"文档已创建"，故都接受；真正的错误码仍会被这条断言挡下。
    assert r.status_code in (201, 202), r.text
    return r.json()["id"]


@pytest.mark.asyncio
async def test_first_question_creates_session_with_title(
    client, auth_headers, fake_llm: FakeLLM
):
    await _upload(client, auth_headers, "退款流程：先提交申请，再等待审核")
    first = await _ask(client, auth_headers, "退款流程")
    assert first["session_id"] > 0

    # 会话详情里 title 应为首轮问题（§2.1：不调 LLM 生成标题，省钱）
    r = await client.get(
        f"/api/v1/chat/sessions/{first['session_id']}", headers=auth_headers
    )
    assert r.status_code == 200
    assert r.json()["session"]["title"] == "退款流程"


@pytest.mark.asyncio
async def test_accessing_others_session_returns_404(client, auth_headers):
    """★ 越权统一 404 而不是 403——不暴露"这个会话 id 存在"（§3.1）。"""
    first = await _ask(client, auth_headers, "你好")
    sid = first["session_id"]

    token_b = await register_and_login(client, "bob_iso", "secret123")
    hb = {"Authorization": f"Bearer {token_b}"}
    r = await client.post(
        "/api/v1/chat/sync", headers=hb, json={"question": "偷看", "session_id": sid}
    )
    assert r.status_code == 404, r.text
    assert r.json()["detail"]["code"] == 40402

    # GET / PATCH / DELETE 同样 404
    for method in ("get", "patch", "delete"):
        rr = await getattr(client, method)(
            f"/api/v1/chat/sessions/{sid}",
            headers=hb,
            **({"json": {"title": "劫持"}} if method == "patch" else {}),
        )
        assert rr.status_code == 404, method


@pytest.mark.asyncio
async def test_sliding_window_keeps_only_recent_turns(
    client, auth_headers, fake_llm: FakeLLM
):
    """聊 8 轮后，发给 LLM 的只有：system + 最近 6 轮（12 条）+ 本轮问题。"""
    # 内容含"问题"二字，保证 "问题N" 的 2-gram 能命中检索
    await _upload(client, auth_headers, "库存退款发货保修常见问题手册")
    sid = None
    for i in range(1, 9):
        out = await _ask(client, auth_headers, f"问题{i}", session_id=sid)
        sid = out["session_id"]

    assert len(fake_llm.calls) == 8, "8 轮都应真正调 LLM（每轮历史不同 → 缓存未命中）"
    msgs = fake_llm.calls[-1]["messages"]
    assert msgs[0]["role"] == "system"
    # system + 12（最近 6 轮）+ 本轮 user = 14
    assert len(msgs) == 14, f"窗口应截到最近 6 轮，实际 {len(msgs)} 条"
    # 第 1 轮（窗口外）不应出现在历史里；第 7 轮必须在
    contents = " ".join(m["content"] for m in msgs)
    assert "问题1" not in contents
    assert "问题7" in contents

    # 历史部分角色从 user 开始两两成对（§8.3"窗口对齐"：不出现有答无问）
    history = msgs[1:-1]
    assert len(history) % 2 == 0
    for j in range(0, len(history), 2):
        assert history[j]["role"] == "user"
        assert history[j + 1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_followup_request_carries_previous_turn(
    client, auth_headers, fake_llm: FakeLLM
):
    """§8.1 示例：第二次请求的 messages 应含第一轮的 user + assistant。"""
    # 文档同时含"退款"与"退货"，保证两问都能检索命中（否则走无结果分支不调 LLM）
    await _upload(
        client, auth_headers, "退款流程：先提交申请。如需退货请走同一退款流程"
    )
    r1 = await _ask(client, auth_headers, "退款要多久？")
    await _ask(client, auth_headers, "那退货呢？", session_id=r1["session_id"])

    assert len(fake_llm.calls) == 2, "两轮都应真正调 LLM"
    msgs = fake_llm.calls[-1]["messages"]
    assert any(m["content"] == "退款要多久？" for m in msgs)
    assert msgs[0]["role"] == "system"          # system 恒在最前（缓存友好）


@pytest.mark.asyncio
async def test_session_list_sorted_by_last_active(
    client, auth_headers, fake_llm: FakeLLM
):
    await _upload(client, auth_headers, "库存退款与保修政策常见问题")
    a = await _ask(client, auth_headers, "保修多久")
    b = await _ask(client, auth_headers, "退款流程")
    # 再碰一次会话 a → 它应排到最前（§3.3：按 last_active_at 不是 created_at）
    await _ask(client, auth_headers, "保修范围", session_id=a["session_id"])

    r = await client.get("/api/v1/chat/sessions", headers=auth_headers)
    assert r.status_code == 200
    ids = [s["id"] for s in r.json()["items"]]
    assert ids[0] == a["session_id"], "刚活跃的会话必须排最前"
    assert b["session_id"] in ids


@pytest.mark.asyncio
async def test_rename_and_delete_session(client, auth_headers):
    await _upload(client, auth_headers, "退款流程：先提交申请")
    out = await _ask(client, auth_headers, "退款流程")
    sid = out["session_id"]

    r = await client.patch(
        f"/api/v1/chat/sessions/{sid}",
        headers=auth_headers,
        json={"title": "我的退款讨论"},
    )
    assert r.status_code == 200
    assert r.json()["title"] == "我的退款讨论"

    r = await client.delete(f"/api/v1/chat/sessions/{sid}", headers=auth_headers)
    assert r.status_code == 204

    # 已删 → 再访问 404；会话列表里没有
    r = await client.get(f"/api/v1/chat/sessions/{sid}", headers=auth_headers)
    assert r.status_code == 404
    r = await client.get("/api/v1/chat/sessions", headers=auth_headers)
    assert sid not in [s["id"] for s in r.json()["items"]]


@pytest.mark.asyncio
async def test_deleting_session_keeps_cost_logs(client, auth_headers, session_factory):
    """★ §2.3：会话是业务数据可以删；chat_logs 是审计数据必须留着做成本分析。"""
    await _upload(client, auth_headers, "退款流程：先提交申请")
    out = await _ask(client, auth_headers, "退款流程")
    sid = out["session_id"]

    r = await client.delete(f"/api/v1/chat/sessions/{sid}", headers=auth_headers)
    assert r.status_code == 204

    async with session_factory() as s:
        logs = (
            await s.execute(select(ChatLog).where(ChatLog.question == "退款流程"))
        ).scalars().all()
        assert logs, "删除会话后成本记录必须还在"
