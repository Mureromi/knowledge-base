"""多轮对话下的缓存正确性测试（阶段 2 §4 + §8.3）。

三条防回归里最关键的两条在这里：
1. 防止 §3.2 写入顺序写错 → 同问题第二次能命中缓存；
2. 防止会话间串答案 → 历史不同的两个会话，同问题绝不互相命中。

★ 一条文档内部自相矛盾、以语义正确性为准的说明（§4.2 vs §8.3）：
  同一会话里连问两次同一问题，第二次的窗口**确实多了第一轮**——历史变了、
  指纹必须变、未命中是唯一语义正确的结果（§4.2 表格第 2 行明确承认这一点）。
  若强行让它命中，等价于"少算了影响输出的输入"，就是串上下文。
  防 §3.2 顺序 bug 的职责由 test_repeated_question_across_fresh_sessions 承担：
  顺序一旦写错，连那个场景都会静默永不命中。
"""

import pytest
from httpx import AsyncClient

from app.core.config import settings
from app.tests.conftest import FakeLLM


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


async def _ask(client: AsyncClient, headers: dict, question: str, session_id=None):
    body = {"question": question}
    if session_id is not None:
        body["session_id"] = session_id
    r = await client.post("/api/v1/chat/sync", headers=headers, json=body)
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.asyncio
async def test_repeated_question_across_fresh_sessions_hits_cache(
    client, auth_headers, fake_llm: FakeLLM
):
    """★ 防 §3.2 顺序 bug：不传 session_id（每次都是空窗口），
    一旦实现改成"先写消息再读窗口"，窗口会含本轮问题、键永远错位、这里必挂。"""
    await _upload(client, auth_headers, "退款流程：先提交申请，再等待审核")

    first = await _ask(client, auth_headers, "退款流程")
    assert first["usage"]["cache_hit"] is False
    second = await _ask(client, auth_headers, "退款流程")
    assert second["usage"]["cache_hit"] is True
    assert len(fake_llm.calls) == 1, "命中缓存时绝不能再调 LLM"


@pytest.mark.asyncio
async def test_two_sessions_with_different_history_do_not_share_cache(
    client, auth_headers, fake_llm: FakeLLM
):
    """★ 防串会话（§8.3 第二条）：两个会话各自聊过不同内容后，
    再问同一句"要提交申请吗"——历史指纹不同，第二个绝不能命中第一个的答案。"""
    await _upload(
        client, auth_headers, "退款流程：提交申请。发货流程：下单后发出"
    )

    a = await _ask(client, auth_headers, "退款流程")   # 会话 A 有了退款历史
    b = await _ask(client, auth_headers, "发货流程")   # 会话 B 有了发货历史

    followup_a = await _ask(client, auth_headers, "要提交申请吗", session_id=a["session_id"])
    assert followup_a["usage"]["cache_hit"] is False
    n = len(fake_llm.calls)

    followup_b = await _ask(client, auth_headers, "要提交申请吗", session_id=b["session_id"])
    assert followup_b["usage"]["cache_hit"] is False, (
        "B 的历史窗口与 A 不同，绝不能命中 A 刚写入的缓存"
    )
    assert len(fake_llm.calls) == n + 1, "B 必须自己调一次 LLM"

    # 并且两次请求带给 LLM 的历史确实是各自会话的
    msgs_a = fake_llm.calls[-2]["messages"]
    msgs_b = fake_llm.calls[-1]["messages"]
    assert any("退款流程" in m["content"] for m in msgs_a)
    assert not any("退款流程" in m["content"] for m in msgs_b[1:-1]), (
        "B 的历史里不该混进 A 的会话内容"
    )


@pytest.mark.asyncio
async def test_same_session_repeat_regenerates_with_updated_context(
    client, auth_headers, fake_llm: FakeLLM
):
    """同会话连问两次同问题：未命中是**正确行为**（窗口真的变了），
    但第二次必须携带第一轮历史，保证答案随上下文重新生成。"""
    await _upload(client, auth_headers, "退款流程：先提交申请，再等待审核")

    out = await _ask(client, auth_headers, "退款要多久")
    sid = out["session_id"]
    again = await _ask(client, auth_headers, "退款要多久", session_id=sid)

    assert again["usage"]["cache_hit"] is False
    assert len(fake_llm.calls) == 2
    # 第二次的 messages 含第一轮问答（历史进入上下文）
    msgs = fake_llm.calls[-1]["messages"]
    assert msgs[1]["role"] == "user" and msgs[1]["content"] == "退款要多久"


@pytest.mark.asyncio
async def test_uploading_new_doc_invalidates_multiturn_cache(
    client, auth_headers, fake_llm: FakeLLM
):
    """阶段 1 的检索失效机制在多轮下仍然有效：新文档进 Top-K → 未命中。"""
    await _upload(client, auth_headers, "退款流程：提交申请")
    await _ask(client, auth_headers, "退款流程")
    assert len(fake_llm.calls) == 1

    await _upload(client, auth_headers, "退款补充：审核 3 日到账，退款流程详见此")
    out = await _ask(client, auth_headers, "退款流程")
    assert out["usage"]["cache_hit"] is False, "检索指纹变化必须优先于历史命中"
    assert len(fake_llm.calls) == 2


@pytest.mark.asyncio
async def test_cache_keys_use_v2_namespace_after_multiturn_ask(
    client, auth_headers, fake_llm: FakeLLM, cache_backend
):
    """Redis 里写入的答案键必须是 kb:v2:answer:*（§4.3 版本升级兑现）。"""
    await _upload(client, auth_headers, "退款流程：先提交申请")
    await _ask(client, auth_headers, "退款流程")

    redis = cache_backend.redis
    keys = [k async for k in redis.scan_iter(match=f"{settings.cache_key_prefix}:*")]
    answer_keys = [k for k in keys if ":answer:" in k]
    assert answer_keys, "问答后 Redis 里应有答案键"
    assert all(":v2:answer:" in k for k in answer_keys), answer_keys
    assert not [k for k in keys if ":v1:answer:" in k], "不应再产生 v1 答案键"
