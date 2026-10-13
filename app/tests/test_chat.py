"""问答端到端测试（LLM 全 mock）。

对应 §9 表格里 test_chat 需要覆盖的 3 项：
- 无相关资料 → 不调用 LLM
- 有资料 → prompt 里包含参考资料
- chat_logs 落库
另外：缓存友好性的回归测试——断言 system prompt 里没有时间戳等易变内容。
"""

import re

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.llm import LLMError, get_llm
from app.main import app
from app.models.chat_log import ChatLog
from app.services.qa import SYSTEM_PROMPT
from app.tests.conftest import FakeLLM, register_and_login


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


@pytest.mark.asyncio
async def test_no_relevant_chunk_does_not_call_llm(client, fake_llm: FakeLLM):
    """知识库里没有相关内容时，绝不发起 LLM 请求（省成本、避免幻觉）。"""
    token = await register_and_login(client, "solo_a", "secret123")
    headers = {"Authorization": f"Bearer {token}"}
    # 什么都不上传，直接问
    r = await client.post(
        "/api/v1/chat/sync", headers=headers, json={"question": "如何调试 Kubernetes"}
    )
    assert r.status_code == 200, r.text
    assert fake_llm.calls == [], "无检索结果时不应调用 LLM"
    assert r.json()["citations"] == []
    assert r.json()["usage"]["cost_yuan"] == 0


@pytest.mark.asyncio
async def test_prompt_contains_retrieved_context_and_logs_persist(
    client, fake_llm: FakeLLM, session_factory
):
    token = await register_and_login(client, "solo_b", "secret123")
    headers = {"Authorization": f"Bearer {token}"}
    doc_id = await _upload(
        client, headers, "policy.md", "退款需先提交申请，再等待审核"
    )

    r = await client.post(
        "/api/v1/chat/sync", headers=headers, json={"question": "退款流程", "top_k": 3}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["citations"], "应命中至少 1 条引用"
    assert body["citations"][0]["document_id"] == doc_id

    # 断言传给 LLM 的 user prompt 包含参考资料
    assert len(fake_llm.calls) == 1
    messages = fake_llm.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "【参考资料】" in messages[1]["content"]
    assert "退款" in messages[1]["content"]

    # 断言 chat_logs 落库
    async with session_factory() as s:
        assert isinstance(s, AsyncSession)
        rows = (
            await s.execute(select(ChatLog).where(ChatLog.answer == fake_llm.content))
        ).scalars().all()
        assert len(rows) >= 1
        log = rows[0]
        assert log.prompt_tokens == 1000
        assert log.completion_tokens == 50
        assert log.cited_chunk_ids, "引用切片 id 列表应已写入"


@pytest.mark.asyncio
async def test_system_prompt_is_static_for_cache(client, fake_llm: FakeLLM):
    """缓存命中价差 50 倍：SYSTEM_PROMPT 里绝不能出现时间戳、日期、随机值。

    这是"成本意识"的自动化回归测试（§9 结尾要求）。
    """
    # 1. 常量本身不含时间戳模式（13 位毫秒 / ISO 日期 / Unix 时间戳）
    assert not re.search(r"\b\d{13}\b", SYSTEM_PROMPT)
    assert not re.search(r"\d{4}-\d{2}-\d{2}", SYSTEM_PROMPT)
    assert not re.search(r"\b\d{10}\b", SYSTEM_PROMPT)
    # 2. 也不含"当前时间/日期/now/timestamp"等字样
    lowered = SYSTEM_PROMPT.lower()
    for kw in ("当前时间", "日期", "timestamp", "now()", "uuid"):
        assert kw not in lowered, f"系统提示词不应包含易变字段：{kw}"

    # 3. 连续两次问答，system 消息应完全一致
    token = await register_and_login(client, "solo_c", "secret123")
    headers = {"Authorization": f"Bearer {token}"}
    await _upload(client, headers, "a.md", "苹果 香蕉")

    await client.post("/api/v1/chat/sync", headers=headers, json={"question": "苹果"})
    await client.post("/api/v1/chat/sync", headers=headers, json={"question": "香蕉"})

    assert len(fake_llm.calls) == 2
    sys_a = fake_llm.calls[0]["messages"][0]["content"]
    sys_b = fake_llm.calls[1]["messages"][0]["content"]
    assert sys_a == sys_b == SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_llm_error_returns_502_with_friendly_message(client, auth_headers):
    """§12 验收清单：LLM 暂时性故障（5xx/超时）→ 502 + 友好提示，不是 500 堆栈。

    临时把 get_llm 的 override 换成会抛 LLMError 的假客户端，
    验证 register_exception_handlers 能接住并返回统一错误体。
    """
    # 先上传一个文档，让检索能命中，确保真的会去调 LLM
    await _upload(client, auth_headers, "refund.md", "退款流程：先提交申请")

    class BrokenLLM:
        async def chat(self, messages, **kwargs):
            # 模拟上游 5xx（属于"暂时不可用"，客户端可重试）
            raise LLMError(
                "LLM 返回 503: upstream unavailable, trace_id=abc-123",
                status_code=503,
                retryable=True,
            )

    original_override = app.dependency_overrides.get(get_llm)
    app.dependency_overrides[get_llm] = lambda: BrokenLLM()
    try:
        r = await client.post(
            "/api/v1/chat/sync",
            headers=auth_headers,
            json={"question": "退款流程", "top_k": 3},
        )
        assert r.status_code == 502, r.text
        body = r.json()
        assert body["code"] == 50201
        assert "AI 服务暂时不可用" in body["message"]
        # 不能泄露堆栈或上游响应细节
        assert body["detail"] is None
        assert "trace_id" not in r.text
        assert "abc-123" not in r.text
    finally:
        # 还原，避免污染其他测试（虽然 client fixture 会 clear，但显式更稳）
        if original_override is not None:
            app.dependency_overrides[get_llm] = original_override
        else:
            app.dependency_overrides.pop(get_llm, None)


@pytest.mark.asyncio
async def test_llm_auth_error_is_distinguishable_from_outage(client, auth_headers):
    """401/403 是"我们自己的 key 配错了"，必须与"上游挂了"区分开。

    若两者都返回同一个 502 + "AI 服务暂时不可用"，排障时会被误导成上游故障，
    方向完全错——所以要断言错误码与文案确实不同。
    """
    await _upload(client, auth_headers, "auth.md", "退款流程：先提交申请")

    class UnauthorizedLLM:
        async def chat(self, messages, **kwargs):
            raise LLMError(
                "LLM 请求失败 401: invalid api key",
                status_code=401,
                retryable=False,
            )

    original_override = app.dependency_overrides.get(get_llm)
    app.dependency_overrides[get_llm] = lambda: UnauthorizedLLM()
    try:
        r = await client.post(
            "/api/v1/chat/sync",
            headers=auth_headers,
            json={"question": "退款流程", "top_k": 3},
        )
        assert r.status_code == 502, r.text
        body = r.json()
        assert body["code"] == 50202                    # 与 50201 区分
        assert "鉴权失败" in body["message"]
        assert "暂时不可用" not in body["message"]       # 不能误报成上游故障
        assert body["detail"] is None
        assert "invalid api key" not in r.text          # 不泄露上游原文
    finally:
        if original_override is not None:
            app.dependency_overrides[get_llm] = original_override
        else:
            app.dependency_overrides.pop(get_llm, None)
