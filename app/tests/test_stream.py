"""SSE 流式测试（阶段 2 §5 + §8.2 / §8.3）。

注意（§8.2）：httpx ASGITransport 会把整个流缓冲后一次性返回，
所以这里能拿到完整 SSE 文本做断言，但**测不出"是否真的逐块推送"**——
真要验证得用真实 uvicorn + 网络（本阶段记明此局限）。

覆盖 §8.3 的流式用例：事件顺序 / 缓存命中仍走流式 / keep-alive 过滤 /
[DONE] / reasoning 不进正文 / 流中错误 / 落库。
"""


import pytest
from httpx import AsyncClient

from app.clients.llm import LLMError, LLMResult, StreamChunk, get_llm
from app.main import app
from app.tests.conftest import FakeLLM, parse_sse


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


async def _stream(client: AsyncClient, headers: dict, question: str, session_id=None):
    body = {"question": question}
    if session_id is not None:
        body["session_id"] = session_id
    r = await client.post("/api/v1/chat", headers=headers, json=body)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")
    return r


@pytest.mark.asyncio
async def test_stream_event_order_and_content(client, auth_headers, fake_llm: FakeLLM):
    await _upload(client, auth_headers, "退款流程：先提交申请，再等待审核")
    r = await _stream(client, auth_headers, "退款流程")
    events = parse_sse(r.text)
    kinds = [e for e, _ in events]

    assert kinds[0] == "meta", "首条必须是 meta（前端要立刻拿 session_id）"
    assert kinds[-1] == "done", "末条必须是 done"
    assert "delta" in kinds

    meta = events[0][1]
    assert meta["session_id"] > 0
    assert meta["cache_hit"] is False

    # 拼接 delta 应等于 FakeLLM 的完整回答
    joined = "".join(d["text"] for e, d in events if e == "delta")
    assert joined == fake_llm.content
    # done 里带完整 usage
    done = events[-1][1]
    assert done["usage"]["cache_hit"] is False


@pytest.mark.asyncio
async def test_stream_citations_come_after_content(
    client, auth_headers, fake_llm: FakeLLM
):
    """引用在正文结束后、done 之前单独发（§5.3 设计决定 3）。"""
    await _upload(client, auth_headers, "退款流程：先提交申请，再等待审核")
    r = await _stream(client, auth_headers, "退款流程")
    kinds = [e for e, _ in parse_sse(r.text)]
    assert "citations" in kinds
    # citations 在所有 delta 之后、done 之前
    last_delta = max(i for i, k in enumerate(kinds) if k == "delta")
    cit_idx = kinds.index("citations")
    done_idx = kinds.index("done")
    assert last_delta < cit_idx < done_idx


@pytest.mark.asyncio
async def test_stream_cache_hit_skips_llm_but_still_streams(
    client, auth_headers, fake_llm: FakeLLM
):
    """命中缓存时：meta(cache_hit=true) + 切块 delta + done，且不调流式 LLM。"""
    await _upload(client, auth_headers, "退款流程：先提交申请，再等待审核")

    # 第一次：非流式，真正调 LLM 并写缓存
    await client.post(
        "/api/v1/chat/sync", headers=auth_headers, json={"question": "退款流程"}
    )
    assert len(fake_llm.calls) == 1

    # 第二次：流式问同样的（无 session_id → 空窗口 → 命中缓存）
    r = await _stream(client, auth_headers, "退款流程")
    events = parse_sse(r.text)
    assert events[0][1]["cache_hit"] is True
    joined = "".join(d["text"] for e, d in events if e == "delta")
    assert joined == fake_llm.content
    assert fake_llm.stream_calls == [], "命中缓存不应触发流式 LLM 调用"


@pytest.mark.asyncio
async def test_stream_persists_messages(client, auth_headers, fake_llm: FakeLLM):
    """流结束后 user + assistant 两条消息落库，会话详情能读到（§8.3）。"""
    await _upload(client, auth_headers, "退款流程：先提交申请，再等待审核")
    r = await _stream(client, auth_headers, "退款流程")
    sid = parse_sse(r.text)[0][1]["session_id"]

    detail = await client.get(f"/api/v1/chat/sessions/{sid}", headers=auth_headers)
    msgs = detail.json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "退款流程"


@pytest.mark.asyncio
async def test_thinking_mode_reasoning_not_in_delta(
    client, auth_headers, fake_llm: FakeLLM, monkeypatch
):
    """★ 防"打开思考模式用户看到思维链"：reasoning_content 走独立事件，不进 delta。"""

    class ThinkingFake:
        async def chat(self, messages, **kwargs):
            raise AssertionError("流式路径不该走 chat")

        async def chat_stream(self, messages, **kwargs):
            fake_llm.stream_calls.append({"messages": messages, **kwargs})
            yield StreamChunk(kind="reasoning", text="用户在问退款，我先想想内部独白")
            yield StreamChunk(kind="content", text="退款需")
            yield StreamChunk(kind="reasoning", text="更多不该露出的内心戏")
            yield StreamChunk(kind="content", text="先提交申请")
            yield StreamChunk(
                kind="usage",
                result=LLMResult(
                    content="", prompt_tokens=100, completion_tokens=20,
                    cached_tokens=0, latency_ms=80, model="deepseek-flash",
                    cost_yuan=0.0001,
                ),
            )
            yield StreamChunk(kind="done")

    app.dependency_overrides[get_llm] = lambda: ThinkingFake()
    await _upload(client, auth_headers, "退款流程：先提交申请，再等待审核")
    r = await _stream(client, auth_headers, "退款流程")
    events = parse_sse(r.text)

    deltas = "".join(d["text"] for e, d in events if e == "delta")
    assert deltas == "退款需先提交申请", "正文只含 content，绝不混入思维链"
    assert "内心戏" not in deltas and "先想想" not in deltas
    # 思维链单独成 reasoning 事件（前端默认忽略即可）
    reasonings = [d["text"] for e, d in events if e == "reasoning"]
    assert reasonings == ["用户在问退款，我先想想内部独白", "更多不该露出的内心戏"]


@pytest.mark.asyncio
async def test_stream_mid_error_emits_error_event(
    client, auth_headers, fake_llm: FakeLLM
):
    """流中途出错：HTTP 仍 200，收到 event: error + done(incomplete)，非静默断开。"""

    class BrokenStreamFake:
        async def chat(self, messages, **kwargs):
            raise AssertionError

        async def chat_stream(self, messages, **kwargs):
            yield StreamChunk(kind="content", text="退款流程是")
            raise LLMError("上游 500", status_code=500, retryable=True)

    app.dependency_overrides[get_llm] = lambda: BrokenStreamFake()
    await _upload(client, auth_headers, "退款流程：先提交申请，再等待审核")
    r = await _stream(client, auth_headers, "退款流程")
    events = parse_sse(r.text)
    kinds = [e for e, _ in events]

    assert "error" in kinds, "必须发 error 事件，不能静默关闭"
    err = next(d for e, d in events if e == "error")
    assert err["code"] == 50201
    done = events[-1]
    assert done[0] == "done" and done[1].get("incomplete") is True
    # 已生成的部分内容照常落库（便于排障）
    assert any(d.get("text") == "退款流程是" for e, d in events if e == "delta")


@pytest.mark.asyncio
async def test_parse_sse_ignores_keepalive_comment():
    """§1.3：SSE 里的 `: keep-alive` 注释行不能当数据处理（parse_sse 要跳过）。"""
    raw = (
        ": keep-alive\n\n"
        "event: meta\ndata: {\"session_id\": 1, \"cache_hit\": false}\n\n"
        "\n\n"
        "event: done\ndata: {\"usage\": null}\n\n"
    )
    events = parse_sse(raw)
    kinds = [e for e, _ in events]
    assert kinds == ["meta", "done"]


# ---- LLMClient.chat_stream 底层解析（喂假 SSE 行，覆盖 keep-alive/[DONE]/reasoning）----


class _FakeStreamResp:
    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code

    async def aread(self):
        return b"{}"

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeHttpClient:
    def __init__(self, lines, status_code=200):
        self._lines = lines
        self._status = status_code

    def stream(self, method, url, json=None):
        return _FakeStreamResp(self._lines, self._status)


@pytest.mark.asyncio
async def test_chat_stream_filters_keepalive_and_stops_at_done():
    """§1.2 + §1.3：真实 DeepSeek 流会穿插空行与 `: keep-alive`，解析器必须无视，
    遇到 [DONE] 正常收尾，末尾给出 usage 事件。"""
    from app.clients.llm import LLMClient

    llm = LLMClient()
    llm._client = _FakeHttpClient(
        [
            ": keep-alive",
            "",
            'data: {"choices":[{"delta":{"content":"你"}}]}',
            "data: {\"choices\":[{\"delta\":{\"content\":\"好\"}}]}",
            'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2}}',
            "data: [DONE]",
            "data: {\"choices\":[{\"delta\":{\"content\":\"绝不该出现\"}}]}",
        ]
    )
    chunks = [c async for c in llm.chat_stream([{"role": "user", "content": "hi"}])]
    text = "".join(c.text for c in chunks if c.kind == "content")
    assert text == "你好", "[DONE] 之后的数据必须被丢弃"
    assert "绝不该出现" not in text
    kinds = [c.kind for c in chunks]
    assert kinds == ["content", "content", "usage", "done"]


@pytest.mark.asyncio
async def test_chat_stream_separates_reasoning_from_content():
    """§1.4：delta 里的 reasoning_content 走 kind=reasoning，不混进 content。"""
    from app.clients.llm import LLMClient

    llm = LLMClient()
    llm._client = _FakeHttpClient(
        [
            'data: {"choices":[{"delta":{"reasoning_content":"内心独白"}}]}',
            'data: {"choices":[{"delta":{"content":"正式答案"}}]}',
            "data: [DONE]",
        ]
    )
    chunks = [c async for c in llm.chat_stream([{"role": "user", "content": "x"}])]
    reasoning = "".join(c.text for c in chunks if c.kind == "reasoning")
    content = "".join(c.text for c in chunks if c.kind == "content")
    assert reasoning == "内心独白"
    assert content == "正式答案"
