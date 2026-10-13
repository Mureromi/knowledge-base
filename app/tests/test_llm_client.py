"""LLM 客户端预检与请求头的回归测试。

背景：2026-10 真实冒烟测试失败，报错是
    LLMError: LLM 网络错误: Illegal header value b'Bearer '
根因：.env 不存在 → llm_api_key 为空 → 拼出 "Bearer " 这种非法请求头，
      httpx 在发出请求前就拒绝，错误被重试循环包装成"网络错误"，根因被完全掩盖。

本文件锁死两件事：
1. key 为空时必须 fail fast，且错误信息要能指出怎么修；
2. key 正常时，Authorization 头必须正确拼成 "Bearer <key>"（并带 DeepSeek 的 user_id 扩展字段）。
"""

import httpx
import pytest
import respx

from app.clients.llm import LLMClient, LLMError
from app.core.config import settings

CHAT_URL = f"{settings.llm_base_url}/chat/completions"

SUCCESS_BODY = {
    "id": "chatcmpl-test",
    "model": "deepseek-flash",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}}],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 2,
        "prompt_cache_hit_tokens": 8,   # 缓存命中字段，用于校准 _parse()
        "prompt_cache_miss_tokens": 4,
        "prompt_tokens_details": {"cached_tokens": 8},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
}

THINKING_BODY = {
    "id": "chatcmpl-think",
    "model": "deepseek-flash",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}}],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 187,                       # 实测：思考占绝大部分
        "prompt_cache_hit_tokens": 0,
        "completion_tokens_details": {"reasoning_tokens": 151},
    },
}


@pytest.fixture
def no_api_key(monkeypatch):
    """模拟 .env 不存在 / LLM_API_KEY 未填的状态。"""
    monkeypatch.setattr(
        "app.clients.llm.settings", settings.model_copy(update={"llm_api_key": ""})
    )


@pytest.fixture
def fake_api_key(monkeypatch):
    """模拟已正确配置 key 的状态。"""
    monkeypatch.setattr(
        "app.clients.llm.settings",
        settings.model_copy(update={"llm_api_key": "sk-test-1234567890"}),
    )


async def test_startup_without_key_fails_fast(no_api_key):
    """key 为空 → 抛 LLMError（而不是让 httpx 抛非法头），且提示要可操作。"""
    client = LLMClient()
    with pytest.raises(LLMError) as exc_info:
        await client.startup()
    message = str(exc_info.value)

    assert "LLM_API_KEY" in message
    assert ".env" in message                      # 必须告诉用户怎么修
    assert "Illegal header value" not in message  # 不再是那个看不懂的底层报错


async def test_startup_with_placeholder_key_fails_fast(monkeypatch):
    """key 还是 .env.example 的占位符 → 也要提前拦住，别白跑一次拿 401。"""
    monkeypatch.setattr(
        "app.clients.llm.settings",
        settings.model_copy(update={"llm_api_key": "sk-your-key-here"}),
    )
    client = LLMClient()
    with pytest.raises(LLMError) as exc_info:
        await client.startup()
    assert "占位符" in str(exc_info.value)


async def test_startup_with_key_succeeds(fake_api_key):
    """key 正常 → 能启动，且请求头正确。"""
    client = LLMClient()
    await client.startup()
    try:
        assert client._client is not None
        assert client._client.headers["Authorization"] == "Bearer sk-test-1234567890"
    finally:
        await client.shutdown()


@respx.mock
async def test_chat_sends_correct_headers_and_parses_usage(fake_api_key):
    """端到端（网络被 mock）：验证请求头、user_id 扩展字段、usage 解析与成本计算。"""
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=SUCCESS_BODY)
    )

    client = LLMClient()
    await client.startup()
    try:
        result = await client.chat(
            [{"role": "user", "content": "只回复：OK"}],
            max_tokens=10,
            user_id="u1",
        )
    finally:
        await client.shutdown()

    assert route.called
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer sk-test-1234567890"

    import json

    body = json.loads(request.content)
    assert body["model"] == settings.llm_model
    assert body["max_tokens"] == 10
    assert body["user_id"] == "u1"          # DeepSeek 多用户隔离扩展字段
    assert body["stream"] is False

    assert result.content == "OK"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 2
    assert result.cached_tokens == 8        # 缓存命中字段被正确解析
    assert result.cost_yuan > 0


@respx.mock
async def test_thinking_disabled_by_default(fake_api_key):
    """阶段 0 默认关闭思考模式：请求体要明确传 disabled，且传 temperature。

    实测依据（见 config.py 注释）：开启时 completion=187(思考 151)/1733ms/≈0.00188元，
    关闭后 completion=36/703ms/≈0.00049元，RAG 问答质量不变。
    """
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=SUCCESS_BODY)
    )
    client = LLMClient()
    await client.startup()
    try:
        await client.chat([{"role": "user", "content": "hi"}], temperature=0.2)
    finally:
        await client.shutdown()

    import json

    body = json.loads(route.calls.last.request.content)
    assert body["thinking"] == {"type": "disabled"}
    assert body["temperature"] == 0.2            # 非思考模式才传 temperature
    assert "reasoning_effort" not in body


@respx.mock
async def test_thinking_enabled_omits_temperature_and_counts_reasoning(fake_api_key):
    """开启思考模式时：不传 temperature（会被静默忽略），并解析出 reasoning_tokens。"""
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=THINKING_BODY)
    )
    client = LLMClient()
    await client.startup()
    try:
        result = await client.chat(
            [{"role": "user", "content": "hi"}], temperature=0.2, thinking=True
        )
    finally:
        await client.shutdown()

    import json

    body = json.loads(route.calls.last.request.content)
    assert body["thinking"] == {"type": "enabled"}
    assert "temperature" not in body             # 避免"以为设了 0.2 其实没生效"
    assert body["reasoning_effort"] == settings.llm_reasoning_effort

    assert result.reasoning_tokens == 151
    assert result.completion_tokens == 187
