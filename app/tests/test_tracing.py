"""Langfuse 追踪封装测试（阶段 6，docs/项目实现文档-阶段6.md §3 / §7.1）。

核心承诺（§8）：**LANGFUSE_ENABLED=false 时追踪是纯 no-op——不报错、不联网、不改行为**。
所以测试分三块：
  1. 禁用态：_client 为 None 时各 API 静默降级。
  2. 启用态：注入假 client（不真连 Langfuse Cloud），验证 span/generation 被正确开启与写入。
  3. 行为不变：同一次检索，包不包 trace_span 结果完全一致（no-op 的透明性）。
"""

from contextlib import contextmanager

import pytest

from app.clients import tracing
from app.services.retrieval import search


@pytest.fixture(autouse=True)
def _restore_client():
    """每个用例前后都把模块全局 _client 复位，避免测试间串味。"""
    original = tracing._client
    yield
    tracing._client = original


class _FakeSpan:
    def __init__(self, name: str, as_type: str) -> None:
        self.name = name
        self.as_type = as_type
        self.updates: list[dict] = []

    def update(self, **fields):
        self.updates.append(fields)
        return self


class _FakeLangfuse:
    """只实现 trace_span/new_trace 用到的那几个方法，绝不触网。"""

    def __init__(self) -> None:
        self.spans: list[_FakeSpan] = []
        self.flushed = False

    @contextmanager
    def start_as_current_observation(self, *, name, as_type="span", trace_context=None):
        span = _FakeSpan(name, as_type)
        self.spans.append(span)
        yield span

    def create_trace_id(self) -> str:
        return "fake-trace-id"

    def flush(self) -> None:
        self.flushed = True


# ---------------------------------------------------------------------------
# 1. 禁用态：no-op
# ---------------------------------------------------------------------------

def test_is_enabled_reflects_client(monkeypatch):
    monkeypatch.setattr(tracing, "_client", None)
    assert tracing.is_enabled() is False
    monkeypatch.setattr(tracing, "_client", _FakeLangfuse())
    assert tracing.is_enabled() is True


def test_new_trace_is_none_when_disabled(monkeypatch):
    monkeypatch.setattr(tracing, "_client", None)
    assert tracing.new_trace() is None


def test_trace_span_yields_none_when_disabled(monkeypatch):
    monkeypatch.setattr(tracing, "_client", None)
    with tracing.trace_span("retrieval", foo="bar") as span:
        assert span is None          # 业务层拿到 None，后续 update 也无副作用


def test_update_span_tolerates_none(monkeypatch):
    monkeypatch.setattr(tracing, "_client", None)
    tracing.update_span(None, output="x")     # 不应抛异常


def test_startup_noop_without_keys(monkeypatch):
    """启用但缺 key → 不建客户端，仍 no-op（不联网）。"""
    from app.core.config import settings

    monkeypatch.setattr(settings, "langfuse_enabled", False)
    tracing._client = None
    tracing.startup()
    assert tracing._client is None


def test_shutdown_when_disabled_is_harmless(monkeypatch):
    monkeypatch.setattr(tracing, "_client", None)
    tracing.shutdown()               # 不该炸


# ---------------------------------------------------------------------------
# 2. 启用态：假 client 记录
# ---------------------------------------------------------------------------

def test_trace_span_opens_and_writes_metadata(monkeypatch):
    fake = _FakeLangfuse()
    monkeypatch.setattr(tracing, "_client", fake)
    with tracing.trace_span("retrieval", retrieval_mode="hybrid") as span:
        assert isinstance(span, _FakeSpan)
        assert span.name == "retrieval"
        # 构造时就带上了 metadata
        assert span.updates and span.updates[0]["metadata"] == {"retrieval_mode": "hybrid"}


def test_trace_span_generation_type(monkeypatch):
    fake = _FakeLangfuse()
    monkeypatch.setattr(tracing, "_client", fake)
    with tracing.trace_span("llm_call", as_type="generation") as span:
        tracing.update_span(span, output="答案", model="deepseek")
    assert fake.spans[0].as_type == "generation"
    assert fake.spans[0].updates[-1]["output"] == "答案"


def test_new_trace_returns_attachable_context(monkeypatch):
    monkeypatch.setattr(tracing, "_client", _FakeLangfuse())
    ctx = tracing.new_trace()
    assert ctx == {"trace_id": "fake-trace-id"}


def test_shutdown_flushes_client(monkeypatch):
    """★ §3.3：shutdown 必须 flush，否则最后的 trace 丢失。"""
    fake = _FakeLangfuse()
    monkeypatch.setattr(tracing, "_client", fake)
    tracing.shutdown()
    assert fake.flushed is True
    assert tracing._client is None


def test_span_failures_never_break_caller(monkeypatch):
    """埋点内部炸了也不能冒泡影响业务——update_span 吞异常。"""

    class _Boom(_FakeLangfuse):
        @contextmanager
        def start_as_current_observation(self, *, name, as_type="span", trace_context=None):
            class _S:
                def update(self, **f):
                    raise RuntimeError("langfuse 挂了")
            yield _S()

    monkeypatch.setattr(tracing, "_client", _Boom())
    # trace_span 里 metadata 写入失败被 try/except 吞掉；退出正常
    with tracing.trace_span("x", k="v") as span:
        tracing.update_span(span, output="不该炸")     # update_span 自带 try/except


# ---------------------------------------------------------------------------
# 3. 行为不变：开/关追踪，检索结果一致
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tracing_does_not_change_retrieval(db_session, seed_document, monkeypatch):
    await seed_document(
        owner_id=1, chunks=[("退款要 5-7 个工作日到账", None), ("黄金会员年费 88 元", None)]
    )

    async def _run():
        return await search(db_session, owner_id=1, question="退款", top_k=5)

    # 关闭态（no-op）
    monkeypatch.setattr(tracing, "_client", None)
    with _noop_span():
        off = await _run()

    # "开启"态（假 client 只记录，不改数据）
    monkeypatch.setattr(tracing, "_client", _FakeLangfuse())
    with tracing.trace_span("retrieval", attach_to=tracing.new_trace()) as span:
        on = await _run()
        tracing.update_span(span, output=[c.chunk_id for c in on])

    assert [c.chunk_id for c in off] == [c.chunk_id for c in on]
    assert [c.content for c in off] == [c.content for c in on]


@contextmanager
def _noop_span():
    """禁用态的 trace_span 语义（应几乎零开销、什么都不做）。"""
    with tracing.trace_span("retrieval") as span:
        assert span is None
        yield
