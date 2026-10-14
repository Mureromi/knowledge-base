"""Langfuse 追踪封装（阶段 6，docs/项目实现文档-阶段6.md §3）。

★ 与既有基础设施的关系：
  - 结构化日志（阶段 4）负责"机器可读的运行事件"，是**必选**的
  - Langfuse 负责"一次问答的内部拆解与可视化"，是**可选增强**
  两者互补，不互相替代。LANGFUSE_ENABLED=false（默认）时所有 trace 调用退化为 no-op，
  功能完全不受影响（§8 验收项）——这也是"没账号也能跑全部测试"的前提。

★ 隐私边界（必须明确，§9）：开启后**用户问题、检索到的切片内容、模型回答**都会发往
  Langfuse Cloud。个人知识库自用可接受；文档含敏感内容时必须改用自托管。

★ 版本适配（实现时核实）：文档 §3.2 伪代码写的是 `start_as_current_span`，
  但已安装的 **langfuse 4.17.0 没有该方法**——它改用 OTel 风格的
  `start_as_current_observation(name=..., as_type=...)`。这里按真实 API 实现，
  封装在 trace_span 里，业务层调用面与伪代码保持一致（换 API 只改这一个文件）。
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

_client: Any | None = None


def startup() -> None:
    """在 lifespan 里调用。未启用 / 缺 key / 鉴权失败时留空，后续调用自动 no-op。"""
    global _client
    if not settings.langfuse_enabled:
        logger.info('{"event":"tracing","enabled":false,"note":"no-op（LANGFUSE_ENABLED=false）"}')
        return
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        logger.warning("Langfuse 已启用但缺少 key，追踪退化为 no-op")
        return
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        # ★ auth_check() 在启动时就发现 key 配错，而不是等到第一次问答才发现（§3.2）
        if client.auth_check():
            _client = client
            logger.info("Langfuse 已连接: %s", settings.langfuse_host)
        else:
            logger.error("Langfuse 鉴权失败（key 或 host 不对），追踪退化为 no-op")
            _client = None
    except Exception as exc:                     # noqa: BLE001
        logger.error("Langfuse 连接失败，追踪退化为 no-op: %s", exc)
        _client = None


def shutdown() -> None:
    """★ 必须调用：Langfuse 是异步批量上报，不 flush 会丢掉最后的 trace（§3.3）。

    web 与 worker 的 shutdown 里都要调——worker 长期运行、退出时才上报，
    漏了 flush 整个 worker 的 trace 全丢。
    """
    global _client
    if _client is not None:
        try:
            _client.flush()
        except Exception as exc:                 # noqa: BLE001
            logger.warning("Langfuse flush 失败: %s", exc)
        _client = None


def is_enabled() -> bool:
    return _client is not None


@contextmanager
def trace_span(
    name: str,
    *,
    as_type: str = "span",
    attach_to: Any | None = None,
    **metadata: Any,
) -> Iterator[Any]:
    """记录一个 span/generation。未启用时是纯粹的 no-op（几乎零开销）。

    用法（业务层）：
        with trace_span("retrieval") as span:
            chunks = await search(...)
            update_span(span, output=..., metadata={...})   # span 为 None 时自动跳过

    as_type="generation" 用于 llm_call（可携带 usage/cost）。
    attach_to=一次问答的根 trace（new_trace() 所得）——把各 span 挂到同一条
    trace 下（§3.4 的树状结构）；为 None 时各自成独立 trace。
    """
    if _client is None:
        yield None
        return
    kwargs: dict = {"name": name, "as_type": as_type}
    if attach_to is not None:
        kwargs["trace_context"] = attach_to
    with _client.start_as_current_observation(**kwargs) as span:
        if metadata:
            try:
                span.update(metadata=metadata)
            except Exception as exc:             # noqa: BLE001
                logger.warning("Langfuse span metadata 写入失败: %s", exc)
        yield span


def new_trace() -> Any | None:
    """为一次问答开一条新 trace，返回可传给 trace_span(attach_to=...) 的上下文。

    为什么需要：langfuse 4.x 里 `start_as_current_observation` 靠当前 OTel 上下文找父节点，
    而我们的流水线被拆在多个 await 段里（尤其流式跨多次 yield），没有一个"包住整个
    请求"的 with 块可开根 span。于是：先建一个 trace_id，各 span 显式 attach 到它（§3.4）。
    未启用时返回 None → attach_to=None → 行为与平时完全一致（no-op）。
    """
    if _client is None:
        return None
    try:
        return {"trace_id": _client.create_trace_id()}
    except Exception as exc:                     # noqa: BLE001
        logger.warning("Langfuse 创建 trace 失败（降级为各自独立 trace）: %s", exc)
        return None


def update_span(span: Any, **fields: Any) -> None:
    """安全地给已开启的 span 补结果（未启用时 span 为 None → no-op）。

    单独抽出来是因为埋点处常在 await 之后才拿得到结果（检索顺序、tokens 用量），
    且要保证 span=None 时不炸——这样业务代码里可以无脑调用，不必每处判空。
    """
    if span is None or not fields:
        return
    try:
        span.update(**fields)
    except Exception as exc:                     # noqa: BLE001
        logger.warning("Langfuse span 更新失败: %s", exc)
