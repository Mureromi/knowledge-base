"""唯一的 LLM 出口。业务层只调 chat()，不碰 HTTP。

关键约束（见 §6.4）：
1. 保活内容：DeepSeek 非流式会持续返回空行；流式返回 `: keep-alive`。
   用 httpx + resp.json() 没问题，但阶段 2 自己解析 SSE 时必须先过滤。
2. 绝不用 requests：async def 里调用同步库会阻塞事件循环。必须 httpx.AsyncClient。
3. 客户端必须单例：在 lifespan 里 startup/shutdown。每请求新建会让 TLS 握手开销翻倍。
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

import httpx

from app.clients.pricing import calc_cost, is_peak_now
from app.core.config import settings

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

# .env.example 里的占位符，出现这些值说明用户忘了替换
PLACEHOLDER_KEYS = {
    "sk-your-key-here",
    "sk-xxxxxxxxxxxxxxxx",
    "your-api-key",
    "changeme",
}


class LLMError(RuntimeError):
    """LLM 调用失败的统一异常，业务层只需捕获它。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


@dataclass
class LLMResult:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0      # 思考模式产生的思维链 token（计入 completion_tokens）
    latency_ms: int = 0
    model: str = ""
    cost_yuan: float = 0.0
    peak: bool = False
    raw: dict = field(default_factory=dict)


@dataclass
class StreamChunk:
    """流式过程中的一个事件（阶段 2 §5.1）。"""

    kind: Literal["content", "reasoning", "usage", "done"]
    text: str = ""
    # 仅 kind="usage" 时携带，由 _parse() 统一解析（字段解析只留一处实现）
    result: LLMResult | None = None


class StreamError(LLMError):
    """流式过程中的错误。此时 HTTP 200 已经发出，无法再改状态码（§5.6）。"""


class LLMClient:
    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def startup(self) -> None:
        # 预检：key 为空时，httpx 会在发送前抛 "Illegal header value b'Bearer '"，
        # 从重试循环里冒出来变成"网络错误"，完全看不出根因。这里提前拦住并说清怎么修。
        if not settings.llm_api_key.strip():
            raise LLMError(
                "LLM_API_KEY 为空，无法调用 LLM。\n"
                "  修复：cp .env.example .env 然后填入真实 key（申请地址 "
                "https://platform.deepseek.com/api_keys）\n"
                "  注意：.env 必须位于启动进程的工作目录（uvicorn 要在工程根目录启动）"
            )
        # 占位符也拦掉：否则会白跑一次请求，拿回一个语焉不详的 401
        if settings.llm_api_key.strip() in PLACEHOLDER_KEYS:
            raise LLMError(
                "LLM_API_KEY 仍是 .env.example 里的占位符，不是真实 key。\n"
                "  修复：编辑 .env，把 LLM_API_KEY 换成你在 "
                "https://platform.deepseek.com/api_keys 申请的真实 key"
            )
        if not settings.llm_api_key.startswith("sk-"):
            logger.warning(
                "LLM_API_KEY 不以 sk- 开头（当前长度 %s），请确认是否填错",
                len(settings.llm_api_key),
            )

        self._client = httpx.AsyncClient(
            base_url=settings.llm_base_url,
            timeout=httpx.Timeout(
                connect=5.0,
                read=settings.llm_timeout_seconds,   # 必须 < 服务端 10 分钟保活上限
                write=30.0,
                pool=5.0,
            ),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=10),
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
        )

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("LLMClient 未初始化：请在 lifespan 中 await startup()")
        return self._client

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        user_id: str | None = None,
        json_mode: bool = False,
        thinking: bool | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResult:
        thinking = settings.llm_thinking if thinking is None else thinking

        payload: dict = {
            "model": settings.llm_model,
            "messages": messages,
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "stream": False,
            # DeepSeek 扩展：默认关闭思考模式（省 75% 成本、快 60%，见 config 注释）
            "thinking": {"type": "enabled" if thinking else "disabled"},
        }
        if thinking:
            # 思考模式下 temperature 会被静默忽略（官方文档明确说明），
            # 所以只在关闭思考时才传，避免"以为设了 0.2 其实没生效"的误解。
            payload["reasoning_effort"] = settings.llm_reasoning_effort
        else:
            payload["temperature"] = temperature
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if user_id:
            # DeepSeek 扩展：多用户隔离 + KVCache 隔离。不可含隐私信息。
            payload["user_id"] = user_id
        if tools:
            # ★ 阶段 5：Function Calling。显式声明 tool_choice=auto（默认值可能随版本变），
            #   让模型自己决定是否调工具。见 docs/项目实现文档-阶段5.md §5.3。
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        start = time.perf_counter()
        last_exc: Exception | None = None

        for attempt in range(1, settings.llm_max_retries + 1):
            try:
                resp = await self.client.post("/chat/completions", json=payload)
            except httpx.TimeoutException:
                last_exc = LLMError("LLM 请求超时", retryable=True)
                logger.warning("LLM 超时 attempt=%s", attempt)
            except httpx.RequestError as exc:
                last_exc = LLMError(f"LLM 网络错误: {exc}", retryable=True)
                logger.warning("LLM 网络错误 attempt=%s: %s", attempt, exc)
            else:
                if resp.status_code == 200:
                    return self._parse(resp.json(), start)
                if resp.status_code in RETRYABLE_STATUS:
                    last_exc = LLMError(
                        f"LLM 返回 {resp.status_code}: {resp.text[:300]}",
                        status_code=resp.status_code,
                        retryable=True,
                    )
                else:
                    # 4xx（除 408/429）是请求本身有问题，重试无意义
                    raise LLMError(
                        f"LLM 请求失败 {resp.status_code}: {resp.text[:300]}",
                        status_code=resp.status_code,
                        retryable=False,
                    )

            if attempt < settings.llm_max_retries:
                delay = min(2 ** (attempt - 1) * 0.5, 8.0)
                delay += time.perf_counter() % 0.3     # 抖动，避免 429 时齐步重试
                await asyncio.sleep(delay)

        raise last_exc or LLMError("LLM 调用失败")

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        user_id: str | None = None,
        thinking: bool | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """流式对话（阶段 2 §5.1）。

        实测结论（§1）：
        - DeepSeek 流式默认就返回 usage，**不需要** stream_options:{include_usage}
        - 结束标记是 `data: [DONE]`，必须显式判断
        - 等待推理时会穿插 `: keep-alive` 注释行，必须过滤，否则 json.loads 直接崩
        - 思考模式下 delta 有 reasoning_content / content 两个字段，必须分流

        重试策略（§5.2）：一旦有 chunk 推给客户端就绝不能重试（内容会重复/错乱）。
        本项目流式不做客户端重试，只靠建连超时，体验上"报错重问"优于"默默等重试"。
        """
        thinking = settings.llm_thinking if thinking is None else thinking

        payload: dict = {
            "model": settings.llm_model,
            "messages": messages,
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "stream": True,
            "thinking": {"type": "enabled" if thinking else "disabled"},
        }
        if thinking:
            payload["reasoning_effort"] = settings.llm_reasoning_effort
        else:
            payload["temperature"] = temperature
        if user_id:
            payload["user_id"] = user_id

        usage: dict | None = None
        start = time.perf_counter()

        try:
            async with self.client.stream(
                "POST", "/chat/completions", json=payload
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", errors="replace")[:300]
                    raise StreamError(
                        f"LLM 流式请求失败 {resp.status_code}: {body}",
                        status_code=resp.status_code,
                        retryable=resp.status_code in RETRYABLE_STATUS,
                    )

                async for line in resp.aiter_lines():
                    # ★ 过滤空行与 `: keep-alive` 注释（§1.3）：
                    #   本地测不出不代表线上没有，模型排队时注释行一定会来
                    if not line.strip() or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if data == "[DONE]":                  # §1.2
                        break

                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning("无法解析的 SSE 数据行: %s", data[:120])
                        continue

                    if obj.get("usage"):                  # 最后一个 chunk 携带 usage
                        usage = obj["usage"]

                    for choice in obj.get("choices") or []:
                        delta = choice.get("delta") or {}
                        # ★ 思维链单独分流，绝不能混进正文（§1.4）
                        if delta.get("reasoning_content"):
                            yield StreamChunk(
                                kind="reasoning", text=delta["reasoning_content"]
                            )
                        if delta.get("content"):
                            yield StreamChunk(kind="content", text=delta["content"])
        except httpx.RequestError as exc:
            raise StreamError(f"LLM 流式网络错误: {exc}", retryable=True) from exc

        # ★ 复用非流式的 _parse()：缓存 token 字段名、reasoning_tokens、成本、
        #   高峰判定全都只有一处实现。代价是 content 字段对流式无意义
        #   （正文已逐块发走），只看 usage/cost/latency。
        result = self._parse(
            {"usage": usage or {}, "choices": [], "model": settings.llm_model}, start
        )
        yield StreamChunk(kind="usage", result=result)
        yield StreamChunk(kind="done")

    def _parse(self, data: dict, start: float) -> LLMResult:
        latency_ms = int((time.perf_counter() - start) * 1000)
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        usage = data.get("usage") or {}

        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        # 实测（2026-10）DeepSeek 响应里两个字段同时存在，任取其一即可：
        #   "prompt_cache_hit_tokens": 512          ← 顶层的命中数
        #   "prompt_tokens_details": {"cached_tokens": 512}
        # 缓存按 512 token 块命中；价差 50 倍（0.02 vs 1.0 元/M），务必算准。
        cached_tokens = int(
            usage.get("prompt_cache_hit_tokens")
            or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            or 0
        )
        # 思考模式：思维链 token 被计入 completion_tokens，单独记下来用于成本优化观察
        reasoning_tokens = int(
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        )
        model = data.get("model") or settings.llm_model
        peak = is_peak_now()

        # ★ 阶段 5：把 message 级的 tool_calls / reasoning_content 上浮到 raw 顶层。
        #   真实响应里它们在 choices[0].message 下，而 Agent 循环（§5）习惯写
        #   result.raw.get("tool_calls")——不修正这里，真机上永远取到 None。
        #   测试用的 ScriptedLLM 直接把 tool_calls 放在 raw 顶层，两种写法从此统一。
        if message.get("tool_calls") is not None:
            data["tool_calls"] = message["tool_calls"]
        if message.get("reasoning_content") is not None:
            data["reasoning_content"] = message["reasoning_content"]

        return LLMResult(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            latency_ms=latency_ms,
            model=model,
            cost_yuan=calc_cost(
                model, prompt_tokens, completion_tokens, cached_tokens, peak
            ),
            peak=peak,
            raw=data,
        )


llm_client = LLMClient()


async def get_llm() -> LLMClient:
    """FastAPI 依赖，方便测试整体替换。"""
    return llm_client
