"""多级限流：L1 频率 / L2 并发 / L3 每日配额（阶段 1 文档 §6）。

Redis 故障策略由 RATELIMIT_FAIL_OPEN 控制：
  true （默认）= 放行。可用性优先。本项目是学习/内部工具，宁可多花钱也别让服务不可用。
  false        = 拒绝。成本优先。对外收费的产品应该这样。
⚠️ 这是产品决策，不是技术决策，部署到真实环境前必须明确选一个。

设计选择：**限流函数接受 CacheBackend 而不是调用全局 get_redis()**。
这样测试只需 override 一处 get_cache，与 §10.1 的建议一致（原方案是文档为展示"设计缺陷"故意留的）。
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import HTTPException, status

from app.clients.cache import CST, CacheBackend, MemoryCache, RedisCache, seconds_until_midnight
from app.core.config import settings

logger = logging.getLogger(__name__)

# L1 滑动窗口：清理过期 + 加入本次 + 计数 + 续期，四步原子完成
_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]

redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
redis.call('ZADD', key, now, member)
local count = redis.call('ZCARD', key)
redis.call('EXPIRE', key, window)
if count > limit then
  -- 超限：把自己刚加的那条撤回，避免"被拒绝的请求也占额度"（§6.3 常见 bug）
  redis.call('ZREM', key, member)
end
-- 无论是否拒绝，都返回加入后的逻辑计数：拒绝时返回 count（> limit），
-- Python 侧用 `if count > limit` 判定 429；实际存储的 ZSET 已回退。
return count
"""

# L2 并发信号量：只在 count <= max 时占用
_ACQUIRE_LUA = """
local key = KEYS[1]
local max_c = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local cur = tonumber(redis.call('GET', key) or '0')
if cur >= max_c then
  return 0
end
redis.call('INCR', key)
redis.call('EXPIRE', key, ttl)
return 1
"""

# L2 释放：只在 key 存在且 > 0 时 DECR，避免"过期后 DECR 变负数"导致永久占位
_RELEASE_LUA = """
local key = KEYS[1]
local cur = tonumber(redis.call('GET', key) or '0')
if cur > 0 then
  redis.call('DECR', key)
  return cur - 1
end
return 0
"""


def _fail_open() -> bool:
    return settings.ratelimit_fail_open


def _extract_redis(cache: CacheBackend):
    """从 CacheBackend 里取 raw redis client；MemoryCache 返回 None（视为未启用）。"""
    if isinstance(cache, MemoryCache):
        return None
    if isinstance(cache, RedisCache):
        return cache.redis
    # 其他自定义实现：有 .redis 属性就用，没有就跳过
    return getattr(cache, "redis", None)


async def check_rate_limit(cache: CacheBackend, owner_id: int) -> None:
    """L1：每用户在滑动窗口内的请求数。超限抛 429。

    在端点入口最早调用，最省资源（连检索都不做）。
    """
    if not settings.redis_enabled:
        return
    redis = _extract_redis(cache)
    if redis is None:
        return  # 内存降级：跳过 L1
    try:
        now = time.time()
        member = f"{now:.6f}-{uuid.uuid4().hex[:8]}"
        key = f"{settings.cache_key_prefix}:v1:rl:user:{owner_id}"
        count = await redis.eval(
            _SLIDING_WINDOW_LUA,
            1,
            key,
            now,
            settings.rate_limit_window_seconds,
            settings.rate_limit_per_minute,
            member,
        )
        if int(count) > settings.rate_limit_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "code": 42901,
                    "message": (
                        f"请求过于频繁，{settings.rate_limit_window_seconds} 秒内"
                        f"最多 {settings.rate_limit_per_minute} 次"
                    ),
                    "detail": None,
                },
                headers={"Retry-After": str(settings.rate_limit_window_seconds)},
            )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - Redis 故障不能影响主流程
        logger.warning("L1 限流检查失败: %s", exc)
        if not _fail_open():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": 50302,
                    "message": "限流服务不可用，请稍后重试",
                    "detail": None,
                },
            ) from exc


async def check_daily_quota(cache: CacheBackend, owner_id: int) -> None:
    """L3：每用户每日 LLM 调用次数。

    ★ **只在缓存未命中、即将真正调 LLM 时调用**——见 §6.3.1，
      如果在入口就扣，缓存命中的请求也吃配额，等于把省下来的钱又扣掉了。
    """
    if not settings.redis_enabled:
        return
    redis = _extract_redis(cache)
    if redis is None:
        return
    try:
        today = datetime.now(CST).strftime("%Y%m%d")
        key = f"{settings.cache_key_prefix}:v1:quota:user:{owner_id}:{today}"
        used = await redis.incr(key)
        if int(used) == 1:
            # 首次写这个 key：TTL 到次日零点，而不是固定 86400（否则窗口会漂移）
            await redis.expire(key, seconds_until_midnight())
        if int(used) > settings.daily_llm_quota:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "code": 42902,
                    "message": (
                        f"今日 AI 调用额度已用完（{settings.daily_llm_quota} 次），"
                        "明日零点重置"
                    ),
                    "detail": None,
                },
            )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("L3 配额检查失败: %s", exc)
        if not _fail_open():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": 50302,
                    "message": "配额服务不可用，请稍后重试",
                    "detail": None,
                },
            ) from exc


@asynccontextmanager
async def distributed_llm_slot(cache: CacheBackend) -> AsyncIterator[None]:
    """L2：全局并发槽位。替代阶段 0 的进程内 asyncio.Semaphore。

    用轮询等待而不是阻塞队列——实现简单、不会死锁。
    并发上限 10 时，第 11 个请求最多等约 0.3 秒，完全可接受。
    """
    if not settings.redis_enabled:
        yield
        return
    redis = _extract_redis(cache)
    if redis is None:
        yield
        return

    key = f"{settings.cache_key_prefix}:v1:conc:llm"
    acquired = False
    deadline = time.monotonic() + 10.0  # 最多等 10 秒

    try:
        while time.monotonic() < deadline:
            try:
                ok = await redis.eval(
                    _ACQUIRE_LUA,
                    1,
                    key,
                    settings.llm_max_concurrency,
                    30,  # 安全 TTL：进程崩溃时槽位 30 秒后自动释放
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("L2 并发控制失败（放行）: %s", exc)
                if not _fail_open():
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail={
                            "code": 50303,
                            "message": "AI 服务繁忙，请稍后重试",
                            "detail": None,
                        },
                    ) from exc
                # fail_open：直接放行，不再进 async with 的等待逻辑
                yield
                return
            if int(ok) == 1:
                acquired = True
                break
            await asyncio.sleep(0.1)

        if not acquired:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": 50303,
                    "message": "AI 服务繁忙，请稍后重试",
                    "detail": None,
                },
            )

        yield
    finally:
        if acquired:
            try:
                await redis.eval(_RELEASE_LUA, 1, key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("L2 释放槽位失败（靠 TTL 兜底）: %s", exc)
