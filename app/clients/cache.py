"""Redis 客户端与缓存抽象。

设计要点（对应阶段 1 文档 §5.1）：
1. 所有 Redis 操作都包 try/except —— Redis 故障绝不能导致问答接口 500。
2. 提供 CacheBackend 协议，测试可注入 fakeredis 实现。
3. 缓存值存 JSON 字符串（不用 Hash），读写都是整体操作。
4. 暴露底层 aioredis 客户端给限流模块（Lua / ZSET / INCR），避免另建全局单例。
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from app.core.config import settings

logger = logging.getLogger(__name__)

CST = ZoneInfo("Asia/Shanghai")


def seconds_until_midnight(now: datetime | None = None) -> int:
    """到次日零点的秒数。用于每日配额键的 TTL（不能用固定 86400，会漂移）。"""
    now = now or datetime.now(CST)
    tomorrow = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return max(int((tomorrow - now).total_seconds()), 1)


class CacheBackend(Protocol):
    """缓存后端协议：RedisCache 与 MemoryCache 都满足它。"""

    async def get_json(self, key: str) -> dict[str, Any] | None: ...
    async def set_json(self, key: str, value: dict[str, Any], ttl: int) -> None: ...
    async def delete(self, key: str) -> None: ...


class RedisCache:
    """生产实现：包装 redis-py 异步客户端。

    `.redis` 属性给限流模块用（ZSET / Lua / INCR 等原生命令，超出 Cache 协议范围）。
    """

    def __init__(self, client: "aioredis.Redis") -> None:
        self._redis = client

    @property
    def redis(self) -> "aioredis.Redis":
        return self._redis

    async def get_json(self, key: str) -> dict[str, Any] | None:
        try:
            raw = await self._redis.get(key)
        except RedisError as exc:
            logger.warning("缓存读取失败（降级为未命中）key=%s err=%s", key, exc)
            return None
        if raw is None:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("缓存内容损坏，删除该键 key=%s", key)
            await self.delete(key)
            return None
        return data if isinstance(data, dict) else None

    async def set_json(self, key: str, value: dict[str, Any], ttl: int) -> None:
        try:
            await self._redis.set(key, json.dumps(value, ensure_ascii=False), ex=ttl)
        except RedisError as exc:
            # 写缓存失败不影响本次回答，只记日志
            logger.warning("缓存写入失败 key=%s err=%s", key, exc)

    async def delete(self, key: str) -> None:
        try:
            await self._redis.delete(key)
        except RedisError as exc:
            logger.warning("缓存删除失败 key=%s err=%s", key, exc)


class MemoryCache:
    """纯内存实现：只在 redis_enabled=false、连不上 Redis、或单测降级时使用。

    ⚠️ 不是生产方案：多 worker 下每个进程一份，命中率被 worker 数量摊薄。
    无 `.redis` 属性 → 限流模块会识别为"未启用"直接放行（fail-open）。
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, str]] = {}

    async def get_json(self, key: str) -> dict[str, Any] | None:
        item = self._store.get(key)
        if item is None:
            return None
        expire_at, raw = item
        if expire_at < time.time():
            self._store.pop(key, None)
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            self._store.pop(key, None)
            return None

    async def set_json(self, key: str, value: dict[str, Any], ttl: int) -> None:
        self._store[key] = (time.time() + ttl, json.dumps(value, ensure_ascii=False))

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)


# ---- 全局状态：仅在 lifespan 里初始化，测试通过 dependency_overrides 注入 ----

_redis_client: "aioredis.Redis | None" = None
_cache: CacheBackend | None = None


async def startup_cache() -> CacheBackend:
    """在 lifespan 里调用。Redis 连不上时自动降级为内存缓存，服务照常可用。"""
    global _redis_client, _cache

    if not settings.redis_enabled:
        logger.warning("REDIS_ENABLED=false，使用进程内内存缓存（生产不可用）")
        _cache = MemoryCache()
        return _cache

    _redis_client = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_timeout=settings.redis_socket_timeout,   # 必须设！否则 Redis 卡住会拖垮接口
        socket_connect_timeout=settings.redis_socket_timeout,
        health_check_interval=30,
    )
    try:
        await _redis_client.ping()
        logger.info("Redis 已连接: %s", settings.redis_url)
        _cache = RedisCache(_redis_client)
    except RedisError as exc:
        logger.error("Redis 连接失败，降级为内存缓存: %s", exc)
        await _redis_client.aclose()
        _redis_client = None
        _cache = MemoryCache()

    return _cache


async def shutdown_cache() -> None:
    global _redis_client, _cache
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
    _cache = None


def get_cache() -> CacheBackend:
    """FastAPI 依赖；测试里用 dependency_overrides 换成 RedisCache(fake_redis)。"""
    if _cache is None:
        raise RuntimeError("缓存未初始化：请在 lifespan 中 await startup_cache()")
    return _cache
