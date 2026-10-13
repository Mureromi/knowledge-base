"""RedisCache / MemoryCache 的行为测试（阶段 1 §10.2）。

核心：Redis 故障必须降级为未命中，绝不能冒泡打挂问答接口。
"""

import pytest
from redis.exceptions import RedisError

from app.clients.cache import MemoryCache, RedisCache


@pytest.mark.asyncio
async def test_redis_cache_roundtrip(fake_redis):
    cache = RedisCache(fake_redis)
    await cache.set_json("k", {"answer": "42"}, ttl=60)
    got = await cache.get_json("k")
    assert got == {"answer": "42"}


@pytest.mark.asyncio
async def test_redis_cache_ttl_expiry(fake_redis):
    """ttl=1 秒后应取不到（fakeredis 支持 TTL 语义）。"""
    import asyncio

    cache = RedisCache(fake_redis)
    await cache.set_json("k", {"v": 1}, ttl=1)
    assert await cache.get_json("k") is not None
    await asyncio.sleep(1.1)
    assert await cache.get_json("k") is None


@pytest.mark.asyncio
async def test_redis_cache_returns_none_for_missing_key(fake_redis):
    cache = RedisCache(fake_redis)
    assert await cache.get_json("nope") is None


@pytest.mark.asyncio
async def test_redis_cache_handles_corrupted_value(fake_redis):
    """缓存内容不是合法 JSON → 返回 None 且不抛异常，同时把坏键删掉。"""
    await fake_redis.set("bad", "not-a-json{{")
    cache = RedisCache(fake_redis)
    assert await cache.get_json("bad") is None
    # 应已被清理
    assert await fake_redis.get("bad") is None


@pytest.mark.asyncio
async def test_redis_cache_swallows_read_errors(fake_redis, monkeypatch):
    """Redis 抛 RedisError → get_json 返回 None（降级为未命中），绝不冒泡打挂接口。"""
    cache = RedisCache(fake_redis)

    async def boom(_key):
        raise RedisError("redis down")

    monkeypatch.setattr(fake_redis, "get", boom)
    assert await cache.get_json("k") is None


@pytest.mark.asyncio
async def test_redis_cache_swallows_write_errors(fake_redis, monkeypatch):
    cache = RedisCache(fake_redis)

    async def boom(*_a, **_kw):
        raise RedisError("redis down")

    monkeypatch.setattr(fake_redis, "set", boom)
    # 不应抛
    await cache.set_json("k", {"v": 1}, ttl=1)


# ---- MemoryCache ----

@pytest.mark.asyncio
async def test_memory_cache_roundtrip():
    cache = MemoryCache()
    await cache.set_json("k", {"answer": "hi"}, ttl=60)
    assert (await cache.get_json("k")) == {"answer": "hi"}


@pytest.mark.asyncio
async def test_memory_cache_expiry():
    cache = MemoryCache()
    await cache.set_json("k", {"v": 1}, ttl=0)  # 立即过期
    assert await cache.get_json("k") is None
