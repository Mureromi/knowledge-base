"""多级限流测试（阶段 1 §6 + §10.2）。

用 fakeredis[lua]，所有 Lua 脚本能原样执行。
"""

import asyncio

import pytest
from fastapi import HTTPException

from app.clients.cache import MemoryCache, RedisCache, seconds_until_midnight
from app.core.config import settings
from app.services.ratelimit import (
    _ACQUIRE_LUA,
    _RELEASE_LUA,
    check_daily_quota,
    check_rate_limit,
    distributed_llm_slot,
)

# ---- seconds_until_midnight ----

def test_seconds_until_midnight_is_positive_and_le_86400():
    s = seconds_until_midnight()
    assert 0 < s <= 86400


# ---- L1 滑动窗口 ----

@pytest.mark.asyncio
async def test_l1_rejects_when_over_limit(fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 3)
    monkeypatch.setattr(settings, "rate_limit_window_seconds", 60)
    cache = RedisCache(fake_redis)

    # 前 3 次应通过
    for _ in range(3):
        await check_rate_limit(cache, owner_id=101)

    # 第 4 次应 429
    with pytest.raises(HTTPException) as exc:
        await check_rate_limit(cache, owner_id=101)
    assert exc.value.status_code == 429
    assert exc.value.detail["code"] == 42901
    assert "Retry-After" in (exc.value.headers or {})


@pytest.mark.asyncio
async def test_l1_rejected_request_does_not_consume_slot(fake_redis, monkeypatch):
    """§6.3 第 2 点：被拒的请求要撤回，不能占额度（否则用户重试会无限延长限流）。"""
    monkeypatch.setattr(settings, "rate_limit_per_minute", 2)
    monkeypatch.setattr(settings, "rate_limit_window_seconds", 60)
    cache = RedisCache(fake_redis)

    await check_rate_limit(cache, owner_id=201)
    await check_rate_limit(cache, owner_id=201)
    # 之后 5 次都超限，但都不该占用额度
    for _ in range(5):
        with pytest.raises(HTTPException):
            await check_rate_limit(cache, owner_id=201)

    # 检查底层 ZSET：只应有 2 条记录（Lua 里做了拒绝时的 ZREM 撤回）
    key = f"{settings.cache_key_prefix}:v1:rl:user:201"
    card = await fake_redis.zcard(key)
    assert card == 2, f"被拒请求不应占额度，ZSET 实际长度 {card}"


@pytest.mark.asyncio
async def test_l1_isolates_users(fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 1)
    cache = RedisCache(fake_redis)

    await check_rate_limit(cache, owner_id=301)   # alice 用完额度
    with pytest.raises(HTTPException):
        await check_rate_limit(cache, owner_id=301)
    # bob 独立计数
    await check_rate_limit(cache, owner_id=302)


# ---- L3 每日配额 ----

@pytest.mark.asyncio
async def test_l3_quota_exhausted_returns_42902(fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "daily_llm_quota", 2)
    cache = RedisCache(fake_redis)

    await check_daily_quota(cache, owner_id=401)
    await check_daily_quota(cache, owner_id=401)
    with pytest.raises(HTTPException) as exc:
        await check_daily_quota(cache, owner_id=401)
    assert exc.value.status_code == 429
    assert exc.value.detail["code"] == 42902
    assert "额度已用完" in exc.value.detail["message"]


@pytest.mark.asyncio
async def test_l3_writes_key_with_midnight_ttl(fake_redis, monkeypatch):
    """§4.5：TTL 必须到次日零点，而不是固定 86400（否则窗口漂移）。"""
    monkeypatch.setattr(settings, "daily_llm_quota", 5)
    cache = RedisCache(fake_redis)

    await check_daily_quota(cache, owner_id=402)
    from datetime import datetime

    from app.clients.cache import CST

    today = datetime.now(CST).strftime("%Y%m%d")
    key = f"{settings.cache_key_prefix}:v1:quota:user:402:{today}"
    ttl = await fake_redis.ttl(key)
    # 与 seconds_until_midnight 相差不超过 5 秒
    expected = seconds_until_midnight()
    assert abs(ttl - expected) <= 5, f"TTL 应接近次日零点，实际 {ttl}s vs 预期 {expected}s"


# ---- L2 并发槽位 ----

@pytest.mark.asyncio
async def test_l2_blocks_when_over_capacity(fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "llm_max_concurrency", 1)
    cache = RedisCache(fake_redis)

    # 手动抢一个槽位
    key = f"{settings.cache_key_prefix}:v1:conc:llm"
    ok = await fake_redis.eval(_ACQUIRE_LUA, 1, key, 1, 30)
    assert int(ok) == 1

    # 第二个进入 slot → 应超时 503（这里缩短 deadline 以免测试慢）
    async def _short_wait_slot():
        # 直接测试 acquire 分支返回 0 时的行为
        return await fake_redis.eval(_ACQUIRE_LUA, 1, key, 1, 30)

    ok2 = await _short_wait_slot()
    assert int(ok2) == 0, "并发上限 1 时，第二个应抢槽失败"

    # 释放后又能拿到
    await fake_redis.eval(_RELEASE_LUA, 1, key)
    ok3 = await fake_redis.eval(_ACQUIRE_LUA, 1, key, 1, 30)
    assert int(ok3) == 1


@pytest.mark.asyncio
async def test_l2_release_prevents_negative_count(fake_redis):
    """§6.3 第 3 点：进程崩溃后 key 会因 TTL 过期，随后 finally 里的 DECR 不能变负数。"""
    key = f"{settings.cache_key_prefix}:v1:conc:llm"
    # 从未 acquire 过，直接 release → Lua 应返回 0 而不是把值变 -1
    result = await fake_redis.eval(_RELEASE_LUA, 1, key)
    assert int(result) == 0
    current = await fake_redis.get(key)
    # key 不应存在或值为 0，绝不能为负
    assert current is None or int(current) == 0


@pytest.mark.asyncio
async def test_l2_context_manager_releases_on_exit(fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "llm_max_concurrency", 2)
    cache = RedisCache(fake_redis)
    key = f"{settings.cache_key_prefix}:v1:conc:llm"

    async with distributed_llm_slot(cache):
        assert int(await fake_redis.get(key)) == 1
    assert int(await fake_redis.get(key)) == 0


@pytest.mark.asyncio
async def test_l2_releases_on_exception(fake_redis, monkeypatch):
    """即使 inside async with 抛异常，finally 也要释放槽位，不能泄漏。"""
    monkeypatch.setattr(settings, "llm_max_concurrency", 2)
    cache = RedisCache(fake_redis)
    key = f"{settings.cache_key_prefix}:v1:conc:llm"

    with pytest.raises(RuntimeError):
        async with distributed_llm_slot(cache):
            raise RuntimeError("boom")
    assert int(await fake_redis.get(key)) == 0


# ---- 失败策略 ----

@pytest.mark.asyncio
async def test_redis_failure_fail_open_allows_request(fake_redis, monkeypatch):
    """Redis 全线失败 + fail_open=true → 请求放行。"""
    monkeypatch.setattr(settings, "ratelimit_fail_open", True)

    async def boom(*_a, **_kw):
        raise RuntimeError("redis down")

    fake_redis.eval = boom
    fake_redis.incr = boom
    cache = RedisCache(fake_redis)

    # L1 应放行
    await check_rate_limit(cache, owner_id=501)
    # L3 应放行
    await check_daily_quota(cache, owner_id=501)


@pytest.mark.asyncio
async def test_redis_failure_fail_closed_returns_503(fake_redis, monkeypatch):
    """Redis 全线失败 + fail_open=false → 返回 503（成本优先模式）。"""
    monkeypatch.setattr(settings, "ratelimit_fail_open", False)

    async def boom(*_a, **_kw):
        raise RuntimeError("redis down")

    fake_redis.eval = boom
    fake_redis.incr = boom
    cache = RedisCache(fake_redis)

    with pytest.raises(HTTPException) as exc:
        await check_rate_limit(cache, owner_id=601)
    assert exc.value.status_code == 503
    assert exc.value.detail["code"] == 50302


@pytest.mark.asyncio
async def test_memory_cache_skips_ratelimits(monkeypatch):
    """降级为 MemoryCache 时（Redis 未启用）→ L1/L3 直接放行，不阻断问答。"""
    monkeypatch.setattr(settings, "rate_limit_per_minute", 0)  # 若真去查肯定超限
    cache = MemoryCache()

    # 应无异常
    await check_rate_limit(cache, owner_id=1)
    await check_daily_quota(cache, owner_id=1)


# ---- 集成：并发下 L2 不会突破上限 ----

@pytest.mark.asyncio
async def test_l2_enforces_global_limit_under_concurrency(fake_redis, monkeypatch):
    """同时打 5 个 slot，上限设 2 → 至少 3 个应被延迟。"""
    monkeypatch.setattr(settings, "llm_max_concurrency", 2)
    cache = RedisCache(fake_redis)

    active = {"now": 0, "peak": 0}

    async def worker():
        async with distributed_llm_slot(cache):
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
            await asyncio.sleep(0.05)
            active["now"] -= 1

    await asyncio.gather(*[worker() for _ in range(5)])
    assert active["peak"] <= 2, f"并发不应超过 2，实际峰值 {active['peak']}"
