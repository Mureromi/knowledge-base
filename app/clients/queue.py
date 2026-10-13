"""ARQ 任务队列连接池（web 进程侧）。

★ 只有 web 进程用它 enqueue；worker 进程通过 `arq app.worker.WorkerSettings`
  自己连同一个 Redis 消费，两边不共享内存，只共享 Redis + 数据库（§3.1）。

沿用阶段 1 缓存客户端的做法：全局单例，lifespan 里 startup/shutdown，
测试通过 dependency_overrides 注入假池（§9.1），不真连 Redis。
"""

import logging

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from app.core.config import settings

logger = logging.getLogger(__name__)

_pool: ArqRedis | None = None


async def startup_queue() -> ArqRedis | None:
    """在 lifespan 里调用。连不上 Redis 时记日志并返回 None（投递会降级为不同步）。"""
    global _pool
    try:
        _pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
        logger.info("ARQ 队列已连接：%s", settings.redis_url)
    except Exception as exc:  # noqa: BLE001 - Redis 不可用不应阻止 web 启动
        logger.error("ARQ 队列连接失败，异步上传将不可用： %s", exc)
        _pool = None
    return _pool


async def shutdown_queue() -> None:
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None


def get_arq() -> ArqRedis:
    """FastAPI 依赖；测试里用 dependency_overrides 换成 FakeArqPool。"""
    if _pool is None:
        raise RuntimeError("ARQ 队列未初始化：请在 lifespan 中 await startup_queue()")
    return _pool
