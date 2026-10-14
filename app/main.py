"""FastAPI 入口。

lifespan 负责：
- init_models：阶段 0 用 create_all；阶段 1 换 alembic
- llm_client.startup / shutdown：单例 HTTP 客户端的生命周期（§6.4）
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.clients.cache import shutdown_cache, startup_cache
from app.clients.embedding import embedding_client
from app.clients.llm import llm_client
from app.clients.queue import shutdown_queue, startup_queue
from app.clients.rerank import rerank_client
from app.clients import tracing
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import log_startup_banner, setup_logging
from app.db.session import dispose_engine, init_models

# ★ 模块顶层就配一次日志：uvicorn 在 import app **之前** 已经打了几行
#   （"Started server process"、"Waiting for application startup"），
#   只在 lifespan 里配的话那几行是裸格式。这里先配，lifespan 里再配一次
#   （幂等，不叠加 handler）以兜住 fork/reload 场景。
setup_logging(force=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ★ 日志配置必须在这里（而不是模块顶层）：uvicorn 会在导入 app 之后才配置它自己的
    #   logging，顶层配置会被它覆盖掉，导致结构化日志静默失效。
    setup_logging(force=True)
    log_startup_banner("web")

    await init_models()
    await llm_client.startup()
    await startup_cache()
    await startup_queue()
    # ★ 阶段 4：预加载本地向量/精排模型（首次含下载较慢；之后命中 .model_cache）。
    #   web 进程两样都要：检索时 query 向量化在关键路径上（跨进程 IPC 得不偿失），
    #   rerank 也在 web 进程的 search() 里跑。startup() 是同步 CPU 活，在开始接待前跑不阻塞服务。
    #   rerank_client.startup() 自带容错（加载失败仅记日志，检索时降级为 RRF 顺序）。
    if settings.embedding_enabled:
        embedding_client.startup()
    if settings.rerank_enabled:
        rerank_client.startup()
    # ★ 阶段 6：可观测。未启用/缺 key 时内部自动降级为 no-op，不影响启动（§3.2）。
    tracing.startup()
    yield
    # ★ tracing.shutdown() 必须调：Langfuse 异步批量上报，不 flush 会丢最后几次 trace（§3.3）。
    tracing.shutdown()
    await llm_client.shutdown()
    await shutdown_cache()
    await shutdown_queue()
    await dispose_engine()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
register_exception_handlers(app)
app.include_router(api_router, prefix="/api/v1")


@app.get("/healthz", tags=["ops"])
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
