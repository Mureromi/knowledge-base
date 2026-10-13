"""FastAPI 入口。

lifespan 负责：
- init_models：阶段 0 用 create_all；阶段 1 换 alembic
- llm_client.startup / shutdown：单例 HTTP 客户端的生命周期（§6.4）
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.clients.cache import shutdown_cache, startup_cache
from app.clients.llm import llm_client
from app.clients.queue import shutdown_queue, startup_queue
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.db.session import dispose_engine, init_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_models()
    await llm_client.startup()
    await startup_cache()
    await startup_queue()
    yield
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
