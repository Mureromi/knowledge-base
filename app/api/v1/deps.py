"""API 依赖：当前用户 + 数据库 session + LLM + 缓存 + 流式会话工厂。"""

from typing import Annotated, Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.clients.cache import CacheBackend, get_cache
from app.clients.embedding import EmbeddingClient, get_embedding
from app.clients.llm import LLMClient, get_llm
from app.clients.queue import get_arq
from app.clients.rerank import RerankClient, get_reranker
from app.core.security import decode_access_token
from app.crud import user as user_crud
from app.db.session import SessionLocal, get_db
from app.models.user import User

# OAuth2 密码流的 bearerToken 提取端点：/api/v1/auth/token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    try:
        payload = decode_access_token(token)
        user_id = int(payload["sub"])
    except (KeyError, ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 无效",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    user = await user_crud.get_by_id(db, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
DBSession = Annotated[AsyncSession, Depends(get_db)]
LLMDep = Annotated[LLMClient, Depends(get_llm)]
# 阶段 1：缓存依赖（给 qa.answer 与 ratelimit 共用）
CacheDep = Annotated[CacheBackend, Depends(get_cache)]
# 阶段 3：ARQ 队列依赖（上传端点投递任务用）
ArqDep = Annotated[object, Depends(get_arq)]
# 阶段 4：向量 / 精排依赖（检索层注入；测试用 dependency_overrides 换假实现）
EmbedderDep = Annotated[EmbeddingClient, Depends(get_embedding)]
RerankerDep = Annotated[RerankClient, Depends(get_reranker)]


def get_stream_session_factory() -> "async_sessionmaker[AsyncSession]":
    """★ 阶段 2：流式生成器专用会话工厂（§6.2）。

    StreamingResponse 的生成器运行时，请求级 get_db 的 yield 已经退出，
    拿到的 session 是关着的——落库会静默失败。生成器必须自开 session、自行 commit。
    单独做成依赖是为了测试：conftest 里 override 成内存 SQLite 的 session_factory。
    """
    return SessionLocal


StreamSessionFactoryDep = Annotated[
    Callable[[], AsyncSession], Depends(get_stream_session_factory)
]
