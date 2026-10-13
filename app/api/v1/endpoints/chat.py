"""问答端点（阶段 2 §6.1 路由总览）：

| 方法   | 路径                        | 形态      |
| ------ | --------------------------- | --------- |
| POST   | /api/v1/chat                | SSE 流式  |
| POST   | /api/v1/chat/sync           | JSON      |
| GET    | /api/v1/chat/sessions       | 会话列表  |
| GET    | /api/v1/chat/sessions/{id}  | 会话详情  |
| PATCH  | /api/v1/chat/sessions/{id}  | 重命名    |
| DELETE | /api/v1/chat/sessions/{id}  | 删除      |

★ 流式端点的两个关键点：
1. L1 限流与会话越权检查必须在返回 StreamingResponse **之前**做完——
   流一旦开始，HTTP 状态码已发出，429/404 都改不了了（§5.6）。
2. 生成器内部用 StreamSessionFactoryDep 自开 session（§6.2），
   请求级 db 在生成器运行时已经关闭。
3. 新会话（session_id=None）由生成器在自有 session 里创建+commit——
   请求级 db 预建的行只 flush 未 commit，多连接引擎下生成器看不到（冒烟踩坑）。
"""

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.deps import (
    CacheDep,
    CurrentUser,
    DBSession,
    EmbedderDep,
    LLMDep,
    RerankerDep,
    StreamSessionFactoryDep,
)
from app.core.config import settings
from app.models.chat_message import ChatMessage
from app.models.chat_session import ChatSession
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    Citation,
    SessionDetail,
    SessionList,
    SessionOut,
    SessionRename,
    Usage,
)
from app.services import qa as qa_service
from app.services import session as session_service
from app.services.ratelimit import check_rate_limit

router = APIRouter(prefix="/chat", tags=["chat"])


def sse_event(event: str, data: dict) -> str:
    """SSE 单条消息。结尾必须两个 \\n；data 里不能有裸换行，否则被解析成多条。

    json.dumps 默认把换行转义成 \\n，所以安全——但手动拼字符串就会踩坑。
    """
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


async def _chat_event_source(
    session_factory,
    llm,
    cache,
    *,
    owner_id: int,
    question: str,
    session_id: int,
    top_k: int | None,
    embedder,
    reranker,
) -> AsyncIterator[str]:
    try:
        async for event, data in qa_service.run_answer_stream(
            session_factory,
            llm,
            cache,
            owner_id=owner_id,
            question=question,
            session_id=session_id,
            top_k=top_k,
            embedder=embedder,
            reranker=reranker,
        ):
            yield sse_event(event, data)
    except HTTPException as exc:
        # 典型：L3 每日配额在"未命中后才扣"（阶段 1 的设计），流已开始就无法 429，
        # 把原始错误码透传成 error 事件，前端能区分"额度用完"和"服务挂了"。
        detail = exc.detail if isinstance(exc.detail, dict) else {
            "code": exc.status_code,
            "message": str(exc.detail),
        }
        yield sse_event("error", detail)
        yield sse_event("done", {"usage": None, "incomplete": True})
    except Exception:                      # noqa: BLE001
        # 走到这里说明是未预期异常（LLMError 已在 qa 内部转成 error 事件）。
        # 状态码改不了，只能发 error 事件让前端至少知道"断了"。
        yield sse_event(
            "error", {"code": 50001, "message": "服务异常，请稍后重试"}
        )
        yield sse_event("done", {"usage": None, "incomplete": True})


@router.post("", summary="问答（SSE 流式）")
async def chat_stream_endpoint(
    payload: ChatRequest,
    current_user: CurrentUser,
    db: DBSession,
    llm: LLMDep,
    cache: CacheDep,
    session_factory: StreamSessionFactoryDep,
    embedder: EmbedderDep,
    reranker: RerankerDep,
) -> StreamingResponse:
    # L1 频率：最早拦，最省资源（连检索都不做）
    await check_rate_limit(cache, current_user.id)

    # ★ 流开始前的会话预检——只查不建：
    #   传了 session_id 才做越权检查（404 还能以正常状态码返回）；
    #   session_id=None 的新会话**不在这里创建**——请求级 db 只 flush，
    #   commit 时机与生成器运行顺序没有保证（get_db 的 teardown 在新版
    #   FastAPI 里早于响应体发送，但跨连接能否可见取决于引擎；测试用
    #   StaticPool 单连接恰好掩盖了这一点，真实 uvicorn 多连接下生成器
    #   SELECT 不到新会话行 → 40402）。新会话交由生成器在自有 session
    #   里 get_or_create + 最终 commit，创建与读取同连接，无时序问题。
    if payload.session_id is not None:
        await session_service.get_or_create_session(
            db,
            owner_id=current_user.id,
            session_id=payload.session_id,
            first_question=payload.question,
        )

    return StreamingResponse(
        _chat_event_source(
            session_factory,
            llm,
            cache,
            owner_id=current_user.id,
            question=payload.question,
            session_id=payload.session_id,
            top_k=payload.top_k,
            embedder=embedder,
            reranker=reranker,
        ),
        media_type="text/event-stream",
        headers={
            # 关掉所有缓冲，否则流会被攒起来一次性发出（代理/nginx 同理）
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/sync", response_model=ChatResponse, summary="问答（非流式 JSON）")
async def chat_sync(
    payload: ChatRequest,
    current_user: CurrentUser,
    db: DBSession,
    llm: LLMDep,
    cache: CacheDep,
    embedder: EmbedderDep,
    reranker: RerankerDep,
) -> ChatResponse:
    """脚本 / 自动化 / 未来 Java 端用。业务逻辑与流式共用 qa.run_answer。"""
    await check_rate_limit(cache, current_user.id)

    result = await qa_service.run_answer(
        db,
        llm,
        cache,
        owner_id=current_user.id,
        question=payload.question,
        session_id=payload.session_id,
        top_k=payload.top_k,
        embedder=embedder,
        reranker=reranker,
    )

    return ChatResponse(
        answer=result.answer,
        citations=[
            Citation(
                document_id=c.document_id,
                document_title=c.document_title,
                chunk_id=c.chunk_id,
                seq=c.seq,
                score=c.score,
                keyword_rank=c.keyword_rank,
                vector_rank=c.vector_rank,
            )
            for c in result.citations
        ],
        usage=Usage(**result.usage),
        retrieval_mode=result.usage.get("retrieval_mode", "keyword"),
        session_id=result.session_id,
    )


# ---------------------------------------------------------------- 会话 CRUD


async def _load_owned_session(
    db: AsyncSession, session_id: int, owner_id: int
) -> ChatSession:
    """越权与不存在统一 404（同 services/session.py 的防探测思路）。"""
    stmt = select(ChatSession).where(
        ChatSession.id == session_id, ChatSession.owner_id == owner_id
    )
    session = (await db.scalars(stmt)).one_or_none()
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": 40402, "message": "会话不存在", "detail": None},
        )
    return session


@router.get("/sessions", response_model=SessionList, summary="会话列表")
async def list_sessions(
    current_user: CurrentUser,
    db: DBSession,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> SessionList:
    """按 last_active_at 倒序（§3.3：不是 created_at，否则刚聊过的排不到前面）。"""
    total = (
        await db.scalar(
            select(func.count(ChatSession.id)).where(
                ChatSession.owner_id == current_user.id
            )
        )
        or 0
    )
    stmt = (
        select(ChatSession)
        .where(ChatSession.owner_id == current_user.id)
        .order_by(ChatSession.last_active_at.desc(), ChatSession.id.desc())
        .limit(limit)
        .offset(offset)
    )
    items = list((await db.scalars(stmt)).all())
    return SessionList(total=total, items=[SessionOut.model_validate(s) for s in items])


@router.get("/sessions/{session_id}", response_model=SessionDetail, summary="会话详情")
async def get_session_detail(
    session_id: int, current_user: CurrentUser, db: DBSession
) -> SessionDetail:
    session = await _load_owned_session(db, session_id, current_user.id)
    msgs = (
        await db.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == session.id)
            .order_by(ChatMessage.id)
        )
    ).all()
    return SessionDetail(
        session=SessionOut.model_validate(session),
        messages=list(msgs),
    )


@router.patch("/sessions/{session_id}", response_model=SessionOut, summary="重命名会话")
async def rename_session(
    session_id: int,
    payload: SessionRename,
    current_user: CurrentUser,
    db: DBSession,
) -> SessionOut:
    session = await _load_owned_session(db, session_id, current_user.id)
    session.title = payload.title.strip()[: settings.session_title_max_chars]
    await db.flush()
    await db.refresh(session)
    return SessionOut.model_validate(session)


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除会话（级联删消息）",
)
async def delete_session(
    session_id: int, current_user: CurrentUser, db: DBSession
) -> Response:
    session = await _load_owned_session(db, session_id, current_user.id)
    # 显式删消息而不是依赖 ORM cascade：异步下删除父行会触发 messages 懒加载
    # （MissingGreenlet），直接两条 DELETE 更稳，也不依赖 SQLite 的 FK pragma。
    await db.execute(
        delete(ChatMessage).where(ChatMessage.session_id == session.id)
    )
    await db.delete(session)
    # ★ chat_logs.session_id 靠 FK ON DELETE SET NULL 保命（§2.3）：
    #   会话可以删，成本审计数据必须留下。SQLite 默认不强制 FK，PostgreSQL 会置空。
    return Response(status_code=status.HTTP_204_NO_CONTENT)
