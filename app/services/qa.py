"""拼 prompt → 调 LLM → 组装引用 → 会话记忆。阶段 2：多轮历史 + SSE 流式。

对应阶段 2 文档 §3 / §5 / §6.3。关键决策：
- 缓存键 = owner_id + 归一化问题 + **窗口历史指纹** + 检索指纹（§4）。
- ★ 写入顺序（§3.2）：先读窗口、算键、查缓存，**最后**才写入本轮 user/assistant 消息。
  顺序写反不会报错，只会让缓存静默永不命中——test_multiturn_cache 专防这一点。
- 流式生成器内不能用请求级 db（依赖 yield 在响应头发出前已退出），
  必须用独立 session_factory 自开 session 自行 commit（§6.2，阶段 2 最容易踩的坑）。
- 只存 chunk_ids 不存切片正文，命中时 _hydrate_citations 从数据库重取。
"""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.clients.cache import CacheBackend
from app.clients.llm import LLMClient, LLMError, LLMResult
from app.core.config import settings
from app.models.chat_log import ChatLog
from app.models.chat_session import ChatSession
from app.models.chunk import Chunk
from app.models.document import Document
from app.services.cache_key import answer_cache_key
from app.services.ratelimit import (
    check_daily_quota,
    distributed_llm_slot,
)
from app.services.retrieval import RetrievedChunk, search
from app.services.session import (
    append_turn,
    get_or_create_session,
    load_window_messages,
)

logger = logging.getLogger(__name__)

# ★ 必须是固定文本（无时间戳、无随机值），否则永远吃不到缓存，成本差 50 倍
SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。请遵守以下规则：

1. 只根据【参考资料】回答，不要使用你自己的知识补充事实。
2. 如果参考资料不足以回答，直接回答"根据现有资料无法回答"，不要编造。
3. 回答简洁准确，不要复述问题，不要输出与问题无关的内容。
4. 在引用具体内容时，用 [1] [2] 的形式标注来源编号。
"""

NO_RESULT_ANSWER = (
    "知识库中没有找到与该问题相关的内容。请先上传相关文档，或换一种问法。"
)


@dataclass
class AnswerResult:
    """非流式问答的完整结果。"""

    answer: str
    citations: list[RetrievedChunk]
    usage: dict[str, Any]
    session_id: int


def build_context(chunks: list[RetrievedChunk], max_chars: int | None = None) -> str:
    max_chars = max_chars or settings.max_context_chars
    parts: list[str] = []
    used = 0
    for i, c in enumerate(chunks, start=1):
        block = f"[{i}] 来源：《{c.document_title}》第 {c.seq + 1} 段\n{c.content}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return f"""【参考资料】
{build_context(chunks)}

【问题】
{question}

请根据参考资料回答，并标注引用编号。"""


def build_messages(
    question: str, chunks: list[RetrievedChunk], window: list[dict[str, str]]
) -> list[dict[str, str]]:
    """system 恒在最前（缓存友好，阶段 0 §7.3），然后窗口历史、最后本轮问题。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *window,
        {"role": "user", "content": build_user_prompt(question, chunks)},
    ]


async def _hydrate_citations(
    db: AsyncSession, chunk_ids: list[int]
) -> list[RetrievedChunk] | None:
    """按 id 重建引用；任一 id 取不到（文档已删）→ 返回 None，调用方视为未命中。

    绝不能"尽力而为"地返回残缺引用——那会让用户看到一个指向不存在文档的答案。
    顺手 join 一次 Document 拿标题（不把标题存进缓存，文档改名缓存自然失效）。
    """
    if not chunk_ids:
        return None
    stmt = (
        select(Chunk, Document.title)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.id.in_(chunk_ids))
    )
    rows = (await db.execute(stmt)).all()
    got_ids = {c.id for c, _title in rows}
    if got_ids != set(chunk_ids):
        return None
    by_id = {c.id: (c, title) for c, title in rows}
    result: list[RetrievedChunk] = []
    for cid in chunk_ids:
        c, title = by_id[cid]
        result.append(
            RetrievedChunk(
                chunk_id=c.id,
                document_id=c.document_id,
                document_title=title,
                seq=c.seq,
                content=c.content,
                score=0.0,
            )
        )
    return result


def _no_result_usage() -> dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_yuan": 0.0,
        "latency_ms": 0,
        "cache_hit": False,
        "retrieval_mode": "keyword",
    }


async def _lookup_cache(
    db: AsyncSession,
    cache: CacheBackend,
    *,
    owner_id: int,
    question: str,
    chunks: list[RetrievedChunk],
    window: list[dict[str, str]],
) -> tuple[str, dict[str, Any] | None, list[RetrievedChunk] | None]:
    """算键 → 查缓存 → 重建引用。返回 (cache_key, 命中值, 命中引用)。

    引用失效时视为未命中（落到生成路径，不删旧键，靠 TTL 自然过期）。
    """
    cache_key = answer_cache_key(
        owner_id=owner_id, question=question, chunks=chunks, window_messages=window
    )
    if not settings.cache_enabled:
        return cache_key, None, None
    cached = await cache.get_json(cache_key)
    if cached is None:
        return cache_key, None, None
    citations = await _hydrate_citations(db, cached.get("chunk_ids", []))
    if citations is None:
        logger.info('{"event":"cache_stale","user":%s}', owner_id)
    return cache_key, cached, citations


def _hit_usage(cached: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_yuan": 0.0,
        "latency_ms": 0,
        "cache_hit": True,
        "saved_cost_yuan": float(cached.get("cost_yuan", 0.0) or 0.0),
        "retrieval_mode": "keyword",
    }


async def _persist_hit(
    db: AsyncSession,
    *,
    session: ChatSession,
    owner_id: int,
    question: str,
    cached: dict[str, Any],
    citations: list[RetrievedChunk],
) -> None:
    saved = float(cached.get("cost_yuan", 0.0) or 0.0)
    logger.info(
        '{"event":"chat","user":%s,"session":%s,"cache_hit":true,"saved_cost_yuan":%s}',
        owner_id, session.id, saved,
    )
    db.add(
        ChatLog(
            owner_id=owner_id,
            session_id=session.id,
            question=question,
            answer=cached.get("answer", ""),
            prompt_tokens=0,
            completion_tokens=0,
            cached_tokens=0,
            cost_yuan=0.0,            # 本次真实花费为 0
            saved_cost_yuan=saved,    # 这次因命中省下的钱
            latency_ms=0,
            model=cached.get("model", ""),
            is_off_peak=False,
            cited_chunk_ids=[c.chunk_id for c in citations],
            cache_hit=True,
        )
    )
    await append_turn(
        db,
        session=session,
        question=question,
        answer=cached.get("answer", ""),
        cited_chunk_ids=[c.chunk_id for c in citations],
        cache_hit=True,
    )


async def _persist_miss(
    db: AsyncSession,
    *,
    session: ChatSession,
    owner_id: int,
    question: str,
    result: LLMResult,
    chunks: list[RetrievedChunk],
    incomplete: bool = False,
) -> None:
    # ★ 正文由调用方逐块流式发出，result.content 在流式路径下是空的
    #   （_parse 拿不到 choices），所以落库用调用方累积下来的 content。
    if incomplete:
        # §5.6：部分答案也有价值（且便于排障），照常落库但标注未完成
        answer_text = f"{result.content}\n\n（回答生成中断）"
    else:
        answer_text = result.content
    db.add(
        ChatLog(
            owner_id=owner_id,
            session_id=session.id,
            question=question,
            answer=answer_text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cached_tokens=result.cached_tokens,
            cost_yuan=result.cost_yuan,
            saved_cost_yuan=0.0,
            latency_ms=result.latency_ms,
            model=result.model,
            is_off_peak=not result.peak,
            cited_chunk_ids=[c.chunk_id for c in chunks],
            cache_hit=False,
        )
    )
    if answer_text.strip():
        await append_turn(
            db,
            session=session,
            question=question,
            answer=answer_text,
            cited_chunk_ids=[c.chunk_id for c in chunks],
            cache_hit=False,
        )


# ---------------------------------------------------------------- 非流式（/chat/sync）


async def run_answer(
    db: AsyncSession,
    llm: LLMClient,
    cache: CacheBackend,
    *,
    owner_id: int,
    question: str,
    session_id: int | None,
    top_k: int | None = None,
) -> AnswerResult:
    """非流式完整问答。逻辑与阶段 1 相同，只是加了会话历史与溯源。"""
    session = await get_or_create_session(
        db, owner_id=owner_id, session_id=session_id, first_question=question
    )
    # ★ 顺序关键（§3.2）：先读窗口（此时不含本轮），再算键
    window = await load_window_messages(db, session_id=session.id)

    chunks = await search(db, owner_id=owner_id, question=question, top_k=top_k)
    if not chunks:
        return AnswerResult(NO_RESULT_ANSWER, [], _no_result_usage(), session.id)

    cache_key, cached, citations = await _lookup_cache(
        db,
        cache,
        owner_id=owner_id,
        question=question,
        chunks=chunks,
        window=window,
    )
    if cached is not None and citations is not None:
        await _persist_hit(
            db,
            session=session,
            owner_id=owner_id,
            question=question,
            cached=cached,
            citations=citations,
        )
        return AnswerResult(
            cached.get("answer", ""), citations, _hit_usage(cached), session.id
        )

    # ---------- 未命中：扣 L3 配额 → 分布式并发槽位 → 调 LLM ----------
    await check_daily_quota(cache, owner_id)
    messages = build_messages(question, chunks, window)

    async with distributed_llm_slot(cache):
        result = await llm.chat(
            messages,
            temperature=0.2,
            user_id=f"u{owner_id}",     # 隔离用，不含隐私信息
        )

    # ---------- 写缓存（空答案不缓存）----------
    if settings.cache_enabled and result.content.strip():
        await cache.set_json(
            cache_key,
            {
                "answer": result.content,
                "chunk_ids": [c.chunk_id for c in chunks],
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "cached_tokens": result.cached_tokens,
                "cost_yuan": result.cost_yuan,
                "model": result.model,
            },
            ttl=settings.cache_ttl_seconds,
        )

    await _persist_miss(
        db,
        session=session,
        owner_id=owner_id,
        question=question,
        result=result,
        chunks=chunks,
    )

    logger.info(
        '{"event":"chat","user":%s,"session":%s,"cache_hit":false,"retrieved":%s,'
        '"prompt_tokens":%s,"completion_tokens":%s,"cached_tokens":%s,'
        '"cost_yuan":%s,"latency_ms":%s,"peak":%s}',
        owner_id, session.id, len(chunks),
        result.prompt_tokens, result.completion_tokens, result.cached_tokens,
        result.cost_yuan, result.latency_ms, result.peak,
    )

    return AnswerResult(
        result.content,
        chunks,
        {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "cost_yuan": result.cost_yuan,
            "latency_ms": result.latency_ms,
            "cache_hit": False,
            "retrieval_mode": "keyword",
        },
        session.id,
    )


# ---------------------------------------------------------------- 流式（/chat SSE）


def stream_cached_answer(
    text: str, chunk_size: int | None = None
) -> AsyncIterator[str]:
    """把缓存答案切块推送，模拟流式手感（§5.4）。

    不加 sleep：缓存的意义就是快——快得明显，用户才知道系统在变快。
    """
    chunk_size = chunk_size or settings.stream_cached_chunk_chars

    async def _gen() -> AsyncIterator[str]:
        for i in range(0, len(text), chunk_size):
            yield text[i : i + chunk_size]

    return _gen()


async def run_answer_stream(
    session_factory: async_sessionmaker[AsyncSession],
    llm: LLMClient,
    cache: CacheBackend,
    *,
    owner_id: int,
    question: str,
    session_id: int | None,
    top_k: int | None = None,
) -> AsyncIterator[tuple[str, dict]]:
    """流式问答，产出 (事件名, 数据) 序列，由端点转成 SSE。

    ★ 用独立 session_factory 自开 session：StreamingResponse 生成器运行时，
      请求级 get_db 的 yield 已经退出，拿到的 session 是关着的（§6.2）。
    """
    async with session_factory() as db:
        try:
            session = await get_or_create_session(
                db, owner_id=owner_id, session_id=session_id, first_question=question
            )
            window = await load_window_messages(db, session_id=session.id)

            chunks = await search(
                db, owner_id=owner_id, question=question, top_k=top_k
            )
            if not chunks:
                yield ("meta", {"session_id": session.id, "cache_hit": False})
                yield ("delta", {"text": NO_RESULT_ANSWER})
                yield ("citations", {"citations": []})
                yield ("done", {"usage": _no_result_usage()})
                await db.commit()
                return

            cache_key, cached, citations = await _lookup_cache(
                db,
                cache,
                owner_id=owner_id,
                question=question,
                chunks=chunks,
                window=window,
            )

            # ---------- 命中：假装流式 ----------
            if cached is not None and citations is not None:
                answer_text = cached.get("answer", "")
                yield ("meta", {"session_id": session.id, "cache_hit": True})
                async for piece in stream_cached_answer(answer_text):
                    yield ("delta", {"text": piece})
                yield (
                    "citations",
                    {"citations": [_citation_dict(c) for c in citations]},
                )
                usage = _hit_usage(cached)
                yield ("done", {"usage": usage, "saved_cost_yuan": usage["saved_cost_yuan"]})
                await _persist_hit(
                    db,
                    session=session,
                    owner_id=owner_id,
                    question=question,
                    cached=cached,
                    citations=citations,
                )
                await db.commit()
                return

            # ---------- 未命中：真流式 ----------
            await check_daily_quota(cache, owner_id)
            messages = build_messages(question, chunks, window)

            yield ("meta", {"session_id": session.id, "cache_hit": False})

            parts: list[str] = []
            usage_result: LLMResult | None = None
            try:
                async with distributed_llm_slot(cache):
                    async for sc in llm.chat_stream(
                        messages,
                        temperature=0.2,
                        user_id=f"u{owner_id}",
                    ):
                        if sc.kind == "reasoning":
                            # ★ 思维链单独成事件，绝不混进 delta 正文（§1.4）
                            yield ("reasoning", {"text": sc.text})
                        elif sc.kind == "content":
                            parts.append(sc.text)
                            yield ("delta", {"text": sc.text})
                        elif sc.kind == "usage":
                            usage_result = sc.result
            except LLMError as exc:
                # §5.6：200 已发出，只能发 error 事件；已生成的部分照常落库
                logger.error("流式生成中断: %s", exc)
                partial = LLMResult(content="".join(parts))
                if partial.content.strip():
                    await _persist_miss(
                        db,
                        session=session,
                        owner_id=owner_id,
                        question=question,
                        result=partial,
                        chunks=chunks,
                        incomplete=True,
                    )
                await db.commit()
                yield (
                    "error",
                    {"code": 50201, "message": "AI 服务暂时不可用，请稍后重试"},
                )
                yield ("done", {"usage": None, "incomplete": True})
                return

            content = "".join(parts)
            usage = usage_result or LLMResult(content=content)
            # 正文用逐块累积的 content，usage/token/成本用尾 chunk 的解析结果
            result = LLMResult(
                content=content,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cached_tokens=usage.cached_tokens,
                reasoning_tokens=usage.reasoning_tokens,
                latency_ms=usage.latency_ms,
                model=usage.model,
                cost_yuan=usage.cost_yuan,
                peak=usage.peak,
            )

            # 写缓存（空答案不缓存；§5.5 缓存击穿已明确记为技术债，本阶段不做锁）
            if settings.cache_enabled and content.strip():
                await cache.set_json(
                    cache_key,
                    {
                        "answer": content,
                        "chunk_ids": [c.chunk_id for c in chunks],
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "cached_tokens": result.cached_tokens,
                        "cost_yuan": result.cost_yuan,
                        "model": result.model,
                    },
                    ttl=settings.cache_ttl_seconds,
                )

            yield (
                "citations",
                {"citations": [_citation_dict(c) for c in chunks]},
            )
            yield (
                "done",
                {
                    "usage": {
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "cost_yuan": result.cost_yuan,
                        "latency_ms": result.latency_ms,
                        "cache_hit": False,
                        "retrieval_mode": "keyword",
                    }
                },
            )
            await _persist_miss(
                db,
                session=session,
                owner_id=owner_id,
                question=question,
                result=result,
                chunks=chunks,
            )
            await db.commit()
        except Exception:
            await db.rollback()
            raise


def _citation_dict(c: RetrievedChunk) -> dict:
    return {
        "document_id": c.document_id,
        "document_title": c.document_title,
        "chunk_id": c.chunk_id,
        "seq": c.seq,
        "score": c.score,
    }
