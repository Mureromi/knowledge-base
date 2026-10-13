"""检索层。阶段 4：关键词 + 向量 → RRF 融合 → rerank。

★ 签名保持不变（`search(db, owner_id=..., question=..., top_k=...)`），
  这样 qa.py 只需要多传一个 resolved_query + embedder/reranker，其余逻辑不动 ——
  这是阶段 0 就定下的"换实现不换签名"约定的兑现（docs/项目实现文档-阶段4.md §5）。

对应 §5.2。相对文档伪代码的两处必要修正：
1. rerank 调用用 try/except 包住（§9.2 要求"rerank 抛异常降级为 RRF 顺序"），
   文档 §5.2 的伪代码没有这层保护。
2. 向量召回按方言分支：PostgreSQL 走 pgvector `<#>` 算子，SQLite（测试）走应用层余弦
   （§9.1"应用层排序"）——`<#>` 是 PG 专属算子，SQLite 无法执行。
"""

import logging
import math
from dataclasses import dataclass

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.embedding import EmbeddingClient
from app.clients.rerank import RerankClient
from app.core.config import settings
from app.models.chunk import Chunk
from app.models.document import Document

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    document_title: str
    seq: int
    content: str
    score: float = 0.0
    # 阶段 4 新增：便于排查"这条是靠关键词还是向量召回的"
    keyword_rank: int | None = None
    vector_rank: int | None = None


def _keywords(question: str) -> list[str]:
    """极简分词：英文按空格，中文按 2-gram。与阶段 0 完全一致——
    现在它是"两路召回中的一路"，而不是唯一手段，无需改动。"""
    tokens: list[str] = [w.strip() for w in question.split() if w.strip()]
    chinese = "".join(ch for ch in question if "\u4e00" <= ch <= "\u9fff")
    tokens += [chinese[i:i + 2] for i in range(len(chinese) - 1)]
    return list({t for t in tokens if len(t) >= 2})[:20]


def _dialect_name(db: AsyncSession) -> str:
    """取当前会话绑定引擎的方言名（用于向量召回的 PG / SQLite 分支）。"""
    return db.get_bind().dialect.name


def _cosine(a: list[float], b: list[float]) -> float:
    """应用层余弦相似度（仅 SQLite 测试路径用）。任一向量为零 → 返回 0.0，
    避免除零，也符合 §9.4"全零向量排序任意"的告诫（这里统一给 0，由 chunk_id 决定次序）。"""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


async def _keyword_candidates(
    db: AsyncSession, *, owner_id: int, query: str, limit: int
) -> list[tuple[int, float]]:
    """关键词召回。返回 [(chunk_id, 命中关键词数)]，按命中数降序。"""
    tokens = _keywords(query)
    if not tokens:
        return []
    stmt = (
        select(Chunk.id, Chunk.content)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.owner_id == owner_id)
        .where(Document.status == "ready")          # ★ 阶段 3 的约定：只检索 ready
        .where(or_(*[Chunk.content.ilike(f"%{t}%") for t in tokens]))
        .limit(limit * 4)
    )
    rows = (await db.execute(stmt)).all()
    scored = [
        (int(cid), float(sum(1 for t in tokens if t in content)))
        for cid, content in rows
    ]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:limit]


async def _vector_candidates(
    db: AsyncSession,
    embedder: EmbeddingClient,
    *,
    owner_id: int,
    query: str,
    limit: int,
) -> list[tuple[int, float]]:
    """向量召回。返回 [(chunk_id, 相似度)]，按相似度降序。

    ★ 只召回 embedding 非空的切片：回填期间新旧混存，NULL 会被 SQL 的 ORDER BY 排到最后，
      显式过滤更清晰。
    """
    # ★ query 侧加指令前缀（文档侧不加）—— 阶段 4 前置实测：区分度 +64%
    query_vec = await embedder.embed_query(query)

    if _dialect_name(db) == "postgresql":
        # pgvector 内积算子 <#>：返回负内积，升序即最相似优先（向量已 L2 归一化 → 等价余弦）
        stmt = (
            select(Chunk.id, Chunk.embedding.max_inner_product(query_vec).label("sim"))
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.owner_id == owner_id)
            .where(Document.status == "ready")
            .where(Chunk.embedding.is_not(None))
            .order_by(Chunk.embedding.max_inner_product(query_vec))
            .limit(limit)
        )
        rows = (await db.execute(stmt)).all()
        return [(int(cid), float(sim)) for cid, sim in rows]

    # SQLite（仅测试）：<#> 是 PG 专属算子，无法执行 → 应用层余弦排序（§9.1）。
    # 语义与生产一致：向量已 L2 归一化，内积 = 余弦。
    stmt = (
        select(Chunk.id, Chunk.embedding)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.owner_id == owner_id)
        .where(Document.status == "ready")
        .where(Chunk.embedding.is_not(None))
    )
    rows = (await db.execute(stmt)).all()
    scored = [
        (int(cid), _cosine(query_vec, vec))
        for cid, vec in rows
        if vec is not None
    ]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:limit]


def rrf_fuse(
    ranked_lists: list[list[int]], *, k: int | None = None
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion：score(d) = Σ 1/(k + rank_i(d))。

    为什么用 RRF 而不是加权分数：关键词得分（命中个数）与向量相似度量纲完全不同，
    直接加权需要归一化和调参；RRF 只用**名次**，天然免调参、对量纲不敏感。
    k=60 来自原论文（Cormack et al. 2009），作用是压低头部名次的权重差。

    ★ 这是本阶段最值得记住的一个"少调参"技巧。
    """
    k = k or settings.rrf_k
    agg: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, cid in enumerate(lst, start=1):
            agg[cid] = agg.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))


async def _load_chunks(
    db: AsyncSession, chunk_ids: list[int]
) -> dict[int, tuple[Chunk, str]]:
    """批量取切片 + 文档标题（一次查询，避免 N+1）。"""
    if not chunk_ids:
        return {}
    stmt = (
        select(Chunk, Document.title)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.id.in_(chunk_ids))
    )
    return {c.id: (c, title) for c, title in (await db.execute(stmt)).all()}


async def search(
    db: AsyncSession,
    *,
    owner_id: int,
    question: str,
    top_k: int | None = None,
    embedder: EmbeddingClient | None = None,
    reranker: RerankClient | None = None,
) -> list[RetrievedChunk]:
    """混合检索主入口。签名与阶段 0 保持一致（新增可选依赖参数）。

    embedder / reranker 允许为 None：便于测试注入假实现，也便于"临时关掉向量"对比效果。
    """
    final_k = top_k or settings.final_top_k

    # ---------- ① 两路召回 ----------
    keyword_pairs = await _keyword_candidates(
        db, owner_id=owner_id, query=question, limit=settings.keyword_top_k
    )
    vector_pairs: list[tuple[int, float]] = []
    if settings.embedding_enabled and embedder is not None:
        try:
            vector_pairs = await _vector_candidates(
                db, embedder, owner_id=owner_id, query=question, limit=settings.vector_top_k
            )
        except Exception as exc:                        # noqa: BLE001
            # ★ 向量路任何异常（模型未加载 / 编码失败 / 维度不符）都不能阻断问答：
            #   降级为纯关键词（§10“假 embedder 返回异常时问答不崩”）。
            logger.warning("向量召回失败，降级为关键词检索: %s", exc)
            vector_pairs = []

    if not keyword_pairs and not vector_pairs:
        return []

    # ---------- ② RRF 融合 ----------
    fused = rrf_fuse(
        [[cid for cid, _ in keyword_pairs], [cid for cid, _ in vector_pairs]]
    )
    kw_rank = {cid: i for i, (cid, _) in enumerate(keyword_pairs, start=1)}
    vec_rank = {cid: i for i, (cid, _) in enumerate(vector_pairs, start=1)}

    # ---------- ③ rerank（只对前 N 个）----------
    head = fused[: settings.rerank_top_k]
    tail = fused[settings.rerank_top_k :]

    if (
        settings.rerank_enabled
        and reranker is not None
        and len(head) >= settings.rerank_min_candidates
    ):
        loaded = await _load_chunks(db, [cid for cid, _ in head])
        pairs = [(cid, loaded[cid][0].content) for cid, _ in head if cid in loaded]
        if len(pairs) >= settings.rerank_min_candidates:
            ids = [cid for cid, _ in pairs]
            docs = [content for _, content in pairs]
            try:
                scores = await reranker.rerank(question, docs)
                order = sorted(range(len(ids)), key=lambda i: -scores[i])
                head = [(ids[i], float(scores[i])) for i in order]
                # rerank 分数量纲与 RRF 不同，尾部保留 RRF 分（仅作补充，通常用不到）
            except Exception as exc:                       # noqa: BLE001
                # ★ §9.2：rerank 抛异常 → 降级为 RRF 顺序（head 保持不变），问答不受影响
                logger.warning("rerank 失败，降级为 RRF 顺序: %s", exc)

    # ---------- ④ 组装 ----------
    ordered = head + tail
    ordered = ordered[: max(final_k, settings.rerank_top_k)]
    loaded = await _load_chunks(db, [cid for cid, _ in ordered])

    results: list[RetrievedChunk] = []
    for cid, score in ordered:
        item = loaded.get(cid)
        if item is None:
            continue
        chunk, title = item
        results.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                document_title=title,
                seq=chunk.seq,
                content=chunk.content,
                score=score,
                keyword_rank=kw_rank.get(cid),
                vector_rank=vec_rank.get(cid),
            )
        )
        if len(results) >= final_k:
            break
    return results


def resolve_retrieval_mode(
    embedder: EmbeddingClient | None = None,
    reranker: RerankClient | None = None,
) -> str:
    """接口契约字段 `retrieval_mode` 的取值（§8）。

    反映"本次配置用到了哪几路"，让同一个接口上可对比 keyword / hybrid 的效果差异：
      - embedding 关掉或没注入 embedder → 退回纯关键词（渐进降级，行为同阶段 3）
      - 向量路生效 + rerank 开启         → hybrid+rerank
      - 向量路生效、rerank 关            → hybrid
    """
    if not settings.embedding_enabled or embedder is None:
        return "keyword"
    if settings.rerank_enabled and reranker is not None:
        return "hybrid+rerank"
    return "hybrid"
