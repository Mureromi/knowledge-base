"""rerank 增量评测（docs/项目实现文档-阶段4.md §9.5）。

★ 这是本阶段"必须做一次"的运行期验收：用**真实 embedder + 真实 reranker + 真实 PG**，
  在自有语料上分别以 RERANK_ENABLED=false / true 各检索一遍，对比 Top-1 / Top-5 命中率，
  据此决定是否长期保留这个约 1GB 的 cross-encoder——而不是"教程说必须加"。

用法：
    .venv/bin/python scripts/eval_rerank.py                # 完整评测（跑完自动清理评测数据）
    .venv/bin/python scripts/eval_rerank.py --keep         # 保留评测语料在库里，便于复查
    .venv/bin/python scripts/eval_rerank.py --top-k 5      # 指定每查询取前 K（默认 5）

设计要点：
  1. 语料灌进一个**独立评测用户**（username=__eval_rerank__），search 按 owner_id 过滤，
     绝不污染 demo 等真实用户；跑完删除该用户，chunks/documents 靠 ON DELETE CASCADE 级联清掉。
  2. **绕开查询改写**：直接调 retrieval.search，把变量收敛到"检索 + 精排"这一层，
     测的就是 rerank 本身的增量，不掺 LLM 改写。
  3. 两趟之间语料与向量完全不变，只有 settings.rerank_enabled 翻转——差异可归因于 rerank。
  4. 向量在**评测内现算**（真实模型、文档侧不加前缀），与生产 backfill 口径一致。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from sqlalchemy import delete

from app.clients.embedding import embedding_client
from app.clients.rerank import rerank_client
from app.core.config import settings
from app.db.session import SessionLocal, engine
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.user import User
from app.services import retrieval

EVAL_USERNAME = "__eval_rerank__"
CASES_PATH = Path(__file__).resolve().parent.parent / "app" / "evals" / "rerank_cases.yaml"


@dataclass
class CaseResult:
    question: str
    expect_key: str
    expect_chunk_id: int
    baseline_top1: int          # 期望切片在不 rerank 结果里的名次（1 起），未进 Top-K → K+1
    rerank_top1: int
    baseline_order: list[int]   # 各模式返回的 chunk_id 顺序，便于逐例复盘
    rerank_order: list[int]


def _load_cases() -> tuple[list[dict], list[dict]]:
    data = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))
    corpus, cases = data["corpus"], data["cases"]
    keys = {c["key"] for c in corpus}
    for case in cases:
        if case["expect"] not in keys:
            raise ValueError(f"用例期望切片 {case['expect']!r} 不在 corpus 中")
    return corpus, cases


async def _seed_corpus(db, corpus: list[dict]) -> tuple[dict[str, int], int]:
    """建独立评测用户 + 灌入语料（每片一条 Document + 一条带真实向量的 Chunk）。

    返回 ({corpus key: chunk_id}, 评测用户 id)，前者供逐例判定命中。
    """
    user = User(username=EVAL_USERNAME, password_hash="eval-only-not-a-real-login")
    db.add(user)
    await db.flush()

    # 真实模型编码文档侧（不加指令前缀，与 backfill_embeddings.py 口径一致）
    texts = [item["content"] for item in corpus]
    vectors = await embedding_client.embed_documents(texts)

    key_to_chunk: dict[str, int] = {}
    for item, vec in zip(corpus, vectors, strict=True):
        doc = Document(
            owner_id=user.id, title=item["title"], source_type="txt",
            raw_text=item["content"], status="ready",
        )
        db.add(doc)
        await db.flush()
        chunk = Chunk(
            document_id=doc.id, owner_id=user.id, seq=0,
            content=item["content"], char_count=len(item["content"]), embedding=vec,
        )
        db.add(chunk)
        await db.flush()
        key_to_chunk[item["key"]] = int(chunk.id)
    await db.commit()
    return key_to_chunk, int(user.id)


async def _rank_of_expect(
    db, owner_id: int, question: str, expect_chunk_id: int,
    *, top_k: int, use_rerank: bool,
) -> tuple[int, list[int]]:
    """跑一次检索，返回 (期望切片名次, 返回顺序)。未进 Top-K 记为 K+1。"""
    original = settings.rerank_enabled
    settings.rerank_enabled = use_rerank
    try:
        results = await retrieval.search(
            db, owner_id=owner_id, question=question, top_k=top_k,
            embedder=embedding_client, reranker=rerank_client,
        )
    finally:
        settings.rerank_enabled = original
    order = [r.chunk_id for r in results]
    rank = order.index(expect_chunk_id) + 1 if expect_chunk_id in order else top_k + 1
    return rank, order


async def evaluate(*, top_k: int, keep: bool) -> None:
    corpus, cases = _load_cases()

    embedding_client.startup()
    if not embedding_client.is_ready:
        raise SystemExit("embedder 未就绪，无法评测（检查模型缓存 / HF 环境变量）")
    rerank_client.startup()
    if not rerank_client.is_ready:
        print("⚠️  rerank 模型未加载——两趟都将等同 baseline，增量必为 0。检查模型缓存。")

    print(
        f"语料 {len(corpus)} 片 / 用例 {len(cases)} 条，Top-K={top_k}，"
        f"embedding={settings.embedding_model}，rerank={settings.rerank_model}"
    )

    results: list[CaseResult] = []
    owner_id: int | None = None
    try:
        async with SessionLocal() as db:
            key_to_chunk, owner_id = await _seed_corpus(db, corpus)

            for case in cases:
                expect_chunk_id = key_to_chunk[case["expect"]]
                b_rank, b_order = await _rank_of_expect(
                    db, owner_id, case["question"], expect_chunk_id,
                    top_k=top_k, use_rerank=False,
                )
                r_rank, r_order = await _rank_of_expect(
                    db, owner_id, case["question"], expect_chunk_id,
                    top_k=top_k, use_rerank=True,
                )
                results.append(CaseResult(
                    question=case["question"], expect_key=case["expect"],
                    expect_chunk_id=expect_chunk_id,
                    baseline_top1=b_rank, rerank_top1=r_rank,
                    baseline_order=b_order, rerank_order=r_order,
                ))

            _report(results, top_k)
    finally:
        if not keep and owner_id is not None:
            async with SessionLocal() as db:
                # 删评测用户即可，chunks/documents 靠 ON DELETE CASCADE 级联清理
                await db.execute(delete(User).where(User.id == owner_id))
                await db.commit()
            print(f"\n🧹 已清理评测数据（用户 id={owner_id} 及其级联文档/切片）")
        elif keep:
            print(f"\n📌 已保留评测数据（用户 id={owner_id}）；复查后记得手动清理")
        await engine.dispose()


def _report(results: list[CaseResult], top_k: int) -> None:
    n = len(results)
    print("\n" + "=" * 96)
    print(f"{'问题':<26}{'期望':<20}{'baseline':>10}{'rerank':>10}{'Δ名次':>8}")
    print("-" * 96)
    for r in results:
        delta = r.baseline_top1 - r.rerank_top1   # 正数 = rerank 让期望切片名次更靠前
        mark = "  ↑" if delta > 0 else ("  ↓" if delta < 0 else "  =")
        q = (r.question[:12] + "…") if len(r.question) > 13 else r.question
        print(f"{q:<26}{r.expect_key:<20}"
              f"{'#'+str(r.baseline_top1):>10}{'#'+str(r.rerank_top1):>10}{mark:>4}{abs(delta):>4}")

    def hit(get_rank, k):
        return sum(1 for r in results if get_rank(r) <= k)

    b1, r1 = hit(lambda r: r.baseline_top1, 1), hit(lambda r: r.rerank_top1, 1)
    b5, r5 = hit(lambda r: r.baseline_top1, 5), hit(lambda r: r.rerank_top1, 5)
    print("-" * 96)
    print(f"Top-1 命中：baseline {b1}/{n} ({b1/n:.0%})   rerank {r1}/{n} ({r1/n:.0%})   "
          f"增量 {r1/n - b1/n:+.0%}")
    print(f"Top-5 命中：baseline {b5}/{n} ({b5/n:.0%})   rerank {r5}/{n} ({r5/n:.0%})   "
          f"增量 {r5/n - b5/n:+.0%}")
    print("=" * 96)
    print("决策阈值（§9.5）：Top-1 增量 ≥ +10% 保留；0~10% 记录权衡；几乎为 0 → RERANK_ENABLED=false 代码留着")


def main() -> None:
    ap = argparse.ArgumentParser(description="rerank 增量评测（§9.5）")
    ap.add_argument("--top-k", type=int, default=settings.final_top_k,
                    help="每查询取前 K（默认取配置 final_top_k）")
    ap.add_argument("--keep", action="store_true", help="保留评测语料不清理")
    a = ap.parse_args()
    asyncio.run(evaluate(top_k=a.top_k, keep=a.keep))


if __name__ == "__main__":
    main()
