"""为 chunks 补算 embedding。

用法：
    .venv/bin/python scripts/backfill_embeddings.py --dry-run
    .venv/bin/python scripts/backfill_embeddings.py --batch-size 32
    .venv/bin/python scripts/backfill_embeddings.py --document-id 7

★ 幂等：只处理 embedding IS NULL 的行。中断、超时、模型下载失败后重跑都安全，
  不会重算已成功的行，也不会产生重复数据。

★ 为什么必须幂等：回填可能因为各种原因中途停止（模型 1GB 要下载、某批编码抛异常、
  被 Ctrl-C）。"再跑一次"必须是安全的——这是所有数据修复脚本的基本要求。

★ 新增切片不需要跑这个脚本：worker 解析入库时就该直接写向量（app/worker.py）。
  回填只用于**历史数据**和**换 embedding 模型后重算**。

对应 docs/项目实现文档-阶段4.md §7.1。四个设计要点：
  1. WHERE embedding IS NULL —— 幂等，中断后重跑只处理剩余的
  2. 每批一个事务 —— 整表一个事务会长时间持锁，几百页文档时阻塞其他写入
  3. 不调 embed_query（不加指令前缀）—— 文档侧加前缀会降低检索效果
  4. asyncio.to_thread 包装编码 —— 编码是同步 CPU 活，直接跑会阻塞事件循环
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select, update

from app.core.config import settings
from app.db.session import SessionLocal, engine
from app.models.chunk import Chunk


async def backfill(*, dry_run: bool, batch_size: int, document_id: int | None) -> None:
    # 必须在 import fastembed 之前设好（§1.3：三个环境变量缺一不可）
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from fastembed import TextEmbedding

    async with SessionLocal() as db:
        total = int((await db.scalar(select(func.count(Chunk.id)))) or 0)
        pending_stmt = select(func.count(Chunk.id)).where(Chunk.embedding.is_(None))
        if document_id is not None:
            pending_stmt = pending_stmt.where(Chunk.document_id == document_id)
        pending = int((await db.scalar(pending_stmt)) or 0)

    print(f"切片总数: {total}  待回填: {pending}  批大小: {batch_size}")
    if pending == 0:
        print("✅ 没有待回填的切片（幂等：重跑安全）")
        return
    if dry_run:
        print("DRY-RUN：未写入。去掉 --dry-run 实际执行。")
        return

    model = TextEmbedding(
        model_name=settings.embedding_model,
        cache_dir=settings.embedding_cache_dir,
    )
    print(f"模型已加载: {settings.embedding_model}（维度 {settings.embedding_dim}）")

    done = 0
    while True:
        # ★ 每批一个事务：不要整表一个大事务（几百页会长时间持锁）
        async with SessionLocal() as db:
            stmt = select(Chunk.id, Chunk.content).where(Chunk.embedding.is_(None))
            if document_id is not None:
                stmt = stmt.where(Chunk.document_id == document_id)
            rows = (await db.execute(stmt.limit(batch_size))).all()
            if not rows:
                break

            ids = [int(r[0]) for r in rows]
            texts = [r[1] for r in rows]

            # ★ 文档侧不加指令前缀（只有 query 加，见 §1.4）
            # ★ 同步 CPU 活在 to_thread 里跑（阶段 3 §1 的同一原则）
            vectors = await asyncio.to_thread(
                lambda: [v.tolist() for v in model.embed(texts)]
            )

            for cid, vec in zip(ids, vectors, strict=True):
                await db.execute(
                    update(Chunk).where(Chunk.id == cid).values(embedding=vec)
                )
            await db.commit()
            done += len(ids)
            pct = 100.0 * done / max(pending, 1)
            print(f"  已回填 {done}/{pending}（{pct:.1f}%）")

    print(f"\n✅ 回填完成，共处理 {done} 个切片")


async def run(*, dry_run: bool, batch_size: int, document_id: int | None) -> None:
    """★ 执行 + 清理必须在**同一个事件循环**里。

    踩过的坑：原先写成
        asyncio.run(backfill(...))
        ...
        asyncio.run(engine.dispose())        # ← 第二个循环
    结果 dispose 时抛 `RuntimeError: Event loop is closed` ——
    因为 `app.db.session.engine` 是模块级单例，它的 asyncpg 连接绑定在**第一个**
    事件循环上；第二个循环里销毁它，连接池持有的 loop 已关闭。

    正确做法（与 scripts/init_db.py 一致）：把 dispose 放进同一个协程里。
    """
    try:
        await backfill(dry_run=dry_run, batch_size=batch_size, document_id=document_id)
    finally:
        await engine.dispose()


def main() -> None:
    ap = argparse.ArgumentParser(description="为 chunks 补算 embedding（幂等）")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写入")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--document-id", type=int, default=None, help="只回填某个文档")
    a = ap.parse_args()
    asyncio.run(run(dry_run=a.dry_run, batch_size=a.batch_size, document_id=a.document_id))


if __name__ == "__main__":
    main()
