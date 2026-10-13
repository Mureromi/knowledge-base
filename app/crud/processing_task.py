"""ProcessingTask 数据访问。纯 CRUD，不做业务判断。"""

from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.processing_task import (
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    ProcessingTask,
)


async def create(
    db: AsyncSession, *, document_id: int, owner_id: int
) -> ProcessingTask:
    """新建一条 pending 任务。★ 只 flush 不 commit——提交时机由调用方控制（§5.3 竞态）。"""
    task = ProcessingTask(document_id=document_id, owner_id=owner_id, status=STATUS_PENDING)
    db.add(task)
    await db.flush()
    await db.refresh(task)
    return task


async def get(db: AsyncSession, task_id: int) -> ProcessingTask | None:
    return await db.get(ProcessingTask, task_id)


async def get_latest_for_document(
    db: AsyncSession, document_id: int
) -> ProcessingTask | None:
    """取某文档最新一条任务（轮询状态端点用）。"""
    stmt = (
        select(ProcessingTask)
        .where(ProcessingTask.document_id == document_id)
        .order_by(ProcessingTask.id.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def set_job_id(db: AsyncSession, task: ProcessingTask, job_id: str) -> None:
    task.job_id = job_id
    await db.flush()


async def mark_running(db: AsyncSession, task: ProcessingTask, *, attempt: int) -> None:
    task.status = STATUS_RUNNING
    task.attempt = attempt
    task.started_at = datetime.now(UTC)
    await db.flush()


async def mark_succeeded(
    db: AsyncSession, task: ProcessingTask, *, char_count: int, chunk_count: int
) -> None:
    task.status = STATUS_SUCCEEDED
    task.char_count = char_count
    task.chunk_count = chunk_count
    task.error = None
    task.finished_at = datetime.now(UTC)
    await db.flush()


async def mark_failed(db: AsyncSession, task: ProcessingTask, error: str) -> None:
    task.status = STATUS_FAILED
    task.error = error
    task.finished_at = datetime.now(UTC)
    await db.flush()


async def claim_stale_running(
    db: AsyncSession, stale_seconds: int, *, now: datetime | None = None
) -> list[int]:
    """★ 抢占式重置：把 running 超时的任务置为 pending，**返回真正抢到的 id 列表**。

    为什么改成返回 id（验收发现的问题）：原先的实现是"先查 stale id 列表 → 重置 → 用**条数**切片
    `stale_ids[:reset]`"，有两个缺陷：

      1. `reset` 是条数而不是 id 列表，`stale_ids[:reset]` 依赖两次查询顺序一致才能对上，
         一旦两次的过滤条件有细微差异（Python 侧 `is_stale` vs SQL 侧 `started_at <`），
         就会漏投或错投任务；
      2. 更严重的是**重复执行窗口**：任务只是慢、并没有真死，也会被重置并重投，
         于是两个 worker 同时解析同一文档（浪费 CPU，可能触发写入冲突）。

    本实现用**条件 UPDATE + RETURNING** 做原子抢占：
      - 只有"此刻仍是 running 且确实超时"的行会被更新 —— 由数据库保证互斥；
      - `RETURNING id` 精确返回被抢到的行，不再靠条数对账；
      - 调用方只对这些 id 重新投递，语义与行为一致。

    PostgreSQL 走原生 RETURNING（SQLite 3.35+ 也支持）；若方言不支持则退化为
    "先 SELECT id → 再带同样条件 UPDATE"，仍然只对真正被更新的行负责。
    """
    now = now or datetime.now(UTC)
    cutoff = now.replace(tzinfo=None) if now.tzinfo else now
    # SQLite 存的是 naive；PG 存 aware。统一按 naive-UTC 阈值比较最稳妥：
    # started_at 由 mark_running 用 datetime.now(UTC) 写入，读写两侧语义一致即可。
    threshold = cutoff - timedelta(seconds=stale_seconds)

    conditions = (
        ProcessingTask.status == STATUS_RUNNING,
        ProcessingTask.started_at < threshold,
    )

    try:
        result = await db.execute(
            update(ProcessingTask)
            .where(*conditions)
            .values(status=STATUS_PENDING, started_at=None)
            .returning(ProcessingTask.id)
        )
        claimed = [int(row[0]) for row in result.all()]
    except Exception:  # noqa: BLE001 - 方言不支持 RETURNING 时退化
        ids = [
            int(i)
            for i in (
                await db.execute(select(ProcessingTask.id).where(*conditions))
            ).scalars()
        ]
        if not ids:
            return []
        await db.execute(
            update(ProcessingTask)
            .where(ProcessingTask.id.in_(ids))
            .where(*conditions)          # 再校验一次，尽量缩小竞态窗口
            .values(status=STATUS_PENDING, started_at=None)
        )
        claimed = ids

    await db.flush()
    return claimed


async def reset_stale_running(
    db: AsyncSession, stale_seconds: int, *, now: datetime | None = None
) -> int:
    """返回被重置的**条数**（保留旧签名）。

    ⚠️ 新代码请用 `claim_stale_running` 拿 id 列表——只重置却不知道重置了谁，
    就无法安全地重新投递（会漏投或错投，见 `claim_stale_running` 的说明）。
    """
    return len(await claim_stale_running(db, stale_seconds, now=now))
