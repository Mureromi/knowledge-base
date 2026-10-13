"""阶段 3 验收修复的回归测试。

三条修复都来自[阶段3-验收报告](../docs/阶段3-验收报告.md) §5，共同的危险特征是
**修复前后 129 个测试都是全绿的**——也就是说，如果不补这些测试，
将来有人把修复改回去，测试也发现不了。

| # | 原问题 | 性质 | 本文件的防线 |
| --- | --- | --- | --- |
| 1 | 解析期间 `documents.status` 一直是 `pending`，`processing` 不可见 | 真 bug | `test_document_status_is_processing_while_parsing` |
| 2 | 删文档时任务历史被级联删除，与"可追溯历史"目标矛盾 | 设计冲突 | `test_task_history_survives_document_delete` |
| 3 | `cleanup_stale_tasks` 按条数切片对账，且无二次校验 | 健壮性缺陷 | `test_claim_stale_running_returns_ids` 等 |
"""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from app.crud import document as doc_crud
from app.crud import processing_task as task_crud
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.processing_task import ProcessingTask
from app.models.user import User
import app.worker as worker


@pytest.fixture
def worker_db(monkeypatch, session_factory):
    """让 worker 用内存 SQLite。"""
    monkeypatch.setattr("app.worker.SessionLocal", session_factory)
    return session_factory


async def _make_user(db) -> int:
    u = User(username=f"u{datetime.now(UTC).timestamp()}", password_hash="x")
    db.add(u)
    await db.flush()
    return u.id


# ---------------------------------------------------------------------------
# 修复 1：解析期间 documents.status 必须是 processing（而不是一直 pending）
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_document_status_is_processing_while_parsing(
    worker_db, make_pdf, monkeypatch, session_factory
):
    """★ 修复 1 的防线：解析进行中，documents.status 必须已被投影为 processing。

    做法：把 extract_text 换成"在被调用的那一刻回查数据库状态"的桩函数。
    如果 worker 只置任务 running、忘了同步文档状态（原 bug），
    桩函数读到的会是 pending，断言失败。

    这条测试比"轮询看 status"更可靠：轮询受时序影响（解析只要 1ms，
    中间态可能一闪而过），而桩函数是在**解析发生的那一刻**读库，时机精确。
    """
    observed: dict[str, object] = {}

    def probe_extract(raw, source_type, max_pages):
        # 用同步方式回查：这里跑在 to_thread 的线程里，不能直接 await
        import asyncio as _asyncio

        async def _read():
            async with session_factory() as db:
                doc = (await db.scalars(select(Document))).first()
                task = (await db.scalars(select(ProcessingTask))).first()
                observed["doc_status"] = doc.status if doc else None
                observed["task_status"] = task.status if task else None
                observed["started_at"] = task.started_at if task else None

        _asyncio.run_coroutine_threadsafe(_read(), loop).result(timeout=5)
        return "退款流程：客服 3 个工作日审核。"

    loop = asyncio.get_running_loop()
    monkeypatch.setattr("app.worker.extract_text", probe_extract)

    async with session_factory() as db:
        uid = await _make_user(db)
        doc = await doc_crud.create_pending(
            db, owner_id=uid, title="t", source_type="pdf",
            raw_content=make_pdf("some text"),
        )
        task = await task_crud.create(db, document_id=doc.id, owner_id=uid)
        await db.commit()
        doc_id, task_id = doc.id, task.id

    result = await worker.process_document({"job_try": 1}, task_id)
    assert result == "succeeded"

    assert observed.get("task_status") == "running", (
        "解析那一刻任务状态应为 running"
    )
    assert observed.get("doc_status") == "processing", (
        f"★ 解析期间 documents.status 必须是 processing，实际是 "
        f"{observed.get('doc_status')!r}。"
        "若为 'pending'，说明 worker 只置了任务 running、漏了同步文档状态——"
        "这正是验收发现的 bug（用户在整个解析期间看到 status=pending）。"
    )
    assert observed.get("started_at") is not None, "running 状态必须写入 started_at"


# ---------------------------------------------------------------------------
# 修复 2：删文档后任务历史必须保留（document_id 置 NULL）
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_task_history_survives_document_delete(session_factory):
    """★ 修复 2 的防线：删文档后任务流水仍在，只是 document_id 变 NULL。

    原设计是 CASCADE，文档一删历史就没了，与 §4.2 声称的
    "可追溯重试历史、统计失败率"矛盾。
    """
    async with session_factory() as db:
        uid = await _make_user(db)
        doc = await doc_crud.create_pending(
            db, owner_id=uid, title="t", source_type="pdf", raw_content=b"x"
        )
        task = await task_crud.create(db, document_id=doc.id, owner_id=uid)
        await task_crud.mark_failed(db, task, "模拟失败原因：扫描件")
        await db.commit()
        doc_id, task_id = doc.id, task.id

    async with session_factory() as db:
        doc = await doc_crud.get(db, doc_id)
        assert doc is not None
        removed_chunks = await doc_crud.delete_document(db, doc)
        await db.commit()
        assert removed_chunks == 0

    async with session_factory() as db:
        # 文档确实删了
        assert await doc_crud.get(db, doc_id) is None

        # ★ 但任务历史必须还在
        task = await db.get(ProcessingTask, task_id)
        assert task is not None, (
            "★ 删文档后任务记录消失了——说明外键又变回了 CASCADE，"
            "任务流水无法用于失败率统计（验收报告 §5.2 的问题复现）"
        )
        assert task.document_id is None, "document_id 应被置为 NULL 而非级联删除"
        assert task.error == "模拟失败原因：扫描件", "失败原因必须保留下来"
        assert task.owner_id == uid, "owner_id 冗余保留，便于按用户统计"


# ---------------------------------------------------------------------------
# 修复 3：claim_stale_running 返回 id 列表，且只影响真正超时的任务
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_claim_stale_running_returns_claimed_ids(session_factory):
    """★ 修复 3 的防线：返回**抢到的 id 列表**，而不是一个条数。

    原实现只返回条数，调用方用 `stale_ids[:reset]` 切片对账——
    两次判定条件不一致时会漏投/错投任务。
    """
    async with session_factory() as db:
        uid = await _make_user(db)
        ids = []
        for i in range(3):
            doc = await doc_crud.create_pending(
                db, owner_id=uid, title=f"t{i}", source_type="pdf", raw_content=b"x"
            )
            t = await task_crud.create(db, document_id=doc.id, owner_id=uid)
            ids.append(t.id)
        await db.commit()

        # 1 号：真僵死（started_at 久远）；2 号：刚开始 running（不该被动）
        await task_crud.mark_running(db, await task_crud.get(db, ids[0]), attempt=1)
        t0 = await task_crud.get(db, ids[0])
        t0.started_at = datetime.now(UTC) - timedelta(seconds=9999)
        await task_crud.mark_running(db, await task_crud.get(db, ids[1]), attempt=1)
        await db.commit()

    async with session_factory() as db:
        claimed = await task_crud.claim_stale_running(db, stale_seconds=600)
        await db.commit()

    assert claimed == [ids[0]], (
        f"只应抢到真正超时的 {ids[0]}，实际 {claimed}。"
        "若把刚 running 的任务也抢走，会导致两个 worker 同时解析同一文档。"
    )

    async with session_factory() as db:
        t0 = await db.get(ProcessingTask, ids[0])
        t1 = await db.get(ProcessingTask, ids[1])
        t2 = await db.get(ProcessingTask, ids[2])
        assert t0.status == "pending" and t0.started_at is None, "被抢到的应变回 pending"
        assert t1.status == "running", "未超时的必须保持 running（不被误抢）"
        assert t2.status == "pending", "从未运行的不受影响"

        # ★ 幂等：再抢一次不应重复抢到同一任务
        again = await task_crud.claim_stale_running(db, stale_seconds=600)
        await db.commit()
        assert again == [], (
            f"第二次调用不应再抢到任何任务（已置 pending），实际 {again} —— "
            "否则会重复投递，造成同一文档被并发解析"
        )


@pytest.mark.asyncio
async def test_reset_stale_running_keeps_int_count(session_factory):
    """兼容层：旧签名仍返回条数（供既有测试/调用方使用）。"""
    async with session_factory() as db:
        uid = await _make_user(db)
        doc = await doc_crud.create_pending(
            db, owner_id=uid, title="t", source_type="pdf", raw_content=b"x"
        )
        t = await task_crud.create(db, document_id=doc.id, owner_id=uid)
        await db.commit()
        await task_crud.mark_running(db, await task_crud.get(db, t.id), attempt=1)
        row = await task_crud.get(db, t.id)
        row.started_at = datetime.now(UTC) - timedelta(seconds=9999)
        await db.commit()

    async with session_factory() as db:
        n = await task_crud.reset_stale_running(db, stale_seconds=600)
        await db.commit()
    assert n == 1 and isinstance(n, int)
