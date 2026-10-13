"""Worker 任务函数测试（阶段 3 §6 / §9.3）。

策略（§9.1）：不起 worker 进程、不连 Redis，直接调用 process_document，
把 app.worker.SessionLocal 换成内存 SQLite 的 session_factory（模拟 worker 进程库视角）。"""


import pytest
from httpx import AsyncClient

from app.crud import document as doc_crud
from app.crud import processing_task as task_crud
from app.models.chunk import Chunk
from app.worker import process_document


async def _upload_pdf(client, headers, raw: bytes, name="d.pdf"):
    r = await client.post(
        "/api/v1/documents", headers=headers,
        files={"file": (name, raw, "application/octet-stream")},
    )
    assert r.status_code == 202, r.text
    return r.json()


@pytest.fixture
def worker_db(monkeypatch, session_factory):
    """把 worker 模块里的 SessionLocal 换成测试内存库工厂。"""
    monkeypatch.setattr("app.worker.SessionLocal", session_factory)
    return session_factory


@pytest.mark.asyncio
async def test_normal_parse_pending_to_ready(
    client: AsyncClient, auth_headers, worker_db, make_pdf
):
    body = await _upload_pdf(client, auth_headers, make_pdf("Refund is reviewed in 3 workdays."))
    result = await process_document({"job_try": 1}, body["task_id"])
    assert result == "succeeded"

    async with worker_db() as db:
        doc = await doc_crud.get(db, body["id"])
        task = await task_crud.get(db, body["task_id"])
        assert doc.status == "ready"
        assert task.status == "succeeded"
        assert task.char_count > 0
        assert task.chunk_count >= 1
        assert doc.raw_text and "Refund" in doc.raw_text


@pytest.mark.asyncio
async def test_scanned_pdf_fails_without_raising(
    client: AsyncClient, auth_headers, worker_db, make_pdf
):
    """★ 扫描件：置 failed、error 说明原因，且【不抛异常】（不触发 ARQ 重试）。"""
    body = await _upload_pdf(client, auth_headers, make_pdf(None))
    result = await process_document({"job_try": 1}, body["task_id"])
    assert result.startswith("failed:")            # 返回而非 raise

    async with worker_db() as db:
        task = await task_crud.get(db, body["task_id"])
        doc = await doc_crud.get(db, body["id"])
        assert task.status == "failed"
        assert ("扫描件" in (task.error or "")) or ("文字层" in (task.error or ""))
        assert doc.status == "failed"


@pytest.mark.asyncio
async def test_repeat_execution_is_idempotent(
    client: AsyncClient, auth_headers, worker_db, make_pdf, session_factory
):
    """★ 重复执行同一任务：第二次 already-succeeded，chunks 不翻倍。"""
    from sqlalchemy import func, select

    body = await _upload_pdf(client, auth_headers, make_pdf("Dispatch within 48 hours."))
    await process_document({"job_try": 1}, body["task_id"])

    async with session_factory() as db:
        first = int((await db.execute(
            select(func.count(Chunk.id)).where(Chunk.document_id == body["id"])
        )).scalar_one())

    second = await process_document({"job_try": 2}, body["task_id"])
    assert second == "already-succeeded"

    async with session_factory() as db:
        after = int((await db.execute(
            select(func.count(Chunk.id)).where(Chunk.document_id == body["id"])
        )).scalar_one())
    assert after == first, "重复执行不应产生额外切片"


@pytest.mark.asyncio
async def test_replace_chunks_removes_residue_on_retry(
    client: AsyncClient, auth_headers, worker_db, make_pdf, session_factory
):
    """失败重试前先写了一半切片 → replace_chunks 先删后写，不产生重复（§6.3 ③）。"""
    from sqlalchemy import select

    body = await _upload_pdf(client, auth_headers, make_pdf("Warranty lasts 24 months."))
    # 手工塞入"上次残留"的错误切片
    async with session_factory() as db:
        db.add_all([
            Chunk(document_id=body["id"], owner_id=1, seq=0, content="残留A", char_count=3),
            Chunk(document_id=body["id"], owner_id=1, seq=1, content="残留B", char_count=3),
        ])
        await db.commit()

    await process_document({"job_try": 1}, body["task_id"])

    async with session_factory() as db:
        contents = (await db.execute(
            select(Chunk.content).where(Chunk.document_id == body["id"])
        )).scalars().all()
    assert all("残留" not in c for c in contents), "replace_chunks 应清掉上次残留"
    assert any("Warranty" in c for c in contents)


@pytest.mark.asyncio
async def test_task_not_found_returns_without_raising(worker_db):
    result = await process_document({"job_try": 1}, 999_999)
    assert result == "task-not-found"


@pytest.mark.asyncio
async def test_stale_running_task_is_detectable_and_resettable(
    client: AsyncClient, auth_headers, worker_db, make_pdf, session_factory
):
    """★ 僵死任务：running 超时能被识别，reset_stale_running 重置为 pending。"""
    from datetime import UTC, datetime, timedelta

    from app.core.config import settings

    body = await _upload_pdf(client, auth_headers, make_pdf("Some content here."))
    # 人为置 running 且 started_at 很久以前
    async with session_factory() as db:
        task = await task_crud.get(db, body["task_id"])
        await task_crud.mark_running(db, task, attempt=1)
        task.started_at = datetime.now(UTC) - timedelta(seconds=settings.task_stale_seconds + 100)
        await db.commit()
        assert task.is_stale(settings.task_stale_seconds) is True

    async with session_factory() as db:
        reset = await task_crud.reset_stale_running(db, settings.task_stale_seconds)
        await db.commit()
    assert reset >= 1

    async with session_factory() as db:
        task = await task_crud.get(db, body["task_id"])
        assert task.status == "pending"


@pytest.mark.asyncio
async def test_unknown_error_triggers_retry_via_transient(
    client: AsyncClient, auth_headers, worker_db, make_pdf, monkeypatch
):
    """未知异常 → raise TransientError 交给 ARQ 重试（区别于确定性失败）。"""
    from app.worker import TransientError

    def boom(*_a, **_k):
        raise RuntimeError("disk full simulated")

    monkeypatch.setattr("app.worker.extract_text", boom)
    body = await _upload_pdf(client, auth_headers, make_pdf("Anything."))
    with pytest.raises(TransientError):
        await process_document({"job_try": 1}, body["task_id"])
