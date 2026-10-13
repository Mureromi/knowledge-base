"""上传异步化端点测试（阶段 3 §5 / §9.3）。

覆盖：pdf/docx → 202 + pending + task_id + 投递一次；投递前已 commit；
超大 → 413 且不落库；不支持格式 → 415；txt/md → 快路径 status=ready 有切片。"""

import pytest
from httpx import AsyncClient

from app.core.config import settings
from app.crud import processing_task as task_crud


async def _upload(client, headers, filename, content: bytes, ctype="application/octet-stream"):
    return await client.post(
        "/api/v1/documents",
        headers=headers,
        files={"file": (filename, content, ctype)},
    )


@pytest.mark.asyncio
async def test_upload_pdf_returns_202_pending_and_enqueues_once(
    client: AsyncClient, auth_headers, fake_arq, make_pdf
):
    raw = make_pdf("Shipments are dispatched within 48 hours.")
    r = await _upload(client, auth_headers, "logistics.pdf", raw)
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] == "pending"
    assert body["task_id"] is not None
    assert body["chunk_count"] == 0
    # 恰好投递一次 process_document
    assert len(fake_arq.jobs) == 1
    fn, args, kwargs = fake_arq.jobs[0]
    assert fn == "process_document"
    assert args == (body["task_id"],)
    assert kwargs.get("_job_id") == f"doc:{body['id']}:{body['task_id']}"


@pytest.mark.asyncio
async def test_task_committed_before_enqueue(client: AsyncClient, auth_headers, session_factory, make_pdf):
    """★ 竞态防线（§5.3）：enqueue 前必须已 commit，否则模拟 worker 立即读会查不到任务。

    这里直接用独立 session（模拟另一个进程的 worker）按 task_id 取任务，应能查到。
    """
    r = await _upload(client, auth_headers, "policy.pdf", make_pdf("Refund in 3 days."))
    assert r.status_code == 202
    task_id = r.json()["task_id"]

    async with session_factory() as db:      # 全新 session，等同 worker 进程视角
        task = await task_crud.get(db, task_id)
        assert task is not None, "投递任务前未提交，worker 会查不到任务"
        assert task.document_id == r.json()["id"]


@pytest.mark.asyncio
async def test_upload_oversize_returns_413_and_no_db_record(
    client: AsyncClient, auth_headers, session_factory, monkeypatch
):
    from sqlalchemy import func, select

    from app.models.document import Document

    monkeypatch.setattr(settings, "max_upload_bytes", 10)   # 压到 10 字节便于触发
    before = None
    async with session_factory() as db:
        before = int((await db.execute(select(func.count(Document.id)))).scalar_one())

    r = await _upload(client, auth_headers, "big.pdf", b"x" * 50)
    assert r.status_code == 413, r.text

    async with session_factory() as db:
        after = int((await db.execute(select(func.count(Document.id)))).scalar_one())
    assert after == before, "413 不应产生任何文档记录"


@pytest.mark.asyncio
async def test_upload_unsupported_format_returns_415(client: AsyncClient, auth_headers):
    r = await _upload(client, auth_headers, "report.xlsx", b"whatever")
    assert r.status_code == 415


@pytest.mark.asyncio
async def test_upload_txt_fast_path_ready_immediately_with_chunks(
    client: AsyncClient, auth_headers, fake_arq
):
    """txt/md 是纯文本，走同步快路径：直接 status=ready、有切片、不投递 worker。"""
    r = await _upload(
        client, auth_headers, "notes.txt", "第一段内容。\n\n第二段内容。".encode(),
        "text/plain",
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] == "ready"
    assert body["chunk_count"] >= 1
    assert body["task_id"] is None
    assert len(fake_arq.jobs) == 0      # 快路径不进队列


@pytest.mark.asyncio
async def test_upload_docx_returns_202_pending(client: AsyncClient, auth_headers, make_docx):
    raw = make_docx(["Annual report summary."], [["Revenue", "100M"]])
    r = await _upload(client, auth_headers, "report.docx", raw)
    assert r.status_code == 202, r.text
    assert r.json()["status"] == "pending"
    assert r.json()["task_id"] is not None
