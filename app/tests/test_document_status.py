"""文档状态轮询端点测试（阶段 3 §5.4 / §9.3）。"""

import pytest
from httpx import AsyncClient

from app.tests.conftest import register_and_login
from app.worker import process_document


@pytest.fixture
def worker_db(monkeypatch, session_factory):
    monkeypatch.setattr("app.worker.SessionLocal", session_factory)
    return session_factory


async def _upload(client, headers, filename, content: bytes):
    r = await client.post(
        "/api/v1/documents", headers=headers,
        files={"file": (filename, content, "application/octet-stream")},
    )
    assert r.status_code == 202, r.text
    return r.json()


@pytest.mark.asyncio
async def test_status_pending_before_processing(
    client: AsyncClient, auth_headers, make_pdf
):
    body = await _upload(client, auth_headers, "d.pdf", make_pdf("Some text."))
    r = await client.get(f"/api/v1/documents/{body['id']}/status", headers=auth_headers)
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["status"] == "pending"
    assert s["task_status"] == "pending"
    assert s["finished"] is False          # 前端据此继续轮询


@pytest.mark.asyncio
async def test_status_after_failure_reports_error(
    client: AsyncClient, auth_headers, worker_db, make_pdf
):
    """失败时把 error 原文返回（§5.4：用户要知道怎么办）。"""
    body = await _upload(client, auth_headers, "scan.pdf", make_pdf(None))
    await process_document({"job_try": 1}, body["task_id"])

    r = await client.get(f"/api/v1/documents/{body['id']}/status", headers=auth_headers)
    assert r.status_code == 200
    s = r.json()
    assert s["status"] == "failed"
    assert s["finished"] is True
    assert s["error"] and ("扫描件" in s["error"] or "文字层" in s["error"])


@pytest.mark.asyncio
async def test_status_after_success_ready(
    client: AsyncClient, auth_headers, worker_db, make_pdf
):
    body = await _upload(client, auth_headers, "ok.pdf", make_pdf("Refund in 3 days."))
    await process_document({"job_try": 1}, body["task_id"])

    r = await client.get(f"/api/v1/documents/{body['id']}/status", headers=auth_headers)
    s = r.json()
    assert s["status"] == "ready"
    assert s["task_status"] == "succeeded"
    assert s["chunk_count"] >= 1
    assert s["finished"] is True


@pytest.mark.asyncio
async def test_status_txt_no_task_returns_ready(
    client: AsyncClient, auth_headers
):
    """txt/md 快路径没有任务记录，直接按文档状态回答。"""
    body = await _upload(client, auth_headers, "n.txt", "内容内容内容".encode())
    r = await client.get(f"/api/v1/documents/{body['id']}/status", headers=auth_headers)
    s = r.json()
    assert s["status"] == "ready"
    assert s["task_id"] is None
    assert s["finished"] is True


@pytest.mark.asyncio
async def test_status_cross_user_returns_404(
    client: AsyncClient, auth_headers, make_pdf
):
    """越权查询他人文档状态 → 404，不暴露资源是否存在。"""
    body = await _upload(client, auth_headers, "priv.pdf", make_pdf("Private content."))
    bob = {"Authorization": f"Bearer {await register_and_login(client, 'bob', 'secret123')}"}
    r = await client.get(f"/api/v1/documents/{body['id']}/status", headers=bob)
    assert r.status_code == 404
