"""并发冒烟测试（阶段 3 §9.4）。

阶段 3 唯一能证明"CPU 解析没阻塞事件循环"的测试：
把真正的解析换成 time.sleep(0.3) 的同步桩，并发跑两个任务——
若实现走了 asyncio.to_thread，总耗时 ≈ 0.3s；若漏了（同步直接解析），串行 ≈ 0.6s。
常规功能测试永远测不出这个 bug，所以单独有一条把它变成可断言的性能正确性。"""

import asyncio
import time

import pytest
from httpx import AsyncClient

from app.worker import process_document


@pytest.fixture
def worker_db(monkeypatch, session_factory):
    monkeypatch.setattr("app.worker.SessionLocal", session_factory)
    return session_factory


@pytest.mark.asyncio
async def test_concurrent_tasks_do_not_block_each_other(
    client: AsyncClient, auth_headers, worker_db, make_pdf, monkeypatch
):
    async def _upload():
        r = await client.post(
            "/api/v1/documents", headers=auth_headers,
            files={"file": ("c.pdf", make_pdf("Some refund policy text."), "application/octet-stream")},
        )
        return r.json()["task_id"]

    id1 = await _upload()
    id2 = await _upload()

    # 纯同步 sleep，放大"阻塞 vs 不阻塞"的差异
    def slow_extract(raw, source_type, max_pages):
        time.sleep(0.3)
        return "退款流程：提交申请后 3 个工作日审核。"

    monkeypatch.setattr("app.worker.extract_text", slow_extract)

    t0 = time.perf_counter()
    await asyncio.gather(
        process_document({"job_try": 1}, id1),
        process_document({"job_try": 1}, id2),
    )
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.5, (
        f"两个任务耗时 {elapsed:.2f}s，接近串行——说明解析没有走 asyncio.to_thread，"
        "事件循环被阻塞了（阶段 3 §1）"
    )
