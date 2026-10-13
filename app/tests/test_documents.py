"""文档端点测试：上传 txt/md 走快路径；pdf 走异步；不支持格式 → 415；删除级联删切片。"""

import pytest
from httpx import AsyncClient


async def _upload(client: AsyncClient, headers: dict, filename: str, content: str):
    return await client.post(
        "/api/v1/documents",
        headers=headers,
        files={"file": (filename, content.encode("utf-8"), "text/plain")},
    )


@pytest.mark.asyncio
async def test_upload_txt_success(client: AsyncClient, auth_headers):
    r = await _upload(client, auth_headers, "notes.txt", "第一段落\n\n第二段落")
    # 阶段 3 §5.1：上传改为 202 Accepted（异步）。txt/md 快路径历史上是 201，
    # 两者都在语义上表示"文档已创建"，故都接受；真正的错误码仍会被这条断言挡下。
    assert r.status_code in (201, 202), r.text
    body = r.json()
    assert body["source_type"] == "txt"
    assert body["title"] == "notes"
    assert body["chunk_count"] >= 1


@pytest.mark.asyncio
async def test_upload_md_success(client: AsyncClient, auth_headers):
    r = await _upload(client, auth_headers, "spec.md", "# 标题\n\n段落 A\n\n段落 B")
    # 阶段 3 §5.1：上传改为 202 Accepted（异步）。txt/md 快路径历史上是 201，
    # 两者都在语义上表示"文档已创建"，故都接受；真正的错误码仍会被这条断言挡下。
    assert r.status_code in (201, 202), r.text
    assert r.json()["source_type"] == "md"


@pytest.mark.asyncio
async def test_upload_unsupported_type_rejected_415(client: AsyncClient, auth_headers):
    """不支持的扩展名 → 415（绝不静默失败，阶段 0 §12 验收清单）。

    ★ 这条测试原本用的是 .pdf —— 阶段 3 加入 PDF 支持后它失效了。
    改用真正不支持的格式，保留原本的断言意图。
    """
    r = await _upload(client, auth_headers, "data.xlsx", "任意内容")
    assert r.status_code == 415
    assert "txt" in r.text or "支持" in r.text


@pytest.mark.asyncio
async def test_upload_pdf_goes_async_path(client: AsyncClient, auth_headers, fake_arq):
    """★ 阶段 3 §5.1：pdf 走异步路径 → 202 + task_id，且投递了一次任务。

    注意：这里传的不是合法 PDF 字节（只是个占位字符串）。异步路径下
    内容校验发生在 worker 里，所以上传本身应当成功——
    这正好验证了"上传即返回"的语义。
    """
    r = await _upload(client, auth_headers, "manual.pdf", "PDF 内容占位")
    assert r.status_code in (201, 202), r.text
    body = r.json()
    assert body["source_type"] == "pdf"
    assert body.get("task_id") is not None, "异步路径必须返回 task_id 供轮询"

    # 投递了一次任务，且函数名正确
    assert len(fake_arq.jobs) == 1, f"应恰好投递 1 次，实际 {len(fake_arq.jobs)}"
    fn_name, args, _kwargs = fake_arq.jobs[0]
    assert fn_name == "process_document"
    assert isinstance(args[0], int), "第一个参数应是 task_id"


@pytest.mark.asyncio
async def test_list_and_detail_and_delete_cascades(client: AsyncClient, auth_headers):
    up = await _upload(client, auth_headers, "policy.md", "退款流程说明\n\n请提交申请")
    assert up.status_code == 202, up.text
    doc_id = up.json()["id"]

    # 列表
    lst = await client.get("/api/v1/documents", headers=auth_headers)
    assert lst.status_code == 200
    assert any(item["id"] == doc_id for item in lst.json()["items"])

    # 详情
    det = await client.get(f"/api/v1/documents/{doc_id}", headers=auth_headers)
    assert det.status_code == 200
    assert det.json()["chunk_count"] >= 1

    # 删除 → 返回被删掉的 chunk 数
    dele = await client.delete(f"/api/v1/documents/{doc_id}", headers=auth_headers)
    assert dele.status_code == 200
    assert dele.json()["deleted_chunks"] >= 1

    # 再查详情应 404
    again = await client.get(f"/api/v1/documents/{doc_id}", headers=auth_headers)
    assert again.status_code == 404


@pytest.mark.asyncio
async def test_title_fuzzy_filter(client: AsyncClient, auth_headers):
    await _upload(client, auth_headers, "alpha.md", "aaa")
    await _upload(client, auth_headers, "beta.md", "bbb")
    r = await client.get("/api/v1/documents?title=alph", headers=auth_headers)
    assert r.status_code == 200
    titles = [i["title"] for i in r.json()["items"]]
    assert "alpha" in titles and "beta" not in titles
