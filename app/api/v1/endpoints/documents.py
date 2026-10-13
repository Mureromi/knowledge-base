"""文档端点：上传 / 列表 / 详情 / 删除 / 处理状态轮询。

阶段 3 改造：
- 上传支持 .txt/.md/.pdf/.docx
- txt/md 走**同步快路径**（本就是纯文本，无需排队）→ 202 + status=ready + 已切片
- pdf/docx 走**异步路径** → 202 + status=pending + task_id，后台 worker 解析
- ★ 契约变更：状态码从阶段 2 的 201 改为 **202 Accepted**（实体已建但内容未必就绪）
- 容量防线：上传字节数超 max_upload_bytes → 413，且不落任何库记录
"""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.deps import ArqDep, CurrentUser, DBSession
from app.core.config import settings
from app.core.exceptions import BizError
from app.crud import document as doc_crud
from app.crud import processing_task as task_crud
from app.schemas.document import (
    DocumentDeleteResult,
    DocumentList,
    DocumentOut,
    DocumentStatusOut,
    PageQuery,
    PageSizeQuery,
    TitleQuery,
)
from app.services.ingest import (
    ASYNC_SOURCE_TYPES,
    SYNC_SOURCE_TYPES,
    decode_bytes,
    detect_source_type,
    split_text,
)

router = APIRouter(prefix="/documents", tags=["documents"])


def _to_out(doc, chunk_count: int, task_id: int | None = None) -> DocumentOut:
    return DocumentOut(
        id=doc.id,
        owner_id=doc.owner_id,
        title=doc.title,
        source_type=doc.source_type,
        status=doc.status,
        created_at=doc.created_at or datetime.now(UTC),
        chunk_count=chunk_count,
        task_id=task_id,
    )


@router.post(
    "", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED
)
async def upload_document(
    current_user: CurrentUser,
    db: DBSession,
    arq: ArqDep,
    file: Annotated[
        UploadFile, File(description="支持 .txt / .md / .pdf / .docx")
    ],
) -> DocumentOut:
    filename = file.filename or ""
    source_type = detect_source_type(filename)
    if source_type is None:
        # 明确不支持 → 415，绝不静默失败
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "code": 41501,
                "message": "仅支持 .txt / .md / .pdf / .docx 文件",
                "detail": None,
            },
        )

    raw = await file.read()

    # ★ 容量防线（§5.2）：await file.read() 会把整个文件读进内存，
    #   没有上限=一个 2GB 上传就能 OOM 打挂服务。必须在落库前拦下。
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={
                "code": 41301,
                "message": f"文件超过上限 {settings.max_upload_bytes // 1024 // 1024}MB",
                "detail": None,
            },
        )

    title = filename.rsplit(".", 1)[0] or filename

    # ---------- 快路径：txt/md 直接同步解析 ----------
    if source_type in SYNC_SOURCE_TYPES:
        try:
            text = decode_bytes(raw)
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": 42201, "message": "文本编码无法识别", "detail": None},
            ) from None
        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": 42202, "message": "文件内容为空", "detail": None},
            )
        doc, chunk_count = await doc_crud.create_with_chunks(
            db,
            owner_id=current_user.id,
            title=title,
            source_type=source_type,
            raw_text=text,
            drafts=split_text(text),
            raw_content=raw if settings.store_raw_upload else None,
        )
        return _to_out(doc, chunk_count)

    # ---------- 异步路径：pdf/docx ----------
    if source_type not in ASYNC_SOURCE_TYPES:  # 防御：detect 与分派集合不一致时
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"code": 41502, "message": f"暂不支持解析 {source_type}", "detail": None},
        )

    doc = await doc_crud.create_pending(
        db,
        owner_id=current_user.id,
        title=title,
        source_type=source_type,
        raw_content=raw if settings.store_raw_upload else raw[:0],
    )
    task = await task_crud.create(
        db, document_id=doc.id, owner_id=current_user.id
    )

    # ★ 先落库并提交，再投递任务（§5.3 竞态）：worker 是另一个进程，
    #   若在 INSERT 提交前就开始执行，会查不到任务/文档。顺序绝不能反。
    await db.commit()

    job = await arq.enqueue_job(
        "process_document",
        task.id,
        _job_id=f"doc:{doc.id}:{task.id}",   # 幂等：同一任务不会被投递两次
    )
    if job is not None:
        await task_crud.set_job_id(db, task, job.job_id)
        await db.commit()

    return _to_out(doc, 0, task_id=task.id)


@router.get("/{doc_id}/status", response_model=DocumentStatusOut)
async def get_document_status(
    doc_id: int, current_user: CurrentUser, db: DBSession
) -> DocumentStatusOut:
    """前端轮询用（§5.4）。返回文档状态 + 最新任务进度 + 失败原因。

    越权/不存在统一 404，不暴露资源是否属于他人。
    """
    doc = await doc_crud.get(db, doc_id)
    if doc is None or doc.owner_id != current_user.id:
        raise BizError("文档不存在", code=40401, http_status=status.HTTP_404_NOT_FOUND)

    task = await task_crud.get_latest_for_document(db, doc_id)
    if task is None:
        # txt/md 快路径没有任务记录，直接用文档状态回答
        return DocumentStatusOut(
            document_id=doc.id,
            status=doc.status,
            finished=True,
        )

    return DocumentStatusOut(
        document_id=doc.id,
        status=doc.status,
        task_id=task.id,
        task_status=task.status,
        attempt=task.attempt,
        error=task.error,
        char_count=task.char_count,
        chunk_count=task.chunk_count,
        finished=task.is_terminal,
    )


@router.get("", response_model=DocumentList)
async def list_documents(
    current_user: CurrentUser,
    db: DBSession,
    page: PageQuery = 1,
    page_size: PageSizeQuery = 20,
    title: TitleQuery = None,
) -> DocumentList:
    items, total = await doc_crud.list_by_owner(
        db, owner_id=current_user.id, page=page, page_size=page_size, title_like=title
    )
    # list_by_owner 用 selectinload 拿了 chunks，直接算数量
    out_items = [_to_out(d, len(d.chunks)) for d in items]
    return DocumentList(total=total, page=page, page_size=page_size, items=out_items)


async def _get_owned_document_or_404(db: AsyncSession, doc_id: int, owner_id: int):
    result = await doc_crud.get_with_chunk_count(db, doc_id)
    if result is None:
        raise BizError("文档不存在", code=40401, http_status=status.HTTP_404_NOT_FOUND)
    doc, chunk_count = result
    if doc.owner_id != owner_id:
        # 越权：不暴露资源是否存在，直接 404
        raise BizError("文档不存在", code=40401, http_status=status.HTTP_404_NOT_FOUND)
    return doc, chunk_count


@router.get("/{doc_id}", response_model=DocumentOut)
async def get_document(
    doc_id: int, current_user: CurrentUser, db: DBSession
) -> DocumentOut:
    doc, chunk_count = await _get_owned_document_or_404(db, doc_id, current_user.id)
    return _to_out(doc, chunk_count)


@router.delete("/{doc_id}", response_model=DocumentDeleteResult)
async def delete_document(
    doc_id: int, current_user: CurrentUser, db: DBSession
) -> DocumentDeleteResult:
    doc, _ = await _get_owned_document_or_404(db, doc_id, current_user.id)
    deleted_chunks = await doc_crud.delete_document(db, doc)
    return DocumentDeleteResult(id=doc_id, deleted_chunks=deleted_chunks)
