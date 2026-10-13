"""ARQ worker：文档解析任务。

启动：
    .venv/bin/arq app.worker.WorkerSettings

★ 与 web 进程的边界（§3.1）：只共享 Redis 和数据库，不共享内存/事件循环。
★ 本阶段最关键的认知（§1）：PDF/DOCX 解析是同步阻塞 CPU 活，必须用
  asyncio.to_thread 丢到线程池，否则会卡死整个 worker 的事件循环——
  而且这个 bug 常规测试测不出来，只有并发冒烟测试（§9.4）能兜住。

确定性失败（扫描件、加密、损坏）→ 置 failed 并 return，绝不 raise（否则 ARQ 反复重试永不成功的文件）。
未知异常（磁盘满、DB 抖动）→ raise TransientError，交给 ARQ 按 max_tries 重试。
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, ClassVar

import arq
from arq import cron
from arq.connections import RedisSettings

from app.clients.cache import shutdown_cache, startup_cache
from app.clients.embedding import embed_documents_if_ready, embedding_client
from app.core.config import settings
from app.crud import document as doc_crud
from app.crud import processing_task as task_crud
from app.db.session import SessionLocal, init_models
from app.services.ingest import split_text
from app.services.parsing import (
    EmptyDocumentError,
    UnsupportedDocumentError,
    extract_text,
)

logger = logging.getLogger("app.worker")


class TransientError(Exception):
    """可重试的临时故障（网络、锁、超时、未知异常）。"""


async def process_document(ctx: dict[str, Any], task_id: int) -> str:
    """解析文档 → 切片入库 → 更新状态。返回值会存进 ARQ 结果里，便于排查。"""
    attempt = ctx.get("job_try", 1)

    # ---------- 第一段事务：读任务、置 running（重活之前先落状态）----------
    async with SessionLocal() as db:
        task = await task_crud.get(db, task_id)
        if task is None:
            logger.error("任务不存在 task_id=%s（可能已被删除）", task_id)
            return "task-not-found"

        # 幂等：已成功的任务不重复处理（重试或重复投递时）
        if task.status == "succeeded":
            logger.info("任务已完成，跳过 task_id=%s", task_id)
            return "already-succeeded"

        await task_crud.mark_running(db, task, attempt=attempt)
        document = await doc_crud.get(db, task.document_id)
        if document is None:
            await task_crud.mark_failed(db, task, "关联文档不存在")
            await db.commit()
            return "document-not-found"

        # ★ 同步把文档状态投影为 processing（验收发现的 bug）：
        #   原先只置任务 running、没动 documents.status，导致轮询接口在整个解析期间
        #   一直返回 status=pending —— 中间态对用户完全不可见。
        #   与 mark_running 同一个事务提交，保证两处状态不会各说各话。
        await doc_crud.set_status(db, document.id, "processing")

        # 提前把需要的字段取出为普通 Python 值：事务外不再碰 ORM
        document_id = document.id
        source_type = document.source_type
        owner_id = document.owner_id
        raw_content = document.raw_content
        await db.commit()

    # ---------- 重活放在事务外（绝不边开着事务边做 CPU）----------
    if not raw_content:
        # store_raw_upload=false 时 worker 拿不到原始文件（§11 已知限制）
        async with SessionLocal() as db:
            task = await task_crud.get(db, task_id)
            if task:
                await task_crud.mark_failed(db, task, "原始文件未持久化，无法解析（请重新上传）")
                await doc_crud.set_status(db, task.document_id, "failed")
            await db.commit()
        return "failed: no-raw-content"

    try:
        # ★ 阶段 3 最关键的一行：同步解析丢到线程池，事件循环保持可调度
        text = await asyncio.to_thread(
            extract_text, raw_content, source_type, settings.max_pdf_pages
        )
    except (EmptyDocumentError, UnsupportedDocumentError) as exc:
        # 确定性失败：重试一万次也不会成功，置 failed 并 return（不 raise → 不触发重试）
        async with SessionLocal() as db:
            task = await task_crud.get(db, task_id)
            if task:
                await task_crud.mark_failed(db, task, str(exc))
                await doc_crud.set_status(db, task.document_id, "failed")
            await db.commit()
        logger.warning("确定性失败 task_id=%s: %s", task_id, exc)
        return f"failed: {type(exc).__name__}"

    except Exception as exc:
        logger.exception("解析异常 task_id=%s", task_id)
        async with SessionLocal() as db:
            task = await task_crud.get(db, task_id)
            if task:
                await task_crud.mark_failed(db, task, f"解析失败：{type(exc).__name__}")
                await db.commit()
        raise TransientError(str(exc)) from exc  # 抛出去让 ARQ 重试

    # ---------- 第二段事务：切片 + 写 chunks + 置成功 ----------
    drafts = split_text(text)
    if not drafts:
        async with SessionLocal() as db:
            task = await task_crud.get(db, task_id)
            if task:
                await task_crud.mark_failed(db, task, "解析出的内容切片后为空")
                await doc_crud.set_status(db, task.document_id, "failed")
            await db.commit()
        return "failed: no-chunks"

    async with SessionLocal() as db:
        task = await task_crud.get(db, task_id)
        if task is None:
            return "task-not-found"
        # 阶段 4：入库时顺带向量化（新增切片不该依赖回填脚本，§5.3）。
        # 模型未就绪时返回 None → 该批切片无向量，仍能被关键词召回（渐进降级）。
        embeddings = await embed_documents_if_ready([d.content for d in drafts])
        # 先删后写：重试可能已写了半截切片，replace 保证不产生重复切片
        await doc_crud.replace_chunks(
            db,
            document_id=document_id,
            owner_id=owner_id,
            raw_text=text,
            drafts=drafts,
            embeddings=embeddings,
        )
        await doc_crud.set_status(db, document_id, "ready")
        await task_crud.mark_succeeded(
            db, task, char_count=len(text), chunk_count=len(drafts)
        )
        await db.commit()

    logger.info(
        '{"event":"parse_done","task":%s,"doc":%s,"chars":%s,"chunks":%s,"attempt":%s}',
        task_id, document_id, len(text), len(drafts), attempt,
    )
    return "succeeded"


async def cleanup_stale_tasks(ctx: dict[str, Any]) -> int:
    """把卡在 running 超时的任务重置为 pending 并重新投递（§11.1）。

    worker 被 kill -9 / 机器重启 → 任务永远停在 running，用户看到永远转圈且无报错。

    ★ 验收修复（原实现的两个缺陷）：
      1. 原实现用 Python 侧 `is_stale` 筛出 id、再用**条数** `stale_ids[:reset]` 切片对账，
         两次判定条件不一致时会漏投/错投；
      2. 原实现没有二次校验，一个"只是慢"的任务也会被重置并重投，
         导致两个 worker 同时解析同一文档。
    现在改为 `claim_stale_running`：条件 UPDATE + RETURNING 原子抢占，
    只对**真正抢到**的 id 重新投递。
    """
    async with SessionLocal() as db:
        claimed_ids = await task_crud.claim_stale_running(
            db, settings.task_stale_seconds
        )
        await db.commit()

    if not claimed_ids:
        return 0

    # 重新投递这些任务（用带时间戳的 job_id，避开旧 job_id 的结果键）
    redis = ctx.get("redis")
    if redis is not None:
        stamp = int(datetime.now(UTC).timestamp())
        for tid in claimed_ids:
            try:
                await redis.enqueue_job(
                    "process_document", tid, _job_id=f"doc:stale:{tid}:{stamp}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("僵死任务重投失败 task_id=%s: %s", tid, exc)

    logger.info("僵死任务清理：重置并重投 %s 条 running 超时任务 %s", len(claimed_ids), claimed_ids)
    return len(claimed_ids)


async def startup(ctx: dict[str, Any]) -> None:
    await init_models()          # 开发期方便；生产应改为 alembic upgrade
    await startup_cache()        # worker 侧若将来要写缓存也需要它
    # ★ 阶段 4：worker 只需 embedder（把切片向量化入库），不需要 reranker（那是查询时的事）。
    #   startup() 同步且会加载模型（首次含下载）；失败不阻断入库——embedding 写不进去时仍走关键词。
    if settings.embedding_enabled:
        try:
            embedding_client.startup()
        except Exception as exc:                        # noqa: BLE001
            logger.warning("worker embedding 模型加载失败（新切片将无向量，仅靠关键词）: %s", exc)
    ctx["started_at"] = datetime.now(UTC)
    logger.info("worker 已启动")


async def shutdown(ctx: dict[str, Any]) -> None:
    await shutdown_cache()
    logger.info("worker 已退出")


class WorkerSettings:
    """ARQ 配置。启动命令：arq app.worker.WorkerSettings"""

    functions: ClassVar[list] = [
        arq.func(
            process_document,
            name="process_document",
            # ★ arq 0.28 的 func() 参数叫 timeout（内部才存成 timeout_s），文档 §6.2 的 timeout_s 会报错
            timeout=settings.worker_job_timeout,
            max_tries=settings.worker_max_tries,
        )
    ]
    cron_jobs: ClassVar[list] = [
        # 每 15 分钟扫一遍僵死任务
        cron(cleanup_stale_tasks, minute={0, 15, 30, 45})
    ]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    on_startup = startup
    on_shutdown = shutdown
    # 单 worker 内并发任务数。CPU 活走 to_thread，线程池默认上限约 32，
    # 这里设小是为了不让多个 PDF 同时抢 CPU 而拖慢彼此。
    max_jobs = settings.worker_max_jobs
    job_timeout = settings.worker_job_timeout
    max_tries = settings.worker_max_tries
    keep_result = 3600           # 结果保留 1 小时，便于排查
    health_check_interval = 60
