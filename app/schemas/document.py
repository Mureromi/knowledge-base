"""文档相关 Pydantic 模型 + 分页查询参数类型。"""

from datetime import datetime
from typing import Annotated

from fastapi import Query
from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    owner_id: int
    title: str
    source_type: str
    status: str
    created_at: datetime
    chunk_count: int = 0
    # 阶段 3：异步上传时返回刚创建的任务 id，便于前端直接去轮询（同步路径为 None）
    task_id: int | None = None


class DocumentList(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[DocumentOut]


class DocumentDeleteResult(BaseModel):
    id: int
    deleted_chunks: int


class DocumentStatusOut(BaseModel):
    """阶段 3 §5.4：前端轮询用。返回文档状态 + 最新任务进度 + 失败原因。"""

    document_id: int
    # pending / processing / ready / failed（来自 documents.status 投影）
    status: str
    task_id: int | None = None
    # running / succeeded / failed；无任务（txt/md 快路径）时为 None
    task_status: str | None = None
    attempt: int = 0
    # ★ 失败时把原因原文返回（扫描件/加密等），用户需要知道怎么办
    error: str | None = None
    char_count: int = 0
    chunk_count: int = 0
    # 告知前端该不该继续轮询（非终态 = 还要轮）
    finished: bool = True


# 分页 / 过滤查询参数：作为类型别名，方便在 endpoints 直接引用
PageQuery = Annotated[int, Query(ge=1, description="页码，从 1 开始")]
PageSizeQuery = Annotated[int, Query(ge=1, le=100, description="每页条数，1-100")]
TitleQuery = Annotated[str | None, Query(max_length=200, description="标题模糊匹配")]
