"""问答（RAG）与会话相关 Pydantic 模型。

对应 §6.1 路由总览；阶段 2 加入 session_id 与会话 CRUD 模型。
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    # 阶段 2：为 None 时自动新建会话；续聊必须带上首轮返回的 session_id
    session_id: int | None = Field(default=None, ge=1)


class Citation(BaseModel):
    document_id: int
    document_title: str
    chunk_id: int
    seq: int
    score: float


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_yuan: float = 0.0
    latency_ms: int = 0
    # 阶段 1：本次是否命中答案缓存；命中时 cost_yuan 为 0
    cache_hit: bool = False
    # 阶段 1：本次因命中缓存而省下的钱（仅命中时有值）
    saved_cost_yuan: float | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    usage: Usage
    # 阶段 0 恒为 "keyword"；阶段 4 升级为 "hybrid"，让新旧效果可在响应里对比
    retrieval_mode: str = "keyword"
    # 阶段 2：本轮所属会话（后续请求带上它即可续聊）
    session_id: int


# ---- 会话管理（§6.1 的 sessions CRUD） ----


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime
    last_active_at: datetime


class SessionList(BaseModel):
    total: int
    items: list[SessionOut]


class ChatMessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    cited_chunk_ids: list[int]
    cache_hit: bool
    created_at: datetime


class SessionDetail(BaseModel):
    session: SessionOut
    messages: list[ChatMessageOut]


class SessionRename(BaseModel):
    title: str = Field(min_length=1, max_length=200)
