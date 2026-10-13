"""ORM 模型包。

★ 必须在这里汇总导入所有模型：Alembic autogenerate 只看得见已导入的表，
漏导入不会报错，只会静默生成空迁移（阶段 2 §2.4）。
"""

from app.models.base import Base
from app.models.chat_log import ChatLog
from app.models.chat_message import ChatMessage
from app.models.chat_session import ChatSession
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.processing_task import ProcessingTask
from app.models.user import User

__all__ = [
    "Base",
    "User",
    "Document",
    "Chunk",
    "ChatLog",
    "ChatSession",
    "ChatMessage",
    "ProcessingTask",
]
