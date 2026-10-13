"""认证相关 Pydantic 模型。"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    # bcrypt 硬上限 72 字节；这里做长度拦截，security.py 里再做字节拦截（defense in depth）
    password: str = Field(min_length=6, max_length=72)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    created_at: datetime


class Token(BaseModel):
    """OAuth2 密码流返回体。"""

    access_token: str
    token_type: str = "bearer"
