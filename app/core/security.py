"""密码哈希与 JWT。

- 密码：直接用 bcrypt，不用 passlib（passlib 1.7.4 与 bcrypt 5.x 不兼容，见 §1.2）。
- JWT：PyJWT，比 python-jose 更活跃，且对弱密钥会主动告警（见 §1.3）。
"""

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import HTTPException, status

from app.core.config import settings

# bcrypt 的硬限制：超过 72 字节的部分会被静默截断，所以必须显式拦截
MAX_PASSWORD_BYTES = 72


def hash_password(password: str) -> str:
    raw = password.encode("utf-8")
    if len(raw) > MAX_PASSWORD_BYTES:
        # 不要静默截断——那会让超长密码的前 72 字节相同就能登录
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"密码长度不能超过 {MAX_PASSWORD_BYTES} 字节",
        )
    return bcrypt.hashpw(raw, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    raw = password.encode("utf-8")
    if len(raw) > MAX_PASSWORD_BYTES:
        return False
    try:
        return bcrypt.checkpw(raw, hashed.encode("utf-8"))
    except ValueError:
        # 哈希串格式非法（例如库里存了脏数据）
        return False


# ---- JWT ----

def create_access_token(subject: str | int, *, expires_minutes: int | None = None) -> str:
    """签发访问令牌。subject 通常放用户 id。"""
    expire_minutes = expires_minutes or settings.jwt_expire_minutes
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(subject),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expire_minutes)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    """解码并校验 JWT；过期 / 非法签名统一抛 401。"""
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 已过期",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 无效",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
