"""认证端点：注册 / 登录（OAuth2 密码流）。"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.exc import IntegrityError

from app.api.v1.deps import DBSession
from app.core.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from app.crud import user as user_crud
from app.schemas.auth import Token, UserCreate, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate, db: DBSession) -> UserOut:
    """注册新用户；用户名重复 → 409。"""
    existing = await user_crud.get_by_username(db, payload.username)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": 40901, "message": "用户名已存在", "detail": None},
        )

    password_hash = hash_password(payload.password)
    try:
        user = await user_crud.create(
            db, username=payload.username, password_hash=password_hash
        )
    except IntegrityError:
        # 并发下 unique 约束兜底
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": 40901, "message": "用户名已存在", "detail": None},
        ) from None
    return UserOut.model_validate(user)


@router.post("/token", response_model=Token)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: DBSession,
) -> Token:
    """OAuth2 密码流：username + password 表单，返回 JWT。"""
    user = await user_crud.get_by_username(db, form_data.username)
    # 用户名不存在与密码错误统一 401，避免用户枚举
    if user is None or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": 40101, "message": "用户名或密码错误", "detail": None},
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(subject=user.id)
    return Token(access_token=access_token)
