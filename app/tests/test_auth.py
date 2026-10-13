"""认证端点测试（§9）：
- 重复用户名 409
- 密码错 401
- 无 token 访问受保护端点 401
- token 过期 401
- 超长密码被拒（422）
"""

import pytest
from httpx import AsyncClient

from app.core.security import create_access_token


@pytest.mark.asyncio
async def test_register_success_and_duplicate(client: AsyncClient):
    r = await client.post(
        "/api/v1/auth/register",
        json={"username": "alice", "password": "secret123"},
    )
    assert r.status_code == 201, r.text
    assert r.json()["username"] == "alice"

    r2 = await client.post(
        "/api/v1/auth/register",
        json={"username": "alice", "password": "other-pwd"},
    )
    assert r2.status_code == 409


@pytest.mark.asyncio
async def test_login_wrong_password_401(client: AsyncClient):
    await client.post(
        "/api/v1/auth/register",
        json={"username": "bob", "password": "right-pwd"},
    )
    r = await client.post(
        "/api/v1/auth/token",
        data={"username": "bob", "password": "wrong-pwd"},
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_login_unknown_user_401(client: AsyncClient):
    r = await client.post(
        "/api/v1/auth/token",
        data={"username": "ghost", "password": "whatever"},
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_no_token_returns_401(client: AsyncClient):
    r = await client.get("/api/v1/documents")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_expired_token_returns_401(client: AsyncClient):
    # 直接签一个已过期的 token
    token = create_access_token(subject=1, expires_minutes=-1)
    r = await client.get(
        "/api/v1/documents",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_overlong_password_rejected(client: AsyncClient):
    """超长密码：schema 层 max_length=72 拦截 422；security 层字节检查是兜底。"""
    too_long = "x" * 100
    r = await client.post(
        "/api/v1/auth/register",
        json={"username": "charlie", "password": too_long},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_healthz(client: AsyncClient):
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
