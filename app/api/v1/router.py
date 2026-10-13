"""v1 API 汇总路由。"""

from fastapi import APIRouter

from app.api.v1.endpoints import auth, chat, documents

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)
