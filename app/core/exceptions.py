"""业务异常 + 全局异常处理器。

统一错误体：`{"code": 40401, "message": "...", "detail": null}`
"""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.clients.llm import LLMError

logger = logging.getLogger(__name__)


class BizError(Exception):
    """业务层可控异常：知道原因、能给前端友好提示的都用它。"""

    def __init__(self, message: str, code: int = 40000, http_status: int = 400):
        super().__init__(message)
        self.message = message
        self.code = code
        self.http_status = http_status


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(BizError)
    async def _biz(request: Request, exc: BizError):
        return JSONResponse(
            status_code=exc.http_status,
            content={"code": exc.code, "message": exc.message, "detail": None},
        )

    @app.exception_handler(LLMError)
    async def _llm(request: Request, exc: LLMError):
        # 401 这类是"我们自己的配置错了"，不是上游故障。若统一按 502 返回，
        # 排障时会被误导成"上游服务挂了"，方向完全错。
        if exc.status_code in (401, 403):
            logger.error("LLM 鉴权失败（检查 LLM_API_KEY）: %s", exc)
            return JSONResponse(
                status_code=status.HTTP_502_BAD_GATEWAY,
                content={
                    "code": 50202,
                    "message": "AI 服务鉴权失败，请检查服务端 API Key 配置",
                    "detail": None,
                },
            )

        if exc.status_code == 429:
            logger.warning("LLM 触发限流: %s", exc)
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "code": 50301,
                    "message": "AI 服务繁忙，请稍后重试",
                    "detail": None,
                },
            )

        # 超时、5xx、网络错误等：确实可以归为"暂时不可用"
        logger.error("LLM 调用失败: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "code": 50201,
                "message": "AI 服务暂时不可用，请稍后重试",
                "detail": None,
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.exception("未处理异常")
        return JSONResponse(
            status_code=500,
            content={"code": 50000, "message": "服务器内部错误", "detail": None},
        )
