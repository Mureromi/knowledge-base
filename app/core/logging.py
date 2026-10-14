"""日志配置。**web 与 worker 两个进程都必须显式调用。**

## 为什么需要这个模块

在此之前 44 处 `logger.*` 调用里，13 处 `logger.info` 是**完全不会输出**的——
root logger 没有配置 handler，级别默认 WARNING。也就是说阶段 0 起精心设计的
结构化事件（`{"event":"chat",...}`、`{"event":"cache_hit",...}`）一直躺在代码里没人看见。

## 两个容易踩的点

1. **worker 是独立进程**：`app.worker` 由 ARQ 启动，不经过 web 的 lifespan，
   所以它必须**自己调一次** `setup_logging()`。只在 main.py 里配是不够的。
2. **不要重复添加 handler**：uvicorn 会预先配置 root handler，
   直接 `basicConfig` 或重复 addHandler 会导致**每条日志打印两遍**。
   本模块先清空自有 handler 再添加，即使被调用多次也安全。
"""

import json
import logging
import os
import sys
from datetime import UTC, datetime

from app.core.config import settings


class JsonLineFormatter(logging.Formatter):
    """单行 JSON 日志。便于 grep、jq 与后续采集（阶段 6 的 Langfuse 也吃这种格式）。

    对"消息本身是 JSON"的情况做**特殊处理**：不是把它塞进字符串字段转义掉，
    而是把它的键**合并到顶层**。因为现有代码里 13 处 info 日志都是这种写法：

        logger.info('{"event":"chat","user":%s,"cost_yuan":%s}', uid, cost)

    合并之后输出的就是规整的一行，能被 `jq -c 'select(.event=="chat")'` 直接筛出来。

    ★ 踩过的坑：`record.getMessage()` 会把 `%s` 替换成参数的 **repr**，
      于是 `%s` 位置的字符串参数带上了 **单引号**（`user:'u1'`），
      整条消息就不再是合法 JSON，`json.loads` 直接失败、合并逻辑静默失效。
      所以这里自己拼：**用参数的替换结果，但字符串参数不加引号**。
    """

    @staticmethod
    def _jsonify(value):
        """把参数转成"内联到 JSON 里合法"的形式。

        ★ 第三个坑：`%s` 渲染布尔会得到 Python 的 `False`/`True`（首字母大写），
          而 JSON 只认 `false`/`true`。所以布尔必须显式转成 JSON 字面量字符串。
          （现有代码里 `"cache_hit":%s` 这种写法很常见，不处理就合并失败。）
        """
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float, str)):
            return value
        return value          # 其他类型交给兜底路径

    @classmethod
    def _render(cls, record: logging.LogRecord) -> str:
        """渲染消息，但**不对字符串参数加引号**，且把布尔转成 JSON 字面量。

        为什么需要它（三次踩坑的结论）：

        1. `record.getMessage()` 用 `%s` 替换参数时走的是 `str()`/`repr()`，
           字符串会带**单引号**（`user:'u1'`），整条消息就不是合法 JSON。
        2. 但也不能反过来给字符串参数**再加双引号** ——
           格式串里本来就写着 `"rewritten":"%s"`，占位符两侧已经有引号了，
           再包一层会变成 `"rewritten":""值""`，同样不是合法 JSON。
        3. 布尔值要被替换成 `true`/`false`，不能是 Python 的 `True`/`False`。

        所以正确做法是：**只做值层面的归一化**（引号交给格式串，布尔转 JSON 字面量），
        然后原样 `%` 替换。
        """
        if not record.args:
            return str(record.msg)
        args = record.args
        try:
            if isinstance(args, dict):
                return str(record.msg) % {k: cls._jsonify(v) for k, v in args.items()}
            if isinstance(args, tuple):
                return str(record.msg) % tuple(cls._jsonify(a) for a in args)
        except (TypeError, ValueError, KeyError):
            pass
        return record.getMessage()      # 兜底：标准渲染（可能带单引号 → 合并失败但不崩）

    def format(self, record: logging.LogRecord) -> str:
        message = self._render(record)
        payload: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
        }

        merged = False
        if message.startswith("{"):
            try:
                parsed = json.loads(message)
                if isinstance(parsed, dict):
                    payload.update(parsed)   # ★ 合并而非转义
                    merged = True
            except json.JSONDecodeError:
                pass
        if not merged:
            payload["msg"] = message

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _silence_onnxruntime_native_noise() -> None:
    """压掉 onnxruntime 的 C++ 层遥测告警。

    ★ 这条消息**绕过 Python logging**（由 C++ 直写 stderr）：

        Failed to persist telemetry device ID; using an in-memory identifier

    所以 `logging.getLogger("onnxruntime").setLevel(ERROR)` 对它无效。
    后果比"吵"更严重：**它会打断 JSON 日志流** ——
    `jq` 遇到这种非 JSON 行会直接**解析中止**，后面的日志全部读不到
    （实测：104 行日志只读出 15 行，`select(.event=="chat")` 返回空，
     一度让我误以为结构化日志没生效）。

    它由 onnxruntime 的环境变量控制（见其 C API 文档）。
    """
    for var in (
        "ORT_DISABLE_TELEMETRY",
        "ORT_TELEMETRY_DISABLED",
    ):
        os.environ.setdefault(var, "1")


def setup_logging(*, force: bool = False) -> None:
    """配置 root logger。web 在模块导入时 + lifespan 各调一次；worker 在 startup 里调。

    force=True 时清空已有 handler（uvicorn 会自带一份），避免重复输出。
    """
    _silence_onnxruntime_native_noise()
    root = logging.getLogger()

    # 幂等保护：重复调用不叠加 handler（否则每条日志打印多次）
    if root.handlers and not force:
        already = any(getattr(h, "_dsh_app_handler", False) for h in root.handlers)
        if already:
            return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler._dsh_app_handler = True          # 自定义标记，用于幂等判断

    if settings.log_json:
        handler.setFormatter(JsonLineFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    if force:
        root.handlers.clear()
    root.addHandler(handler)

    # 让 uvicorn 自己的 logger 也走同一套格式与级别（不 force 的话它会各自为政）
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True                  # 交给 root 统一输出

    # 第三方库降噪：这几个在 INFO 级别下会非常吵
    for name, lvl in (
        ("sqlalchemy.engine", logging.WARNING if not settings.debug else logging.INFO),
        ("httpx", logging.WARNING),
        ("httpcore", logging.WARNING),
        ("arq", logging.INFO),
        ("huggingface_hub", logging.WARNING),
        ("onnxruntime", logging.ERROR),      # 它每次都会打印遥测警告，纯噪声
    ):
        logging.getLogger(name).setLevel(lvl)


def log_startup_banner(component: str) -> None:
    """启动横幅：一眼看出进程、级别、关键开关，排查时非常省事。"""
    logger = logging.getLogger("app.startup")
    logger.info(
        '{"event":"startup","component":"%s","log_level":"%s","log_json":%s,'
        '"db":"%s","redis_enabled":%s,"embedding_enabled":%s,"rerank_enabled":%s}',
        component,
        settings.log_level.upper(),
        str(settings.log_json).lower(),
        settings.database_url.split("@")[-1].split("+")[0],
        str(settings.redis_enabled).lower(),
        str(settings.embedding_enabled).lower(),
        str(settings.rerank_enabled).lower(),
    )
