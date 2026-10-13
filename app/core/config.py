"""全局配置：环境变量 + .env 加载。

换 LLM 厂商只需改 llm_base_url / llm_api_key / llm_model 三项。
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # 允许 .env 存在未知字段；生产请收紧
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Knowledge Base AI"
    debug: bool = True

    # SQLite 也能跑（aiosqlite），阶段 4 前迁到 PostgreSQL
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/knowledge_base"
    )

    # JWT：PyJWT 会警告 <32 字节的密钥，这里默认给足长度
    jwt_secret: str = "dev-only-secret-please-change-me-32bytes+"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # ---- LLM：换厂商只改这几项 ----
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = ""
    llm_model: str = "deepseek-flash"
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3
    llm_max_tokens: int = 1024

    # ---- 思考模式（DeepSeek 专有，其他厂商可忽略）----
    # DeepSeek 默认开启思考模式且 effort=high。实测同一 RAG 问答 prompt：
    #   开启 → completion=187(思考占 151) 1733ms ≈0.00188元，答案正确
    #   关闭 → completion=36            703ms ≈0.00049元，答案同样正确
    # RAG 问答是"从给定资料提取总结"，推理已在检索阶段完成，故默认关闭。
    # 注意：思考模式下 temperature 会被静默忽略（官方文档明确说明）。
    llm_thinking: bool = False
    llm_reasoning_effort: str = "low"   # 仅 llm_thinking=True 时生效：low/high/max

    # ---- RAG 参数（阶段 4 调优的起点）----
    chunk_size: int = 500
    chunk_overlap: int = 80
    retrieval_top_k: int = 5
    max_context_chars: int = 6000

    # ---- 阶段 1：Redis 缓存 + 多级限流（docs/项目实现文档-阶段1.md §2.2） ----
    redis_url: str = "redis://localhost:6379/0"
    redis_enabled: bool = True          # 想临时关掉缓存做对比测试时置 false
    redis_socket_timeout: float = 1.0   # 超时要短！Redis 慢不能拖垮请求

    # 答案缓存
    cache_enabled: bool = True
    cache_ttl_seconds: int = 86400      # 24 小时
    cache_key_prefix: str = "kb"

    # 多级限流
    # L1 每用户请求频率（滑动窗口）
    rate_limit_per_minute: int = 30
    rate_limit_window_seconds: int = 60
    # L2 全局并发 LLM 数（分布式信号量，保护上游）
    llm_max_concurrency: int = 10
    # L3 每用户每日 LLM 调用配额（成本熔断，仅未命中时扣）
    daily_llm_quota: int = 200
    # Redis 故障时的策略：true=放行（可用性优先）；false=拒绝（成本优先）
    ratelimit_fail_open: bool = True

    # ---- 阶段 2：会话记忆 + 流式输出（docs/项目实现文档-阶段2.md §7） ----
    # 会话：滑动窗口按"轮"计（1 轮 = user + assistant 两条消息）
    session_window_turns: int = 6
    session_title_max_chars: int = 200

    # 流式
    llm_stream_enabled: bool = True
    # 流式默认不重试（§5.2）：一旦有 chunk 推给客户端就不能重试，
    # 对用户来说"等重试"比"报错重问"体验更差，故只建连一次
    llm_stream_max_retries: int = 1
    # 缓存命中时把整段答案切块推送的手感参数（§5.4）
    stream_cached_chunk_chars: int = 24

    # ---- 阶段 3：上传异步化 + PDF/DOCX 解析（docs/项目实现文档-阶段3.md §8.2） ----
    # 上传与解析
    max_upload_bytes: int = 20 * 1024 * 1024    # 20MB：await file.read() 全进内存，没有上限=一个 2GB 上传就能打挂服务
    max_pdf_pages: int = 500                    # 防一个 5000 页 PDF 拖垮 worker
    allowed_source_types: tuple[str, ...] = ("txt", "md", "pdf", "docx")

    # 原始文件存储：异步 worker 解析时必须还能拿到上传的 bytes，
    # 进程内存存不了（§3.1），故落库（LargeBinary）。想换对象存储时只改这一处开关。
    store_raw_upload: bool = True

    # Worker
    worker_max_jobs: int = 4        # 单 worker 并发任务数：CPU 活走 to_thread，设太高会互相抢 CPU
    worker_job_timeout: int = 300   # 单任务超时（秒）
    worker_max_tries: int = 3       # 仅“可重试异常”会走到这里（§6.3 ②）
    task_stale_seconds: int = 600   # 卡在 running 超过此时长 → 视为僵死，由 cron 重置重投


settings = Settings()
