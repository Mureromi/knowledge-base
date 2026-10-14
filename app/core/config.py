"""全局配置：环境变量 + .env 加载。

换 LLM 厂商只需改 llm_base_url / llm_api_key / llm_model 三项。
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # 允许 .env 存在未知字段；生产请收紧
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Knowledge Base AI"

    # ---- 日志 ----
    # 级别：DEBUG / INFO / WARNING / ERROR。生产用 INFO，排障临时调 DEBUG。
    log_level: str = "INFO"
    # 单行 JSON（便于 grep/jq 与后续采集）还是人类可读文本。
    # 默认 JSON：现有 13 处 info 日志本来就是 JSON 字符串，合并后能被 jq 直接筛。
    log_json: bool = True
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

    # ---- 阶段 4：向量检索 + 混合检索 + rerank + 多轮查询改写 ----
    # （docs/项目实现文档-阶段4.md §2.2）
    # Embedding（本地 fastembed，零成本零 API key）
    embedding_enabled: bool = True
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512                    # ★ 必须与模型一致，写错只会在查询时报错、难排查
    embedding_cache_dir: str = ".model_cache"   # ★ 必须持久化，别用系统临时目录（重启即失）
    embedding_batch_size: int = 32
    # ★ 只加在 query 上，文档侧不加（实测区分度 +0.044 → +0.072）
    embedding_query_prefix: str = "为这个句子生成表示以用于检索相关文章："

    # 向量检索（混合召回）
    vector_top_k: int = 20          # 向量召回候选数（给融合用，不等于最终上下文条数）
    keyword_top_k: int = 20         # 关键词召回候选数
    rrf_k: int = 60                 # RRF 平滑常数（论文推荐 60）
    final_top_k: int = 5            # 融合+重排后真正进上下文的条数

    # Rerank（本地 bge-reranker-base，模型文件约 1GB → 进程 RSS +801MB）
    # ★ 默认关闭，依据是**自有语料的实测结论**（阶段 4 §15.6，scripts/eval_rerank.py）：
    #     16 条用例 / 18 片语料（刻意含关键词撞车近邻）
    #     Top-1 命中 88% → 94%（+6%）｜ Top-5 命中 100% → 100%（+0%，已饱和）
    #   即：本语料规模下 rerank 只能在"已召回的候选里挪动 #1"，为 +6% 常驻 801MB 不划算。
    # 代码与测试完整保留：语料涨到上千切片、Top-5 不再饱和时，改成 true 或设
    # RERANK_ENABLED=true 即可复测（scripts/eval_rerank.py 会翻转该开关跑对照）。
    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_top_k: int = 10          # 只对前 N 个候选重排（控制延迟）
    rerank_min_candidates: int = 2  # 候选少于这个数就跳过重排（省时间）

    # 多轮查询改写（把「那退货呢？」改写成自包含检索句）
    query_rewrite_enabled: bool = True
    query_rewrite_max_history_turns: int = 3   # 改写时参考的历史轮数（比检索窗口小，够用即可）
    query_rewrite_cache_ttl: int = 3600        # 改写结果缓存 1 小时（同一问题不重复改写）

    # ---- 阶段 6：可观测（Langfuse Cloud，docs/项目实现文档-阶段6.md §3 / §6） ----
    # ★ 默认关：没有账号也必须能跑全部测试；关闭时所有 trace 调用退化为 no-op（§3.2）。
    #   开启后用户问题/切片/回答会发往外部，敏感文档须改自托管（§9 隐私边界）。
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ---- 阶段 6：评测（自写 LLM judge + 规则指标，§4） ----
    eval_cases_path: str = "app/evals/qa_cases.yaml"   # 手标评测集
    eval_judge_model: str = ""          # 留空则复用 llm_model（DeepSeek 自己当裁判）
    eval_judge_temperature: float = 0.0  # ★ 评测必须确定性（§4.3）


settings = Settings()
