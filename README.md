# knowledge-base · 个人知识库 AI 问答系统

> 独立项目：本目录自带 `pyproject.toml` 与 `.venv`，与同级的 `fastapi-learning/`（练习项目）互不影响。

代码位于 `app/` 包，是 [docs/项目实现文档-阶段0.md](docs/%E9%A1%B9%E7%9B%AE%E5%AE%9E%E7%8E%B0%E6%96%87%E6%A1%A3-%E9%98%B6%E6%AE%B50.md)、[docs/项目实现文档-阶段1.md](docs/%E9%A1%B9%E7%9B%AE%E5%AE%9E%E7%8E%B0%E6%96%87%E6%A1%A3-%E9%98%B6%E6%AE%B51.md) 与 [docs/项目实现文档-阶段2.md](docs/%E9%A1%B9%E7%9B%AE%E5%AE%9E%E7%8E%B0%E6%96%87%E6%A1%A3-%E9%98%B6%E6%AE%B52.md) 的落地实现。

### 日志（阶段 4 补上了配置）

**默认输出单行 JSON**（`LOG_JSON=true`），因为阶段 0 起代码里就有 13 处结构化事件写法
（`logger.info('{"event":"chat","user":%s,...}', ...)`），JSON 格式能让它们**合并到顶层字段**
而不是被转义成一个字符串——这样才能用 `jq` 直接筛。

```bash
# 按事件类型筛
jq -c 'select(.event=="chat")' app.log
jq -c 'select(.event=="query_rewrite")' app.log

# 统计问答成本与缓存收益
jq -r 'select(.event=="chat") | [.cache_hit, .cost_yuan, (.saved_cost_yuan // 0)] | @tsv' app.log \
  | awk '{n++; if($1=="true") hit++; cost+=$2; saved+=$3}
         END {printf "问答 %d 次｜命中 %d 次｜花费 %.6f 元｜省下 %.6f 元\n", n, hit, cost, saved}'

# 只看错误
jq -c 'select(.level=="ERROR" or .level=="WARNING")' app.log
```

**真实存在的事件类型**：`startup`（启动横幅，含各开关状态）、`chat`（每次问答）、
`query_rewrite`（多轮追问改写）、`parse_done`（worker 解析完成）、`cache_stale`（缓存引用失效）。

配置项：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `LOG_LEVEL` | `INFO` | `DEBUG` 会把 SQLAlchemy 的 SQL 也打出来（很吵，排障时才用） |
| `LOG_JSON` | `true` | `false` 则输出人类可读文本格式 |

> ⚠️ **web 与 worker 是两个进程，日志各自输出**。
> 排查异步任务时要看 worker 的日志（`arq app.worker.WorkerSettings` 那个终端），
> 而不是 web 的。

### 文档导航

| 阶段 | 文档 | 状态 |
| --- | --- | --- |
| 0 | [项目实现文档-阶段0.md](docs/项目实现文档-阶段0.md) | ✅ 已完成 |
| 1 | [项目实现文档-阶段1.md](docs/项目实现文档-阶段1.md) | ✅ 已完成 |
| 2 | [项目实现文档-阶段2.md](docs/项目实现文档-阶段2.md) | ✅ 已完成 |
| 3 | [项目实现文档-阶段3.md](docs/项目实现文档-阶段3.md) | ✅ 已完成 |
| 3 | [阶段3-验收报告.md](docs/阶段3-验收报告.md) | ✅ 已验收（三项修复已补回归测试） |
| **4** | [项目实现文档-阶段4.md](docs/项目实现文档-阶段4.md) | ✅ 已完成：向量 + 混合检索 + RRF + rerank + 查询改写 |
| **4** | [阶段4-验收报告.md](docs/阶段4-验收报告.md) | ✅ 已验收（13/13 通过；发现并修复 4 个问题） |
| 前置 | [阶段4前置-Embedding选型验证.md](docs/阶段4前置-Embedding选型验证.md) | ✅ 方案已定并实测 |
| **5** | [项目实现文档-阶段5.md](docs/项目实现文档-阶段5.md) | ✅ 已完成：Agent / 工具调用（原生 Function Calling 自写循环，工具选择准确率 1.0） |
| **6** | [项目实现文档-阶段6.md](docs/项目实现文档-阶段6.md) | ✅ 已完成：评估体系（自写 judge + 规则指标）+ Langfuse 可观测（基线表见下） |
| 环境 | [PostgreSQL-本地环境.md](docs/PostgreSQL-本地环境.md) | ✅ PG 16.15 + pgvector 0.8.7 |

目标不是"好用"，而是**链路完整**：切分 → 入库 → 检索 → 拼 prompt → 调模型 → 返回带出处。后面每个阶段只替换其中一环。

---

## 目录结构

```
app/
├── main.py                 # FastAPI 入口 + lifespan
├── core/                   # 配置 / 安全（bcrypt + PyJWT）/ 异常
├── db/session.py           # engine / SessionLocal / get_db / init_models
├── clients/                # LLM 出口 + 价格表 + chat_stream；Redis / 内存缓存抽象
├── models/                 # User / Document / Chunk / ChatLog / ChatSession / ChatMessage
├── schemas/                # Pydantic 请求响应（含会话 CRUD）
├── crud/                   # 纯数据访问
├── services/               # ingest / retrieval / qa(流式+非流式) / session / cache_key(v2) / ratelimit / agent(阶段5)
├── tools/                  # 阶段 5：base / calc / now / search_kb（Function Calling）
├── api/v1/                 # 路由：auth / documents / chat（SSE + sync + sessions CRUD + agent）+ deps
└── tests/                  # 260 个用例（LLM + Redis 全 mock，断网也全绿）

仓库根还有 migrations/ + alembic.ini   # 阶段 2 引入：建表已交给 alembic
```

---

## run

```bash
cd "/Users/mureromi/Documents/FastAPI Learning/knowledge-base"

# 1) 装依赖
UV_CACHE_DIR="$PWD/.uv-cache" uv sync --active

# 2) 配密钥
cp .env.example .env
#   无 Docker 就把 DATABASE_URL 改成 sqlite+aiosqlite:///./kb.db
#   有 Docker：docker run -d --name kb-pg -p 5432:5432 \
#     -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=knowledge_base postgres:16
#   填 LLM_API_KEY=sk-... （去 platform.deepseek.com/api_keys 申请）

# 3) 起服务（lifespan 会自动 create_all 建表）
.venv/bin/uvicorn app.main:app --reload --port 8000

# 4) 打开 http://127.0.0.1:8000/docs
#    手测顺序：注册 → 登录拿 token → 点 Authorize 填 token → 上传一个 md → 提问

# 5) 跑测试（LLM 全 mock，断网也全绿）
.venv/bin/python -m pytest -q

# 6) 真实 LLM 冒烟（只需一次，用到 key）
.venv/bin/python -m pytest app/tests/manual -m real -s
```

---

## 阶段 0 验收清单（§12）

| 项 | 状态 | 说明 |
| --- | --- | --- |
| `pytest -q` 全绿，且断网也全绿 | ✅ | 24 passed / 1 deselected（`real` 标记的默认跳过） |
| 真实 LLM 冒烟 + `usage` 回填 `_parse()` | ⏳ | 需要 API key，跑完 `pytest app/tests/manual -m real -s` 后核对字段 |
| 上传 md → 提问 → `[1]` 引用 + `citations` | ✅ | `test_keyword_hit_returns_citations` |
| 提问知识库里没有的内容 → 明确"无法回答" | ✅ | `test_no_relevant_chunk_does_not_call_llm` |
| 用户 A 检索不到用户 B 的文档 | ✅ | `test_search_is_isolated_between_users` |
| 上传 `.pdf` → 415 | ✅ | `test_upload_pdf_rejected_415` |
| `chat_logs` 记录 token / 缓存 / 成本 / 延迟 / 引用 | ✅ | `test_prompt_contains_retrieved_context_and_logs_persist` |
| §10 三条 SQL 能跑并读结果 | ⏳ | 需要 PostgreSQL；SQLite 也能跑，只是 `interval '7 days'` 语法要换 |
| `LLM_API_KEY` 改错 → 502 + 友好提示，非 500 堆栈 | ✅ | `test_llm_error_returns_502_with_friendly_message` |
| 在 README 里记下踩的坑 | ✅ | 就是本文件 |

---

## 踩过的坑

### 1. `passlib` 与 `bcrypt 5.x` 不兼容（§1.2）
网上几乎所有 FastAPI 认证教程都写 `pip install "passlib[bcrypt]"`，但在 Python 3.14 + bcrypt 5.0 环境下会炸：
```
AttributeError: module 'bcrypt' has no attribute '__about__'
```
根因：passlib 最后发布于 2020 年，靠 `bcrypt.__about__.__version__` 判版本，而 bcrypt 4.x 已删除该属性。
**解法**：跳过 passlib，直接 `bcrypt.hashpw` / `bcrypt.checkpw`。见 [core/security.py](./core/security.py)。
**通用教训**：教程里的依赖组合可能已过期，遇到 `AttributeError: module 'x' has no attribute '__about__'`，先查两边的发版时间差。

### 2. `python-jose` vs `PyJWT`：会告警的库更好（§1.3）
两者都能签 JWT。PyJWT 对 HMAC key 长度 <32 字节会打 `InsecureKeyLengthWarning`，python-jose 沉默。
**安全相关的地方，会报错的库比不报错的库好。** 见 [core/security.py](./core/security.py)。

### 3. 系统提示词里塞时间戳 = 成本翻 50 倍（§7.3）
DeepSeek 缓存命中输入价 `0.02 元/M`，未命中 `1 元/M`——**差 50 倍**。
一旦 `SYSTEM_PROMPT` 里写 `f"当前时间：{now}"`，缓存永远失效，成本直接翻 50 倍。
**这不只是优化技巧，是设计约束。** 已加回归测试 [test_system_prompt_is_static_for_cache](./tests/test_chat.py) 兜底：
断言常量里不含 10/13 位数字、ISO 日期，且两次问答 system 消息字节一致。

### 4. `httpx` 客户端必须单例（§6.4）
每请求新建 `AsyncClient` → TLS 握手开销翻倍。
必须放在 `lifespan` 里 startup/shutdown。见 [main.py](./main.py) 和 [clients/llm.py](./clients/llm.py)。

### 5. bcrypt 硬上限 72 字节：不要静默截断
`bcrypt` 对超过 72 字节的密码会**静默截断**——两个不同长密码只要前 72 字节相同就能互相登录。
在 schema（`max_length=72`）+ `hash_password`（字节数检查）两层拦截。见 [schemas/auth.py](./schemas/auth.py) 与 [core/security.py](./core/security.py)。

### 6. FastAPI 的 `yield` 依赖 + `exception_handler` 的回滚时机
`get_db` 用 `yield`，端点里抛 `LLMError` → 被 `register_exception_handlers` 抓住转 502。
直觉上担心：这样 `except Exception: await session.rollback()` 会不会不执行？
实测：FastAPI 在中间件层向 generator `throw()` 异常，rollback 分支照常触发，未 commit 的 ChatLog 不会残留。
**测试**：[test_llm_error_returns_502_with_friendly_message](./tests/test_chat.py) 里 `db.add(ChatLog(...))` 之前才 raise，之后未 add；即使代码路径 add 了也会回滚。

### 7. Pydantic v2 `model_config = ConfigDict(from_attributes=True)`
从 ORM 对象转 schema，v1 的 `class Config: orm_mode = True` 在 v2 变成了 `from_attributes=True`。
本项目的 `UserOut` / `DocumentOut` 都受影响。见 [schemas/](./schemas/)。

### 8. SQLite + `JSON` 列的 `default=list`
SQLAlchemy `JSON` 列 `default=list` 传的是**函数引用**（不是 `[]` 实例），否则所有记录共享同一个 list。
见 [models/chat_log.py](./models/chat_log.py) 里 `cited_chunk_ids`。

---

## 观测：跑起来后能回答的三个问题

对着 `chat_logs` 表：

```sql
-- 1. 这周花了多少钱？平均每次问答多少钱？
SELECT count(*) AS 次数, round(sum(cost_yuan)::numeric, 4) AS 总花费,
       round(avg(cost_yuan)::numeric, 6) AS 平均单价
FROM chat_logs WHERE created_at > now() - interval '7 days';

-- 2. 缓存命中率（验证系统提示词是否真的吃到缓存）
SELECT round(sum(cached_tokens)::numeric / nullif(sum(prompt_tokens), 0), 3) AS 命中率
FROM chat_logs;

-- 3. 平均延迟与最慢
SELECT round(avg(latency_ms)) AS 平均延迟ms, max(latency_ms) AS 最慢ms
FROM chat_logs;
```

SQLite 下第 1 条要改成 `WHERE created_at > datetime('now', '-7 days')`。

---

## 阶段 1 落地：Redis 答案缓存 + 三级限流

对应 [docs/项目实现文档-阶段1.md](docs/%E9%A1%B9%E7%9B%AE%E5%AE%9E%E7%8E%B0%E6%96%87%E6%A1%A3-%E9%98%B6%E6%AE%B51.md)。

### 新增文件

```
clients/cache.py            # CacheBackend 协议 + RedisCache + MemoryCache + startup/shutdown
services/cache_key.py       # 归一化问题 + 检索结果指纹 + 缓存键
services/ratelimit.py       # L1 滑动窗口 / L2 分布式并发槽 / L3 每日配额 + 三段 Lua
```

### 验收清单

| 项 | 状态 | 锚定测试 |
| --- | --- | --- |
| 同一问题两次 → 第二次不调 LLM、`cache_hit=true`、`saved_cost_yuan` 记账 | ✅ | `test_repeated_question_hits_cache_and_skips_llm` |
| 归一化："退款流程" 与 "退款流程。" 共用一条缓存 | ✅ | `test_normalization_lets_punctuation_versions_share_cache` |
| 新文档进 Top-K → 指纹变 → 缓存自动失效 | ✅ | `test_uploading_relevant_doc_invalidates_cache` |
| 被引用的文档被删 → 不返回失效引用 | ✅ | `test_deleting_cited_doc_makes_cache_stale_and_refetches` |
| 命中缓存不吃 L3 每日配额（否则缓存白做） | ✅ | `test_cache_hit_does_not_consume_daily_quota` |
| `CACHE_ENABLED=false` → 每次都调 LLM | ✅ | `test_cache_disabled_calls_llm_every_time` |
| 缓存键包含 `owner_id`，跨用户不共用（防越权泄露） | ✅ | `test_cache_key_is_isolated_between_users` |
| Redis 挂掉 → `get_json` 返回 None 而不是冒泡打挂接口 | ✅ | `test_redis_cache_swallows_read_errors` |
| L1 超限 → 429 + `code:42901` + `Retry-After`；被拒请求不占额度 | ✅ | `test_l1_rejects_when_over_limit` / `test_l1_rejected_request_does_not_consume_slot` |
| L2 并发上限全局生效（多 worker 也能限住），异常/超时必释放，DECR 不变负 | ✅ | `test_l2_enforces_global_limit_under_concurrency` 等 4 个 |
| L3 TTL 到 Asia/Shanghai 次日零点（不是固定 86400，防窗口漂移） | ✅ | `test_l3_writes_key_with_midnight_ttl` |
| 无 Redis 环境下 uvicorn 能起来（自动降级为内存缓存） | ✅ | 冒烟：`Redis 连接失败，降级为内存缓存` |

全量：`.venv/bin/python -m pytest -q` → **69 passed / 1 deselected**。

### 本地体验一遍（推荐 Docker Redis）

```bash
docker run -d --name kb-redis -p 6379:6379 redis:7-alpine
.venv/bin/uvicorn app.main:app --reload --port 8000
# 起另一个终端，注册登录拿 token 后：
curl -s -X POST localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"question":"退款流程"}' | jq .usage
# 第二次同一问题：cache_hit=true，cost_yuan=0.0，saved_cost_yuan 等于第一次真实花费
```

无 Redis 也能跑：`startup_cache()` 会 ping 失败并打 warning，然后落 `MemoryCache`——单进程缓存，接口不受影响。

### 阶段 1 踩过的坑

**a. `fakeredis` 默认不带 Lua，必须 `fakeredis[lua]`。**
`uv add --dev "fakeredis[lua]>=2.39"` 会自动拉 `lupa`。否则 `redis.eval(_SLIDING_WINDOW_LUA, ...)` 直接抛 `no matching Lua interpreter`。

**b. L1 Lua 拒绝分支的返回值不能 `count - 1`。**
一开始想当然把"撤回的那条 -1"，导致 Python 侧 `if count > limit` 永远 false，429 从不触发。**Lua 里撤回靠 `ZREM`，返回值仍是加入后的 count**，Python 才能凭 `> limit` 判定。教训：分布式限流的"值语义"要在脚本注释里写死，不然后人一改就错。

**c. L3 扣配额的位置：入口 vs 未命中之后。**
文档直觉上想放入口。但一旦入口扣，缓存命中的请求也在烧每日额度——**"缓存越准，配额越快用完"** 的荒谬结果。已下沉到 `qa.answer` 未命中分支、真正调 LLM 之前。测试 `test_cache_hit_does_not_consume_daily_quota` 就是这条的回归。

**d. 缓存键必须带 `owner_id`。**
A 上传的私有文档被检索到 → 存进缓存；B 问同一句话，如果键不带 owner_id → 命中 A 的缓存 → **越权读取 A 的文档内容**。见 `test_cache_key_is_isolated_between_users`。这条不只是效率问题，是安全边界。

**e. 只存 `chunk_ids`，不存切片正文。**
好处 1：文档改名后 join 出来的 `document_title` 就是新的，缓存不用手动失效。好处 2：Redis 值体积不会随 chunk content 增长。代价：`_hydrate_citations` 命中时要多一次 DB JOIN（可接受）。

**f. 引用失效必须整体视为未命中。**
如果"尽力而为"返回残缺引用，用户会看到一条指向已删文档的答案。要么按 chunk_ids 全部重取成功、要么 fallthrough 到 LLM 重新生成。

**g. 每日配额 TTL 不能用 86400。**
`expire 86400` 会让窗口随首个请求滚动，第二天实际生效时间可能到 26 号凌晨——用户"额度还剩一半却被拦"。用 `seconds_until_midnight()` 基于 `Asia/Shanghai` 算到次日零点。测试 `test_l3_writes_key_with_midnight_ttl` 锁死这点。

**h. MemoryCache 与限流的耦合：靠 `_extract_redis` 返回 None 放行。**
`MemoryCache` 没有 `.redis` 属性 → L1/L3 检查发现是 None → 直接 return（fail-open）。这样降级链只写一处：`startup_cache` 里 fail 就换 MemoryCache，限流不用另写 if。但生产上必须让 Redis 可用，否则限流形同虚设——已在 `logger.warning` 明确写"生产不可用"。

---

## 阶段 2 落地：SSE 流式输出 + 多轮会话记忆

对应 [docs/项目实现文档-阶段2.md](docs/%E9%A1%B9%E7%9B%AE%E5%AE%9E%E7%8E%B0%E6%96%87%E6%A1%A3-%E9%98%B6%E6%AE%B52.md)。

四个已确认决策：会话存 SQLite/PG 两张表（`chat_sessions` + `chat_messages`）并引入 Alembic；
滑动窗口最近 6 轮；缓存键加窗口历史指纹（v1→v2）；`/chat` 改流式、`/chat/sync` 保留非流式。

### 新增 / 关键改动文件

```
Alembic                        # alembic.ini + migrations/（env.py 从 settings 读 URL、online baseline）
models/chat_session.py         # 会话表（title / last_active_at）
models/chat_message.py         # 消息表（role / content / cited_chunk_ids / cache_hit）
services/session.py            # get_or_create / 窗口加载 / history_fingerprint / append_turn
services/cache_key.py          # v2 键：owner|问题|窗口指纹|检索指纹
clients/llm.py                 # chat_stream() + StreamChunk（reasoning/content/usage/done 分流）
services/qa.py                 # run_answer（非流式）+ run_answer_stream（事件元组流）
api/v1/endpoints/chat.py       # POST /chat(SSE) / /chat/sync / sessions CRUD
api/v1/deps.py                 # get_stream_session_factory（生成器自开 session 的依赖，测试可 override）
```

### SSE 协议（事件顺序）

`meta`（首条带 session_id，前端立刻存）→ `reasoning`*（思考模式才有，前端默认忽略）→ `delta`* → `citations` → `done`；
流中出错发 `error` + `done(incomplete=true)`——HTTP 已 200，状态码改不了（§5.6）。

### 验收清单

| 项 | 状态 | 锚定测试 |
| --- | --- | --- |
| 事件顺序 meta→delta*→citations→done，delta 拼接 = 完整答案 | ✅ | `test_stream_event_order_and_content` |
| `reasoning_content` 绝不混进正文 delta | ✅ | `test_thinking_mode_reasoning_not_in_delta` |
| `: keep-alive` 注释行 / 空行过滤，`[DONE]` 后数据丢弃 | ✅ | `test_chat_stream_filters_keepalive_and_stops_at_done` |
| 命中缓存仍走流式（切块推送），不调流式 LLM | ✅ | `test_stream_cache_hit_skips_llm_but_still_streams` |
| 流结束后 user+assistant 两条消息落库 | ✅ | `test_stream_persists_messages` |
| 流中途 LLMError → error 事件 + done(incomplete)，部分内容照常落库 | ✅ | `test_stream_mid_error_emits_error_event` |
| §3.2 写入顺序：先读窗口再写本轮 → 同问题（空窗口）第二次命中 | ✅ | `test_repeated_question_across_fresh_sessions_hits_cache` |
| 历史不同的两个会话同问题绝不互串（窗口指纹进键） | ✅ | `test_two_sessions_with_different_history_do_not_share_cache` |
| Redis 键全部是 `kb:v2:answer:*`，不再产生 v1 | ✅ | `test_cache_keys_use_v2_namespace_after_multiturn_ask` |
| 会话越权/不存在统一 404（不区分，防探测） | ✅ | `test_session.py` 越权用例 |
| 删除会话：消息级联删、`chat_logs.session_id` SET NULL 保审计 | ✅ | `test_session.py` 删除用例 |
| 追问带历史：messages = system + 窗口 + 本轮 user | ✅ | `test_multiturn_cache` / 冒烟 prompt_tokens 170>156 |
| alembic upgrade/downgrade 可逆，`chat_logs` 数据无损 | ✅ | 冒烟：downgrade -1 → upgrade head，8 条完好 |

全量：`.venv/bin/python -m pytest -q` → **94 passed / 1 deselected**。

### 冒烟（真实 DeepSeek，非测试）

```bash
bash /tmp/smoke_p2.sh   # 起 8199 端口：注册登录→上传退款文档→SSE 首问→同问题再问→带 session 追问→会话详情
```

实测输出：首问 `meta(cache_hit=false)→delta×13→citations→done(cost 0.000208)`；
同问题第二问 `cache_hit=true, saved_cost_yuan=0.000208`；追问"那退货呢？"答出
"退货走同一退款流程，也需要先提交申请"（上下文承接生效，prompt_tokens 含历史 170）。

局限（§8.2）：httpx ASGITransport 会缓冲整流，测试能断言 SSE 全文但**测不出逐块推送**；逐块体验需真实 uvicorn + `curl -N`。

### 阶段 2 踩过的坑

**a. ★ 后台 `uvicorn --reload` 会污染 alembic autogenerate。**
本机一直跑着 reload 服务，存盘模型文件时 lifespan 的 `create_all` 抢先把两张新表建了（空表），`alembic revision --autogenerate` diff 出来只剩 `chat_logs.session_id`——建表语句全丢。修复：手工补齐 `create_table`，`DROP` 空表后重新 `upgrade head`。教训：**跑迁移前先把 reload 服务停掉，或干脆去掉 lifespan 里的 create_all**。

**b. SQLite 不支持 `ALTER TABLE ADD CONSTRAINT`。**
给 `chat_logs.session_id` 加命名 FK 必须用 `op.batch_alter_table('chat_logs')`（重建表复制数据），downgrade 同理。否则 PostgreSQL 能过、SQLite 直接 OperationalError。

**c. 流式落库被静默跳过：usage 事件的 `content` 是空串。**
DeepSeek 流式的 token 统计在尾 chunk，`_parse({"usage":..., "choices":[]})` 出来的 `LLMResult.content=""`，`_persist_miss` 里 `if answer_text.strip()` 为假 → append_turn 不执行（ChatLog 落了但 answer 为空）。修复：把逐块累积的正文与 usage 的 token/成本字段**合并成新的 LLMResult** 再落库。定位手法：monkeypatch `append_turn` 插桩计数，发现从未被调。

**d. ★ StreamingResponse 下请求级 db 的写入对生成器不可见——StaticPool 测不出来。**
端点用请求级 db `get_or_create_session` 建新会话只 `flush`，靠 `get_db` teardown commit；而生成器用 session_factory 另开连接，真实多连接引擎下 SELECT 不到未提交的新行 → 首问 SSE 返回 `error 40402 会话不存在`。测试环境内存 SQLite 用 StaticPool 共享同一连接，flush 即可见，**94 个测试全绿也掩盖了这个 bug**——冒烟才炸出来。修复：端点"只查不建"（传了 session_id 才做越权预检），新会话完全交由生成器在自有 session 里创建+commit，创建与读取同连接。教训：**依赖 yield teardown 与流式生成器的时序不能靠猜，测试连接模型和生产连接模型不同时，必须真机冒烟**。

**e. 文档 §4.2 与 §8.3 自相矛盾的裁决。**
§8.3 要求"同会话连问两次同问题，第二次命中缓存"，但 §4.2 自己承认此时窗口从 0 轮变 1 轮——**历史变了、指纹必须变，未命中才是语义正确的**（强行命中=少算了影响输出的输入=串上下文）。防 §3.2 顺序 bug 的职责改由"跨全新会话同问题命中"承担。以正确性为准写测试，测试文件头注释写明理由。

**f. 中文检索是 2-gram，写测试的文档内容必须含查询词。**
"那退货呢？"的分词含"退货"，上传文档里没这两个字就走无结果分支（不调 LLM，断言全挂）。多处修正文档内容与问题措辞，并在测试里加 `assert len(fake_llm.calls) == N` 防静默走无结果分支。

**g. 删除会话别依赖 ORM cascade。**
异步下删父行触发 `messages` 懒加载 → `MissingGreenlet`。改显式两条 DELETE；`chat_logs.session_id` 靠 FK `ON DELETE SET NULL` 保留审计数据（SQLite 默认不强制 FK，PG 会置空——README 阶段 0 的 pragma 注记在此兑现）。

---

## 升级钩子（§13）

后面每个阶段都是**加文件、换实现**，不是重构：

| 阶段 | 改动位置 | 是否破坏契约 |
| --- | --- | --- |
| 1 Redis + 分布式限流 | 新增 `clients/cache.py`；`services/retrieval.py` 加缓存层；`_llm_semaphore` 换 Redis | 否 |
| 2 流式 SSE + 会话记忆 | `clients/llm.py` 加 `chat_stream()`；新增 `/api/v1/chat/stream`；新增 `chat_sessions` 表 | 新增端点 |
| 3 异步任务 + PDF | `services/ingest.py` 加解析器；接 ARQ；`documents.status` 开始变化 | 否（字段已留） |
| 4 **RAG 真身** | `Chunk` 加 `embedding` 列；`services/retrieval.py` 换实现；`retrieval_mode` 变 `"hybrid"` | 否（签名不变） |
| 5 Agent | 新增 `services/agent.py` + `tools/`；用 DeepSeek Tool Calls | 新增端点 |
| 6 评估 + Langfuse | 新增 `evals/`；用 `chat_logs` 做基线 | 否 |
| 7 Java 接入 | 独立 Spring Boot 工程 HTTP 调本服务 | 否 |

**如果阶段 4 发现要推倒重来，就是阶段 0 的分层没做对。**

---

## 阶段 5 落地：Agent / 工具调用

对应 [docs/项目实现文档-阶段5.md](docs/项目实现文档-阶段5.md)。决策：**原生 Function Calling + 自写循环（约 150 行）**，不引 LangChain / AutoGen。★ 本阶段是**增量非替换**——快速路径 `/chat` 一行不动，新增 `/chat/agent`，用 `chat_logs.is_agent` 统计两条路径。

### 新增文件

```
tools/base.py               # ToolResult + Tool 协议（runtime_checkable）+ openai_schema()
tools/calc.py               # ast 白名单安全求值（绝不用 eval）
tools/now.py                # 当前时间（回答"今天/三天后"这类相对时间）
tools/search_kb.py          # 复用 retrieval.search，按 owner_id 隔离
tools/__init__.py           # build_tools()：search_kb / calc / now
services/agent.py           # AgentRunner：多轮 tool_calls 循环 + 四层保护 + 兜底作答
scripts/eval_agent.py       # 用阶段 6 评测体系量 Agent（工具选择准确率等 4 指标）
app/evals/{agent_cases.yaml,metrics_agent.py}   # 5 例计算/相对时间/多跳 + 纯函数指标
```

首批 3 工具，价值落在快速路径量不到的三类：**计算**（折扣/会员省多少）、**相对时间**（今天/三天后）、**多跳**（先检索政策再算钱）。

### 四层保护（`agent.py`，任一触发即停并兜底作答）

| 保护 | 配置项 | 语义 |
| --- | --- | --- |
| 最大轮次 | `AGENT_MAX_STEPS=6` | 用尽则 `stopped_reason=max_steps`，追加一次无 tools 调用逼模型就现有信息作答 |
| 成本上限 | `AGENT_MAX_COST_YUAN=0.05` | 累计成本超上限立即停，不回灌下一轮 |
| 总超时 | `AGENT_TIMEOUT_SECONDS=60` | 单请求 wall-clock 兜底 |
| 重复调用 | `AGENT_REPEAT_CALL_LIMIT=2` | 同 name+arguments 超阈值 → 拒执行、不记 step、回灌提示并 break（防死循环刷 token） |

### 真实基线（★ 真实 DeepSeek + 真实 PG 语料，5 例）

| 指标 | 值 | 说明 |
| --- | --- | --- |
| 工具选择准确率 | **1.0（5/5）** | 实际调用集合 == 期望集合（严格相等，多调/漏调都算错） |
| 平均步数 | 1.8 | 效率；无来回试 |
| 保护触发率 | 0.0 | 没有用例撞上上限——路径可用 |
| Agent 平均成本 | 0.0014 元 | vs 快速路径 §7.2 SQL 均价 0.000239 元 |
| 成本倍率 | 2.8×（vs 0.0005 基准）/ 实际约 5.9× | 决定 **Agent 不默认开**，按需走 `/chat/agent` |

- 复现：`.venv/bin/python scripts/eval_agent.py --hybrid --fast-path-cost 0.0005 --json app/evals/baseline_agent.json`。
- 迁移 `e5f6a7b8c9d0`（chat_logs 加 `is_agent`/`agent_steps`/`agent_tools` 三列，NOT NULL + server_default 回填）已在真实 PG `alembic upgrade head` 应用。
- 全量：`.venv/bin/python -m pytest -q` → **260 passed / 2 deselected**（阶段 6 的 213 + 本阶段 47，零回归）。

### 阶段 5 踩过的坑

**a. ★ DeepSeek 的 `tool_calls` 在 `choices[0].message.tool_calls`，不在顶层。** 文档 §5 伪代码 `result.raw.get("tool_calls")` 真机取不到。修复：`llm._parse` 把 message 级 `tool_calls`/`reasoning_content` **上浮到 `raw` 顶层**，同时兼容测试里 ScriptedLLM 的顶层写法。

**b. `calc` 遇 `2**100000` 抛 `OverflowError`——独立于 SyntaxError/ValueError/TypeError。** 不显式捕获会逃出工具 → 崩 agent 循环 → 500。修复：计算前限制指数 `abs<=1000`，`except` 显式带 `OverflowError`。

**c. 工具描述必须互斥、写明"不要用"边界。** 实测描述重叠模型会错调（该 calc 却检索）。`test_tool_descriptions_state_when_not_to_use` 机械钉住每个描述含"不要用于/仅当"字样。

**d. 重复调用被拒时不能记 step。** 否则 `tool_counts` 虚高、评测指标失真。被拒只回灌可读提示 + break，未执行的调用不算一次"调用"。

---

## 阶段 6 落地：评估体系 + 可观测性

对应 [docs/项目实现文档-阶段6.md](docs/项目实现文档-阶段6.md)。

### 基线表（★ 真实 embedder + 真实 DeepSeek judge + 真实 PG，24 条手标用例）

| 配置 | hit@5 | mrr | faithfulness | completeness | 拒答准确率 | 成本 |
| --- | --- | --- | --- | --- | --- | --- |
| 纯关键词（阶段 3 行为） | 0.75 | 0.75 | — | — | — | 0 |
| 混合检索（当前） | **1.0** | 0.95 | 1.0 | 0.925 | 1.0 | ≈0.052 元 / 24 条 |
| 混合 + rerank | 1.0 | **0.975** | — | — | — | 0 |

- **向量检索把 hit@5 从 0.75 → 1.0**：4 条"同义改写"用例全靠向量救回（纯关键词全部 miss）。
- **rerank 只提升 MRR（0.95 → 0.975）而 hit@5 早已饱和**——这正是阶段 4"只看 hit@k 得出 +0%"看不到的那部分，也是本阶段引入 MRR 的意义。
- 复现：`.venv/bin/python scripts/eval_qa.py --mode keyword --no-judge` / `--mode hybrid`；rerank 对比加 `RERANK_ENABLED=true`。原始结果见 `app/evals/baseline_hybrid.json`。

### 关键取舍

- **自写 LLM judge + 强制自检**：DeepEval 忠实度指标在同一组已知坏答案上只判对 1/4（§1）——未经验证的尺子比没有尺子更危险。每次评测前 `validate_judge()` 用 1 好 3 坏自检，不过则结果标 `judge_validated=false` 并非零退出。
- **Langfuse 是旁路增强，非硬依赖**：`LANGFUSE_ENABLED=false`（默认）时 `trace_span` 全链路 no-op、不联网、不改行为；`flush()` 在 web 与 worker 的 shutdown 里都调（否则最后的 trace 丢失）。
- 全量：`.venv/bin/python -m pytest -q` → **213 passed / 2 deselected**（2 个 `real` 需真实 Key）。
