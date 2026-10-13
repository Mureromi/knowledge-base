# PostgreSQL 本地环境说明（阶段 4 起使用）

> 安装日期：2026-10-13 ｜ 版本：**PostgreSQL 16.15**（Homebrew）
> 为什么需要：**pgvector 是 PostgreSQL 专属扩展**，SQLite 无法支持向量检索。

---

## 一、日常启停

```bash
cd "/Users/mureromi/Documents/FastAPI Learning/knowledge-base"

PG_BIN=/opt/homebrew/opt/postgresql@16/bin

# 启动（★ 必须带 LC_ALL，否则启动失败，见 §三.1）
LC_ALL="en_US.UTF-8" $PG_BIN/pg_ctl -D /opt/homebrew/var/postgresql@16 \
    -l "$PWD/.pg/postgres.log" start

# 查看状态
$PG_BIN/pg_ctl -D /opt/homebrew/var/postgresql@16 status

# 停止
$PG_BIN/pg_ctl -D /opt/homebrew/var/postgresql@16 stop

# 连进数据库
PGPASSWORD=postgres LC_ALL="en_US.UTF-8" \
    $PG_BIN/psql -h 127.0.0.1 -U postgres -d knowledge_base
```

> 也可以让 Homebrew 托管（开机自启）：`brew services start postgresql@16`。
> 注意它需要写 `~/Library/LaunchAgents`，某些受限环境下会失败，那时用上面的 `pg_ctl` 方式。

## 二、连接信息

| 项 | 值 |
| --- | --- |
| 主机 / 端口 | `127.0.0.1:5432` |
| 数据库 | `knowledge_base` |
| 角色 / 密码 | `postgres` / `postgres`（**仅本地开发**，trust 认证） |
| 数据目录 | `/opt/homebrew/var/postgresql@16` |
| 日志 | `.pg/postgres.log`（相对工程根目录） |
| 连接串 | `postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/knowledge_base` |

> ⚠️ **`postgres` 这个角色是安装后手动创建的**。Homebrew 的 initdb 默认只创建与当前系统用户同名的超级用户（`mureromi`）。创建 `postgres` 角色是为了让连接串符合通用习惯（也是文档里一直写的那个 URL 形态）。

---

## 三、★ 三个必须知道的坑（都实际踩过）

### 3.1 启动必须带 `LC_ALL`，否则报 "postmaster became multithreaded"

```
FATAL:  postmaster became multithreaded during startup
HINT:  Set the LC_ALL environment variable to a valid locale.
```

这是 macOS 上的已知问题（与系统库的多线程初始化有关），**跟权限无关**。我一开始误判成沙箱权限问题，浪费了一轮排查。

**解法**：启动命令前加 `LC_ALL="en_US.UTF-8"`。

### 3.2 asyncpg 不会自动序列化 JSON 列

实测报错：

```
invalid input for query argument $5: [] (descriptor 'encode' for 'str' objects doesn't apply to a 'list' object)
```

**根因**：SQLite 是弱类型，list/dict 直接能存；但 asyncpg **不像 psycopg2 那样自动把 Python 对象转成 JSON 字符串**，而 SQLAlchemy 需要引擎上配置 `json_serializer` 才会转换。

**已在 `app/db/session.py` 修掉**（切到非 SQLite 时自动加）：

```python
kwargs["json_serializer"] = lambda obj: json.dumps(obj, ensure_ascii=False)
kwargs["json_deserializer"] = json.loads
```

**影响面**：所有含 `JSON` 列的表——`chat_logs.cited_chunk_ids`、`chat_messages.cited_chunk_ids`。症状是"写入时炸"，读取不受影响。

> 附带教训：**`json_serializer` 只对 ORM/Core 构造的语句生效，对 `text()` 裸 SQL 无效**。所以 `scripts/migrate_sqlite_to_pg.py` 里是手动 `json.dumps` 的。两处都做了，缺一不可。

### 3.3 时区：迁移前后字符串格式不同（不是 bug）

| | `created_at` 读出样子 |
| --- | --- |
| SQLite | `2026-10-13 11:05:13`（naive，无时区标记） |
| PostgreSQL | `2026-10-13T11:37:39.749915Z`（**带 Z，表示 UTC**） |

迁移时我把 SQLite 的 naive 字符串**原样当 UTC 存**（`_coerce` 里不附加时区），所以语义上是一致的。但视觉上前端可能会看到差异——**如果一个时间戳看起来差了 8 小时，先确认它是不是被当成了本地时间**。

---

## 四、数据迁移（已执行完毕）

### 4.1 本次迁移做了什么

```bash
# 1. 建基线 schema（ORM 模型 → create_all）+ 给 Alembic 打标
.venv/bin/python scripts/init_db.py --stamp

# 2. 搬迁数据（保留主键 id、重置自增序列）
.venv/bin/python scripts/migrate_sqlite_to_pg.py --dry-run    # 先看
.venv/bin/python scripts/migrate_sqlite_to_pg.py --truncate   # 再搬
```

**结果**：15 行数据从 `kb.db` 搬到 PostgreSQL，6 张表行数逐一校验一致。

### 4.2 ★ 为什么用 `init_db.py` 而不是 `alembic upgrade head`

**本项目的迁移链不完整**：阶段 0/1 的表（users / documents / chunks / chat_logs）是用 `Base.metadata.create_all()` 建的，**从来没生成过基线迁移文件**。唯一的迁移文件（phase2）只创建 `chat_sessions`/`chat_messages`，它**假设前面那些表已存在**。

所以在新库上直接 `alembic upgrade head` 会失败：

```
FOREIGN KEY(owner_id) REFERENCES users (id)   →  relation "users" does not exist
```

**正确顺序**（`init_db.py --stamp` 内部就是这两步）：

1. 用 ORM 模型建出**完整** schema（模型是事实源，与代码永远一致）
2. `alembic stamp head` 让 Alembic 认为基线已应用，之后的增量迁移能正常接续

> **技术债提示**：理想做法是把基线迁移补齐，让"全新环境 → `alembic upgrade head`"这一条命令就能跑通。当前方案（create_all + stamp）能工作，但**多了一步手动操作**，新人接手容易困惑。建议在阶段 4 加 `Vector(512)` 列时顺手把基线迁移补齐。

### 4.3 两个脚本的用途

| 脚本 | 用途 |
| --- | --- |
| `scripts/init_db.py` | 建表 / 重建（`--drop`）/ 打标（`--stamp`）。新环境初始化用 |
| `scripts/migrate_sqlite_to_pg.py` | 跨库搬数据：自动发现列、保留主键、**重置序列**、行数校验、可重复执行（`--truncate`） |

**`migrate_sqlite_to_pg.py` 里三个关键设计**（都是踩坑后加的）：

1. **类型强制转换**：SQLite 的 `DateTime` 是字符串、`Boolean` 是 0/1，asyncpg 要求真类型，不转就报 `expected a datetime, got 'str'`
2. **重置序列**：PG 的 SERIAL 序列**不会因为显式插入 id 而前进**，不重置的话下一条 INSERT 直接主键冲突。这是数据搬迁最经典的坑
3. **按外键依赖顺序搬**：users → documents → chunks、users → chat_sessions → chat_messages，顺序错了会因外键约束失败

---

## 五、验证清单（本次迁移已全部通过）

- [x] PostgreSQL 16.15 安装并启动，5432 监听正常
- [x] `knowledge_base` 库、`postgres` 角色创建完成
- [x] 基线 schema 建出 7 张业务表 + `alembic_version`（已 stamp）
- [x] 15 行数据搬迁，6 张表行数逐一校验一致
- [x] **序列已重置**：新注册用户拿到 id=3（未撞主键）
- [x] **JSON 列往返正常**：`cited_chunk_ids` 读出是 `list`，`cache_hit` 读出是 `bool`
- [x] 端到端：注册 → 上传 → 检索 → 真实 LLM 问答 → Redis 缓存，全部正常
- [x] 全量测试 **99 passed**（测试仍用内存 SQLite，不依赖 PG）

---

## 六、pgvector 安装（✅ 已完成 2026-10-13）

### 6.1 ★ 关键坑：Homebrew 的 pgvector 不支持 PostgreSQL 16

```bash
brew install pgvector     # 装上了 0.8.7，但……
```

装完后**扩展对 PG 16 不可见**：

```
$ psql -c "SELECT * FROM pg_available_extensions WHERE name='vector';"
（空）
```

**原因**：Homebrew 的 pgvector 只把扩展文件装到了 `postgresql@17` 和 `postgresql@18` 目录下：

```
/opt/homebrew/share/postgresql@17/extension/vector.control
/opt/homebrew/share/postgresql@18/extension/vector.control
```

本机是 **PostgreSQL 16**，所以看不到。这是必踩的版本错配。

### 6.2 ✅ 解法：从源码为 PG 16 编译（不升级数据库）

比"升级到 PG 17/18 再迁数据"风险小得多——**数据库不动，只补一个扩展**。

```bash
cd /tmp
git clone --depth 1 --branch v0.8.7 https://github.com/pgvector/pgvector.git
cd pgvector-build

# ★ 关键：让 pg_config 指向 PG 16，否则会编译到别的版本去
export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"
pg_config --version          # 应输出 PostgreSQL 16.15 (Homebrew)

make && make install
```

编译产物落位（已核对）：

```
vector.dylib                                    → /opt/homebrew/opt/postgresql@16/lib/postgresql/
vector.control + sql/vector--*.sql              → /opt/homebrew/opt/postgresql@16/share/postgresql@16/extension/
vector.h / halfvec.h / sparsevec.h              → .../include/postgresql/server/extension/vector/
```

### 6.3 启用并验证

```bash
psql -h 127.0.0.1 -U postgres -d knowledge_base -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

实测结果：

| 检查 | 结果 |
| --- | --- |
| 扩展版本 | `vector 0.8.7` |
| 余弦距离算子 `<=>` | ✅ 同向量距离 `0.0000` 正确排第一 |
| IVFFlat 索引 | ✅ 可用 |
| HNSW 索引 | ✅ 可用 |

### 6.4 阶段 4 怎么用

```sql
-- 加列（维度必须与 embedding 模型一致：bge-small-zh-v1.5 = 512）
ALTER TABLE chunks ADD COLUMN embedding vector(512);

-- 检索（<=> 是余弦距离，越小越相似）
-- 注意：EmbeddingClient 已做 L2 归一化，所以也可用内积 <#>（更快）
SELECT id, content, embedding <=> :query_vec AS distance
FROM chunks
WHERE owner_id = :uid AND status_ready
ORDER BY embedding <=> :query_vec
LIMIT 5;
```

**索引什么时候建**：数据量小（几千行以内）顺序扫描够快，**先不要建索引**——
`ivfflat`/`hnsw` 都需要调参，参数不对反而降低召回率。等有了真实数据量和评测集（阶段 6）再建。

---

## 七、阶段 4 前置清单（全部完成 ✅）

| # | 项 | 状态 |
| --- | --- | --- |
| 1 | embedding 方案 | ✅ 本地 ONNX（fastembed + bge-small-zh-v1.5，512 维），见 [选型报告](阶段4前置-Embedding选型验证.md) |
| 2 | PostgreSQL | ✅ 16.15 已装、数据已迁移（15 行，序列已重置） |
| 3 | **pgvector** | ✅ **0.8.7 已编译安装并启用** |

**阶段 4 可以开始了。**
