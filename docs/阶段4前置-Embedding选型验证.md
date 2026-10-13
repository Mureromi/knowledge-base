# 阶段 4 前置：Embedding 方案选型验证报告

> 验证日期：2026-10-13 ｜ 环境：macOS (Apple Silicon) / Python **3.14.5** / 磁盘可用 189GB
> 结论：**采用本地 ONNX 方案（fastembed + BAAI/bge-small-zh-v1.5）**，已实测跑通，无需任何 API key

---

## 0. 为什么必须先做这个验证

阶段 4 的 RAG 依赖向量检索，而向量检索依赖 embedding。但**本项目的 LLM（DeepSeek）没有 embedding 接口**——已实测确认：

```
GET https://api.deepseek.com/models
→ ['deepseek-flash', 'deepseek-v4-pro']      ← 都是对话模型，无 embedding
```

所以 embedding 必须另找来源。这个决定会**同时影响**：成本模型、部署复杂度、数据隐私、检索效果。**拖到阶段 4 当天再定，会阻塞整条进度。**

---

## 1. 结论（先给答案）

| 项 | 选择 |
| --- | --- |
| 方案 | **本地 ONNX（CPU 推理）** |
| 库 | `fastembed` 0.9.0 |
| 模型 | `BAAI/bge-small-zh-v1.5`（中文专用，512 维） |
| 依赖代价 | **142MB，不引入 torch** |
| 是否需要 API key | **不需要** |
| 是否需要 GPU | 不需要（CPU 7.6ms/条，查询 1.3ms） |

**实测跑通的能力**：同义改写命中 4/4、不相关问题相似度显著更低、批量与单条性能可接受。

---

## 2. 四条路线的对比

| 路线 | Python 3.14 可行性 | 依赖体积 | 需要 key | 数据出本机 | 结论 |
| --- | --- | --- | --- | --- | --- |
| **本地 ONNX**（fastembed） | ✅ 已实测跑通 | **142MB**（无 torch） | 否 | 否 | **✅ 采用** |
| 本地 torch（sentence-transformers） | ✅ wheel 可解析（torch 2.14.1） | 约 2–3GB | 否 | 否 | ⚠️ 可行但过重 |
| 云端 embedding API | ✅ 技术上无障碍 | 0 | **是（需新申请）** | **是** | ⏸️ 备选 |
| 不做向量（继续关键词） | — | 0 | 否 | 否 | ❌ 放弃语义检索能力 |

### 2.1 为什么排除 torch 路线

**torch 2.14.1 在 Python 3.14 上有 wheel**（dry-run 确认），所以 `sentence-transformers` 技术上可行——这一点和常见预期相反，3.14 的生态比我预估的好。

但代价是：**2–3GB 依赖 + 更长的安装与冷启动**，而本项目的 embedding 需求是"小规模中文文档 batch 编码"，ONNX 路线**用 142MB 就能达到同样效果**。

> 什么时候才值得上 torch？当你需要**微调**embedding 模型、或需要使用只有 PyTorch 实现的模型时。**阶段 4 不需要。** 保留这个选项在阶段 6（评估与优化）之后再说。

### 2.2 为什么暂不走云端 API

| | |
| --- | --- |
| 硬障碍 | 本机**没有任何可用的 embedding key**（已逐一检查：DashScope / OpenAI / 智谱 / Moonshot / SiliconFlow / Voyage / Jina 全未设置） |
| 额外成本 | 需要新申请一家厂商账号 + 充值 |
| 数据流向 | 你的文档内容要发到第三方 |
| 真正的优势 | 更大维度的模型（1024–4096 维）通常检索效果更好；零运维 |

**结论：本项目是个人知识库，本地模型在效果上已经够用（实测 4/4 命中），而云端方案要多一个账号、多一条数据外流路径、多一份成本。留作阶段 6 做 A/B 对比时的对照组更合适。**

---

## 3. ★ 六个实测结论（两个是配置陷阱）

### 3.1 ★ 陷阱一：`huggingface.co` 在本机完全连不通

```
huggingface.co            → HTTP 000（连接失败）
hf-mirror.com             → HTTP 200 ✅
www.modelscope.cn         → HTTP 200 ✅
```

**所有**测试的模型仓库在 huggingface.co 上都返回 000，在 hf-mirror.com 上都返回 200。

**后果**：不配置镜像的话，`TextEmbedding(...)` 会**卡在下载阶段然后超时**，报一个和"网络"有关的模糊错误。这会在阶段 4 当天浪费你半天时间。

**必须配置**（二选一）：

```bash
# 方式 1：环境变量（推荐，写进 .env 或启动脚本）
HF_ENDPOINT=https://hf-mirror.com
```

```python
# 方式 2：代码里指定（在 import fastembed 之前执行）
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
```

已实测：配置后从镜像下载 5 个文件、约 7 秒完成。

### 3.2 ★ 陷阱二：默认模型缓存会落在系统临时目录，重启即丢

实测发现模型被缓存到：

```
/var/folders/s1/.../T/fastembed_cache/models--Qdrant--bge-small-zh-v1.5/   共 102MB
```

**`/var/folders/.../T/` 是 macOS 的系统临时目录**，重启或系统清理后会消失 → **每次开机第一次使用都要重新下载 100MB**。

**必须显式指定持久化缓存目录**：

```python
TextEmbedding(
    model_name="BAAI/bge-small-zh-v1.5",
    cache_dir="/path/to/project/.model_cache",     # ← 必须指定
)
```

实测指定后：缓存目录 181.6MB，**首次加载 8.68s，二次加载 0.07s**（差 124 倍）。

> 注意实测的缓存大小有差异（102MB vs 181.6MB）：前者是 ONNX 模型主体，后者含 `.locks` 等元数据与未清理的临时文件。**预算按 200MB 算。**

### 3.3 中文提取与语义效果：达标

**同义改写命中率 4/4**（这是向量检索相对关键词检索的**核心价值**）：

| 查询（与目标无字面重叠） | 相似度 | 命中 |
| --- | --- | --- |
| 「退款多久能到账？」 | 0.630 | ✅ |
| 「几天能退钱回来？」 | 0.665 | ✅ |
| 「退款的到账周期是多久」 | 0.650 | ✅ |
| 「钱什么时候退给我」 | 0.591 | ✅ |

目标句是「审核通过后，款项将在 5-7 个工作日退回原支付渠道。」——**关键词检索对这几句全部会失败**（"退款"二字都不出现），这正是向量检索要解决的问题。

**不相关问题不会误命中**：

| 查询 | 目标相似度 | 最高干扰项 |
| --- | --- | --- |
| 「退款多久能到账？」 | **0.630** | 0.576 |
| 「今天天气怎么样？」 | 0.267 | 0.309 |
| 「推荐一部电影」 | 0.191 | 0.237 |

不相关问题的相似度落在 0.19–0.31 区间，与相关问题（0.59–0.67）**有明显分界**。可以据此设相似度阈值过滤噪声。

### 3.4 ★ 用 BGE 官方的检索指令前缀，区分度提升 64%

BGE 系列建议检索时给 **query** 加指令前缀（文档侧不加）。实测三种情况：

| 前缀 | 命中 | **平均区分度** |
| --- | --- | --- |
| 不加前缀 | 4/4 | +0.044 |
| **中文指令**「为这个句子生成表示以用于检索相关文章：」 | 4/4 | **+0.072** ✅ |
| 英文指令「Represent this sentence for searching relevant passages: 」 | 4/4 | +0.049 |

**区分度 = 目标相似度 − 最高干扰项相似度**，越大越不容易排错。

**结论：用中文指令前缀，区分度提升 64%**（+0.044 → +0.072）。这是零成本的检索质量提升，**必须写进实现**。

注意：只加在 query 上，文档编码时不加——加错位置反而会降低效果。

### 3.5 性能：CPU 足够

| 操作 | 实测 |
| --- | --- |
| 批量编码 50 条（每条约 200 字） | 382ms（**7.6ms/条**） |
| 单条查询编码 | **1.3ms** |
| 首次模型加载 | 8.68s |
| 二次加载 | 0.07s |
| 进程峰值内存 | **813MB** |

**内存 813MB 是本方案最需要关注的数字**：web 进程 + worker 进程如果各自加载一份模型，就是 1.6GB。

**建议**：**只在 worker 里加载模型**（embedding 是文档处理时做的），web 进程只负责检索时编码 query。但 query 也要编码……所以两个进程都需要模型。

**折中方案**（推荐）：query 编码做到足够轻（1.3ms），可以接受在 web 进程也加载模型；如果内存吃紧，把 query 编码也做成一次 worker 调用（增加一次 IPC，换内存）。

> 阶段 4 实现时先按"两个进程各加载一份"做，用 `/usr/bin/time -l` 或 `ps` 实测真实占用再决定是否优化。**不要凭猜测优化内存。**

### 3.6 模型选择对比

fastembed 支持的中文/多语言模型（实测 `list_supported_models()`）：

| 模型 | 维度 | 体积 | 说明 |
| --- | --- | --- | --- |
| **BAAI/bge-small-zh-v1.5** | **512** | **0.09GB** | **✅ 本次选用**：中文专用、最轻、效果达标 |
| jinaai/jina-embeddings-v2-base-zh | 768 | 0.64GB | 中英混合更好，体积 7 倍 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 384 | 0.22GB | 多语言，中文非专精 |
| intfloat/multilingual-e5-large | 1024 | 2.24GB | 效果可能更好，但 2.24GB + 慢 |
| BAAI/bge-small-en-v1.5 | 384 | 0.067GB | **英文模型，中文不可用**（别选错） |

**为什么先选最小的**：阶段 4 的目标是**跑通向量的完整链路**，不是追求最好效果。小模型让迭代快（7.6ms/条），等有了真实评测集（阶段 6），再决定是否换更大的模型——那时你会有**可量化的对比数据**，而不是拍脑袋。

> 注意 fastembed 实际拉取的仓库是 `Qdrant/bge-small-zh-v1.5`（Qdrant 做的 ONNX 转换版本），但 API 里用 `BAAI/bge-small-zh-v1.5` 这个名字。这是正常的，不必困惑。

---

## 4. 落地配置（阶段 4 实现时照抄）

### 4.1 依赖

```bash
# 只加 fastembed，它会带上 onnxruntime + tokenizers + numpy
UV_CACHE_DIR="$PWD/.uv-cache" uv add fastembed

# 实测体积：venv 从 ~250MB 增至 ~392MB（+142MB）
# 反复确认：不会引入 torch
```

### 4.2 配置项

```python
# app/core/config.py 新增
# ---- Embedding ----
embedding_enabled: bool = True
embedding_model: str = "BAAI/bge-small-zh-v1.5"
embedding_dim: int = 512                    # 决定 pgvector 列的定义 Vector(512)
embedding_cache_dir: str = ".model_cache"   # ★ 必须持久化，别用系统临时目录（§3.2）
embedding_batch_size: int = 32
# ★ 只加在 query 上，文档侧不加（§3.4）
embedding_query_prefix: str = "为这个句子生成表示以用于检索相关文章："
# 相似度阈值：低于此值视为"没找到相关内容"（依据 §3.3 的 0.31/0.59 分界）
embedding_min_score: float = 0.45
```

`.env` / `.env.example` 同步追加：

```bash
HF_ENDPOINT=https://hf-mirror.com        # ★ 不配则模型下不下来（§3.1）
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_CACHE_DIR=.model_cache
```

`.gitignore` 追加 `.model_cache/`（200MB 不入库）。

### 4.3 客户端骨架

```python
# app/clients/embedding.py
"""Embedding 客户端。唯一的向量出口，业务层不直接碰 fastembed。

★ 关键约束（§3 实测）：
1. HF_ENDPOINT 必须指向镜像，否则模型下载失败
2. cache_dir 必须持久化，否则重启重新下载
3. query 加中文指令前缀，文档不加（区分度 +64%）
4. encode 是同步 CPU 操作，调用方必须 asyncio.to_thread 包装（阶段 3 §1 同一原则）
"""

import asyncio
import logging
import os

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingClient:
    def __init__(self) -> None:
        self._model = None

    def startup(self) -> None:
        """加载模型（同步，约 8.7s 首次 / 0.07s 命中缓存）。在 lifespan 里调一次。"""
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from fastembed import TextEmbedding

        self._model = TextEmbedding(
            model_name=settings.embedding_model,
            cache_dir=settings.embedding_cache_dir,
        )
        logger.info("embedding 模型已加载: %s", settings.embedding_model)

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        """编码文档（不加前缀）。同步 CPU 活 → 丢线程池。"""
        return await asyncio.to_thread(self._embed_sync, texts)

    async def embed_query(self, query: str) -> np.ndarray:
        """编码查询（加中文指令前缀，§3.4）。"""
        prefixed = f"{settings.embedding_query_prefix}{query}"
        vectors = await asyncio.to_thread(self._embed_sync, [prefixed])
        return vectors[0]

    def _embed_sync(self, texts: list[str]) -> np.ndarray:
        assert self._model is not None, "模型未加载：请在 lifespan 中调用 startup()"
        vectors = np.array(
            list(self._model.embed(texts, batch_size=settings.embedding_batch_size))
        )
        # L2 归一化 → 之后用内积即等价余弦相似度，pgvector 里可用 <#> 算子
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)


embedding_client = EmbeddingClient()


async def get_embedding() -> EmbeddingClient:
    return embedding_client
```

> **归一化很重要**：提前 L2 归一化后，pgvector 用内积算子（`<#>`）就等价于余弦相似度，**比每次查询都算余弦快**。这是向量检索的标准做法。

---

## 5. 阶段 4 的三条实现提醒

| # | 提醒 | 依据 |
| --- | --- | --- |
| 1 | **pgvector 必须先装 PostgreSQL** | 本机目前是 SQLite + 无 Docker。`brew install postgresql@16` 或装 Docker Desktop |
| 2 | **embedding 在 worker 里做，不要在 web 里做** | 阶段 3 的 worker 正好可用；但要给 `chunks` 表加 `embedding Vector(512)` 列 + 迁移 |
| 3 | **历史切片需要补算 embedding** | 现有 `chunks` 表已有数据（我灌的演示数据），加列后需要一次性回填脚本 |

### 5.1 回填脚本的思路（阶段 4 会用到）

```python
async def backfill_embeddings(batch_size: int = 32) -> int:
    """为 embedding 为空的历史切片补算向量。可重复执行（幂等）。"""
    # SELECT id, content FROM chunks WHERE embedding IS NULL LIMIT batch_size
    # → embed_documents → UPDATE chunks SET embedding = ...
    # 循环直到没有 NULL
```

**为什么要幂等**：跑到一半失败要能继续，而不是从头再来。

---

## 6. 附：完整实测命令（可复现）

```bash
cd "/Users/mureromi/Documents/FastAPI Learning/knowledge-base"
export UV_CACHE_DIR="$PWD/.uv-cache"

# 1. 建验证环境并装 fastembed（142MB，无 torch）
uv venv --python 3.14 /tmp/emb/.venv
VIRTUAL_ENV=/tmp/emb/.venv uv pip install fastembed

# 2. 确认没引入 torch
VIRTUAL_ENV=/tmp/emb/.venv uv pip list | grep -i torch    # 应无输出

# 3. 配置镜像（不配则失败）
export HF_ENDPOINT="https://hf-mirror.com"

# 4. 跑通编码 + 检索
/tmp/emb/.venv/bin/python -c "
from fastembed import TextEmbedding
m = TextEmbedding(model_name='BAAI/bge-small-zh-v1.5', cache_dir='/tmp/emb/model_cache')
import numpy as np
v = np.array(list(m.embed(['退款要多久到账？'])))
print('维度:', v.shape)          # (1, 512)
"
```

---

## 7. 一句话总结

**本地 ONNX + bge-small-zh-v1.5 已完整跑通，零成本、零 API key、142MB 依赖、CPU 7.6ms/条，同义改写命中 4/4。** 两个必须记住的配置陷阱：**`HF_ENDPOINT` 必须指向 hf-mirror.com**（huggingface.co 本机不通），**`cache_dir` 必须持久化**（默认落在重启即失的系统临时目录）。另外别忘了 **query 加中文指令前缀**，白捡 64% 的区分度提升。
