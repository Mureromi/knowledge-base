"""唯一的向量出口。业务层只调 embed_query / embed_documents，不碰模型本身。

★ 三个必须记住的点（docs/项目实现文档-阶段4.md §1.3 / §6.1）：
1. 环境变量：HF_ENDPOINT 走国内镜像 + **HF_HUB_DISABLE_XET=1**（新版 huggingface_hub
   默认把下载路由到 Xet 后端，hf-mirror 不支持会 401）。必须在 import fastembed 前设好，
   故放在 startup() 里、用 os.environ.setdefault（早于函数内的 import）。
2. cache_dir 必须持久化（默认落系统临时目录，重启即重新下载）。
3. 编码是同步 CPU 活 → **asyncio.to_thread 包装**（阶段 3 §1 同一原则），否则阻塞事件循环。

query 侧加中文指令前缀、文档侧不加（实测区分度 +64%）；输出做 L2 归一化，
之后 pgvector 用内积算子 `<#>` 即等价余弦相似度且更快。
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
        """加载模型（同步，首次约 8.7s 含下载 / 命中缓存约 0.07s）。在 lifespan 里调一次。"""
        # ★ 必须在 import fastembed 之前设好（§1.3）
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from fastembed import TextEmbedding

        self._model = TextEmbedding(
            model_name=settings.embedding_model,
            cache_dir=settings.embedding_cache_dir,
        )
        logger.info("embedding 模型已加载: %s（维度 %s）", settings.embedding_model, settings.embedding_dim)

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """编码文档（不加前缀）。同步 CPU 活 → 丢线程池。"""
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sync, texts)

    async def embed_query(self, query: str) -> list[float]:
        """编码查询（加中文指令前缀，§1.4）。返回单个向量。"""
        prefixed = f"{settings.embedding_query_prefix}{query}"
        vectors = await asyncio.to_thread(self._embed_sync, [prefixed])
        return vectors[0]

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        assert self._model is not None, "模型未加载：请在 lifespan 中调用 startup()"
        vectors = np.array(
            list(self._model.embed(texts, batch_size=settings.embedding_batch_size)),
            dtype=np.float32,
        )
        # L2 归一化 → 之后用内积即等价余弦相似度，pgvector 里可用 <#> 算子
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        return vectors.tolist()


embedding_client = EmbeddingClient()


async def get_embedding() -> EmbeddingClient:
    """FastAPI 依赖；测试用 dependency_overrides 注入假实现。"""
    return embedding_client


async def embed_documents_if_ready(texts: list[str]) -> list[list[float]] | None:
    """入库时顺带向量化（worker / txt-md 快路径共用）。

    ★ 渐进降级：模型未就绪（未 startup / 未下载）或本批编码失败时返回 None，
      调用方则不写 embedding——切片仍能被关键词召回，不会因向量环节失败而阻断入库。
      测试环境不走 lifespan，embedding_client.is_ready 为 False → 直接返回 None。
    """
    if not texts or not settings.embedding_enabled or not embedding_client.is_ready:
        return None
    try:
        return await embedding_client.embed_documents(texts)
    except Exception as exc:                          # noqa: BLE001
        logger.warning("入库向量化失败（降级为无向量，仍走关键词召回）: %s", exc)
        return None
