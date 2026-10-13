"""Rerank 客户端：cross-encoder 精排（fastembed 本地模型）。

与 embedding 的三点不同（docs/项目实现文档-阶段4.md §6.2）：
1. 输入是 (query, 候选文档) 对，输出是相关性分数（不是向量）
2. 必须成对计算，N 个候选就是 N 次推理 —— 所以只对 Top-K 精排，不能全量
3. 分数是 logit 经 sigmoid 的值（本模型实测约 -11 ~ +5），**只能用于排序，绝不能设阈值**
   （"低于 0.3 丢弃"会把所有负分全丢掉）。要过滤相关性请用向量相似度设阈值。

★ 分数顺序与输入 documents 顺序一致（fastembed rerank 生成器按序产出）。
★ 同步 CPU 活 → asyncio.to_thread 包装（与 embedding 同一原则）。
"""

import asyncio
import logging
import os

from app.core.config import settings

logger = logging.getLogger(__name__)


class RerankClient:
    def __init__(self) -> None:
        self._model = None

    def startup(self) -> None:
        """加载模型（约 1GB，首次较慢）。在 lifespan 里调一次。

        ★ 容错：模型下载/加载失败时不抛出、只记日志，_model 保持 None。
          检索层会在 rerank() 抛错时降级为 RRF 顺序（§9.2），问答不受影响。
          这也是"内存吃紧可整体关掉 rerank"（§2.3.3）之外的第二道保险。
        """
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(
                model_name=settings.rerank_model,
                cache_dir=settings.embedding_cache_dir,   # 与 embedding 共用缓存目录
            )
            logger.info("rerank 模型已加载: %s", settings.rerank_model)
        except Exception as exc:                            # noqa: BLE001
            logger.warning("rerank 模型加载失败（检索将降级为 RRF 顺序）：%s", exc)
            self._model = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """返回与 documents 等长的相关性分数。同步 CPU 活 → 线程池。"""
        if not documents:
            return []
        return await asyncio.to_thread(self._rerank_sync, query, documents)

    def _rerank_sync(self, query: str, documents: list[str]) -> list[float]:
        if self._model is None:
            raise RuntimeError("rerank 模型未加载：请在 lifespan 中调用 startup()")
        # fastembed 的 rerank 是生成器，逐个产出分数，顺序与输入一致
        return [float(s) for s in self._model.rerank(query, documents)]


rerank_client = RerankClient()


async def get_reranker() -> RerankClient:
    """FastAPI 依赖；测试用 dependency_overrides 注入假实现。"""
    return rerank_client
