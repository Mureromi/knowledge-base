"""检索指标：纯规则计算，不需要 LLM，可高频跑（阶段 6，docs/项目实现文档-阶段6.md §4.2）。

★ 为什么检索指标与生成指标必须分开（§4.1）：
  RAG 效果差有两种完全不同的原因——**召回不到**（检索问题）vs **召回到了但答歪**
  （生成问题）。混在一起看总分，无法判断该优化哪一环。

★ `reciprocal_rank` 是本阶段对阶段 4 方法论的补强：
  阶段 4 的 rerank 评测只看 hit@1 / hit@5，得出 "+6% / +0%"——但 rerank 不改变
  召回集合、只改变**名次**，hit@5 对它天然不敏感（Top-5 饱和时恒为 +0%）。
  MRR 能量化"正确答案被提前了多少名"（§8 验收项：能回答"rerank 开启后 MRR 提升了多少"）。
"""


def hit_at_k(ranked_ids: list[int], expected_ids: set[int], k: int) -> bool:
    """Top-k 里是否命中任一期望文档。"""
    return bool(set(ranked_ids[:k]) & expected_ids)


def recall_at_k(ranked_ids: list[int], expected_ids: set[int], k: int) -> float:
    """Top-k 覆盖了多少比例的期望文档（期望多条时比 hit 更细，跨文档用例专用）。"""
    if not expected_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & expected_ids) / len(expected_ids)


def reciprocal_rank(ranked_ids: list[int], expected_ids: set[int]) -> float:
    """第一个命中位置的倒数（MRR 的单项）。

    ★ 对名次敏感、对召回集合不敏感——正是 rerank 的专用尺子。
    """
    for i, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in expected_ids:
            return 1.0 / i
    return 0.0
