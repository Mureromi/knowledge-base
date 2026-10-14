"""检索指标纯函数测试（阶段 6，docs/项目实现文档-阶段6.md §4.2 / §7.1）。

零 LLM、零 DB——这是"高频快回归"的底座。边界要全：空期望、空排序、命中首位/末位，
以及 §4.2 的核心论点：**MRR 对名次敏感，而 hit@5 不敏感**（阶段 4 rerank 评测的补强）。
"""

import pytest

from app.evals.metrics_retrieval import hit_at_k, recall_at_k, reciprocal_rank


# ---------------------------------------------------------------------------
# hit_at_k
# ---------------------------------------------------------------------------

def test_hit_at_k_first_position():
    assert hit_at_k([7, 3, 9], {7}, 3) is True


def test_hit_at_k_last_within_k():
    assert hit_at_k([3, 9, 7], {7}, 3) is True


def test_hit_at_k_miss_when_beyond_k():
    """命中项排在 k 之外 → 不算命中（Top-k 截断语义）。"""
    assert hit_at_k([3, 9, 5, 7], {7}, 3) is False


def test_hit_at_k_empty_ranked():
    assert hit_at_k([], {7}, 5) is False


def test_hit_at_k_empty_expected():
    assert hit_at_k([7, 3], set(), 5) is False


def test_hit_at_k_multiple_expected_any_hit():
    assert hit_at_k([1, 2, 3], {9, 3}, 3) is True


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

def test_recall_at_k_full_coverage():
    assert recall_at_k([1, 2, 3], {1, 2}, 3) == 1.0


def test_recall_at_k_partial():
    assert recall_at_k([1, 3, 5], {1, 2}, 3) == 0.5


def test_recall_at_k_none_in_k():
    assert recall_at_k([3, 5], {1, 2}, 2) == 0.0


def test_recall_at_k_empty_expected_is_zero():
    """空期望返回 0.0（不是 1.0，也不是除零）——拒答类用例走这条。"""
    assert recall_at_k([1, 2], set(), 5) == 0.0


def test_recall_at_k_respects_k_cutoff():
    # 期望 {1,2}，但 2 排在 k=1 之外 → 只覆盖一半
    assert recall_at_k([1, 2], {1, 2}, 1) == 0.5


# ---------------------------------------------------------------------------
# reciprocal_rank（MRR 单项）
# ---------------------------------------------------------------------------

def test_rr_first_is_one():
    assert reciprocal_rank([7, 3, 9], {7}) == 1.0


def test_rr_second_is_half():
    assert reciprocal_rank([3, 7, 9], {7}) == 0.5


def test_rr_third_is_third():
    assert reciprocal_rank([3, 9, 7], {7}) == pytest.approx(1 / 3)


def test_rr_no_hit_is_zero():
    assert reciprocal_rank([3, 9], {7}) == 0.0


def test_rr_empty_ranked_is_zero():
    assert reciprocal_rank([], {7}) == 0.0


def test_rr_first_match_counts():
    """多命中取第一个命中位置的倒数。"""
    assert reciprocal_rank([3, 7, 5, 9], {7, 9}) == 0.5


# ---------------------------------------------------------------------------
# ★ §4.2 核心论点：同一召回集合、不同顺序 → hit@5 相同但 MRR 不同
#   （这正是 rerank 的专用尺子：rerank 不改召回集合，只改名次）
# ---------------------------------------------------------------------------

def test_mrr_sensitive_to_order_while_hit_at_k_is_not():
    expected = {7}
    order_a = [7, 1, 2, 3, 4]     # 正确答案在第 1 位
    order_b = [1, 2, 3, 4, 7]     # 正确答案在第 5 位（同一召回集合，只是排序变了）

    # hit@5 对二者不敏感（都命中，且都在 Top-5 内）
    assert hit_at_k(order_a, expected, 5) is True
    assert hit_at_k(order_b, expected, 5) is True

    # MRR 却能区分：提前一名，分数就更高
    assert reciprocal_rank(order_a, expected) == 1.0
    assert reciprocal_rank(order_b, expected) == pytest.approx(1 / 5)
