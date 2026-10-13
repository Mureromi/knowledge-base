"""RRF 融合（Reciprocal Rank Fusion）单元测试。docs/项目实现文档-阶段4.md §9.2 test_rrf。

为什么单测纯函数：RRF 是混合检索"去重 + 融合两路名次"的核心，量纲无关、只排名次，
用纯函数最容易断言手算期望值，不依赖数据库。
"""

from app.services.retrieval import rrf_fuse


def test_rrf_fuses_two_lists_by_expected_order():
    """两路排名融合：手算期望顺序一致；只在一路出现的条目也被保留。"""
    # k=60：list1=[1,2,3]，list2=[2,3,4]
    fused = rrf_fuse([[1, 2, 3], [2, 3, 4]], k=60)
    ids = [cid for cid, _ in fused]

    # chunk 2：list1 名次2 + list2 名次1 → 1/62 + 1/61（最高）
    # chunk 3：list1 名次3 + list2 名次2 → 1/63 + 1/62
    # chunk 1：仅 list1 名次1 → 1/61
    # chunk 4：仅 list2 名次3 → 1/63
    assert ids[0] == 2, "两路都靠前的应融合后最高"
    assert set(ids) == {1, 2, 3, 4}, "只在一路出现的条目也必须被保留"

    # 精确校验分数
    score_of = dict(fused)
    assert score_of[2] == 1 / 62 + 1 / 61
    assert score_of[1] == 1 / 61
    assert score_of[4] == 1 / 63


def test_rrf_dedups_and_boosts_overlap():
    """去重：同一 chunk 在两路都出现 → 只出现一次且 RRF 分高于任何单路分。"""
    fused = rrf_fuse([[7, 8], [7, 9]], k=60)
    ids = [cid for cid, _ in fused]
    assert ids.count(7) == 1, "同一 chunk 只出现一次"
    score_of = dict(fused)
    # 7 在两路都是第 1 名 → 1/61 + 1/61，严格大于只在一路的 8 或 9
    assert score_of[7] == 2 * (1 / 61)
    assert score_of[7] > score_of[8]
    assert score_of[7] > score_of[9]


def test_rrf_one_list_empty_returns_the_other():
    """空列表：一路为空时结果 = 另一路（保持其名次产生的分数顺序）。"""
    fused = rrf_fuse([[], [5, 6, 7]], k=60)
    ids = [cid for cid, _ in fused]
    assert ids == [5, 6, 7]
    score_of = dict(fused)
    assert score_of[5] == 1 / 61  # 名次 1
    assert score_of[6] == 1 / 62  # 名次 2


def test_rrf_both_empty_returns_empty():
    """两路都空返回 []。"""
    assert rrf_fuse([[], []], k=60) == []
    assert rrf_fuse([], k=60) == []
