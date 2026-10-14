"""评测执行脚本测试（阶段 6，docs/项目实现文档-阶段6.md §5 / §7.1）。

★ 覆盖 §7.1 test_eval_runner 三行：
  - 评测跑通 → 指标可复现：no-judge + keyword 模式，注入种子语料与假 store，全程零模型；
  - judge_validated=false → 自检失败时结果标不可信（非静默通过），退出码 1；
  - --fail-under → 低于阈值非零退出。
  聚合数学还单独用纯函数 _build_result 钉死（不连库、最可复现）。

eval_qa.py 在 scripts/ 包内（已补 __init__.py），故以 `from scripts import eval_qa` 导入。
"""

from types import SimpleNamespace

import pytest

# scripts 已含 __init__.py → 以包形式导入（repo root 在 sys.path 上，pytest 自动注入）
from scripts import eval_qa  # noqa: E402
from scripts.eval_qa import CaseOutcome, _apply_fail_under, _build_result, evaluate  # noqa: E402

from app.models.chunk import Chunk  # noqa: E402
from app.models.document import Document  # noqa: E402


# ---------------------------------------------------------------------------
# 纯函数：聚合数学（§5.2）——最可复现，先钉死
# ---------------------------------------------------------------------------

def _outcome(id_, *, hit, rr, gen=None, passed=True):
    return CaseOutcome(
        id=id_, question="q", tags=[], difficulty="easy",
        retrieval={f"hit@5": hit, f"recall@5": 1.0 if hit else 0.0, "rr": rr},
        generation=gen or {}, passed=passed,
    )


def test_build_result_retrieval_rates_and_mrr():
    outcomes = [
        _outcome("a", hit=True, rr=1.0),
        _outcome("b", hit=False, rr=0.0),
        _outcome("c", hit=True, rr=0.5),
    ]
    res = _build_result(
        outcomes, mode="hybrid", top_k=5, tag=None, no_judge=True, cost_yuan=0.0, judge=None,
    )
    # 命中率 2/3
    assert res["retrieval"]["hit@5"] == pytest.approx(round(2 / 3, 4))
    # MRR = mean(1.0, 0.0, 0.5) = 0.5
    assert res["retrieval"]["mrr"] == pytest.approx(0.5)
    # --no-judge：生成指标不适用、judge_validated=None
    assert res["generation"]["faithfulness"] is None
    assert res["judge_validated"] is None
    assert res["config"]["case_count"] == 3


def test_build_result_generation_means_and_refusal():
    outcomes = [
        _outcome("a", hit=True, rr=1.0, gen={"faithfulness": 1.0, "completeness": 0.5}),
        _outcome("b", hit=True, rr=1.0, gen={"faithfulness": 0.0, "completeness": 1.0}),
        _outcome("c", hit=False, rr=0.0, gen={"refused": True}, passed=True),
        _outcome("d", hit=False, rr=0.0, gen={"refused": False}, passed=False),
    ]
    res = _build_result(
        outcomes, mode="hybrid", top_k=5, tag=None, no_judge=False, cost_yuan=0.0, judge=None,
    )
    assert res["generation"]["faithfulness"] == pytest.approx(0.5)     # mean(1,0)
    assert res["generation"]["completeness"] == pytest.approx(0.75)    # mean(0.5,1.0)
    assert res["generation"]["refusal_accuracy"] == pytest.approx(0.5) # 拒答类 1/2 对
    assert res["judge_validated"] is True


def test_build_result_adds_judge_cost():
    judge = SimpleNamespace(cost_yuan=0.01)
    res = _build_result(
        [_outcome("a", hit=True, rr=1.0, gen={"faithfulness": 1.0})],
        mode="hybrid", top_k=5, tag=None, no_judge=False, cost_yuan=0.02, judge=judge,
    )
    assert res["cost_yuan"] == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# --fail-under：低于阈值 → 非零退出（§5.4）
# ---------------------------------------------------------------------------

def _args(**kw):
    base = dict(top_k=5, fail_under_hit5=None, fail_under_recall=None,
                fail_under_mrr=None, fail_under_faithfulness=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_fail_under_passes_when_above():
    res = {"retrieval": {"hit@5": 0.95, "recall@5": 0.9, "mrr": 0.8},
           "generation": {"faithfulness": 0.9, "completeness": None, "refusal_accuracy": None}}
    assert _apply_fail_under(res, _args(fail_under_hit5=0.90)) == 0


def test_fail_under_fails_when_below():
    res = {"retrieval": {"hit@5": 0.8, "recall@5": 0.8, "mrr": 0.7},
           "generation": {"faithfulness": 0.9}}
    assert _apply_fail_under(res, _args(fail_under_hit5=0.90)) == 1


def test_fail_under_missing_metric_is_not_silent():
    """--no-judge 时没有 faithfulness，却设了阈值 → 必须非零（不能默认放行）。"""
    res = {"retrieval": {"hit@5": 1.0, "recall@5": 1.0, "mrr": 1.0},
           "generation": {"faithfulness": None, "completeness": None, "refusal_accuracy": None}}
    assert _apply_fail_under(res, _args(fail_under_faithfulness=0.9)) == 1


# ---------------------------------------------------------------------------
# 自检失败 → 标不可信 + 退出码 1（§7.1 不要静默通过）
# ---------------------------------------------------------------------------

class _AlwaysFaithfulLLM:
    """坏裁判的来源：无论问什么都判 faithful=true（§1.1 DeepEval 的症状）。"""

    async def chat(self, messages, **kwargs):
        return SimpleNamespace(content='{"faithful": true, "reason": "都对"}', cost_yuan=0.0)


@pytest.mark.asyncio
async def test_evaluate_flags_untrusted_when_self_check_fails(session_factory, monkeypatch):
    monkeypatch.setattr(
        eval_qa, "_load_eval_data",
        lambda: ([{"title": "T", "chunks": ["退款流程说明"]}],
                 [{"id": "c1", "question": "退款", "expected_doc_titles": ["T"]}]),
    )
    store = SimpleNamespace(owner_id=999, title_to_doc_id={"T": 1},
                            seed=lambda db, docs: None, cleanup=lambda db: None)

    result, code = await evaluate(
        no_judge=False, mode="keyword", top_k=5, tag=None,
        store=store, session_factory=session_factory,
        llm=_AlwaysFaithfulLLM(), judge=None,        # judge=None → 内部建 Judge 并自检
    )
    assert code == 1, "自检不过必须用非零退出码拦下"
    assert result["judge_validated"] is False


# ---------------------------------------------------------------------------
# 评测跑通 → 检索指标可复现（no-judge + keyword，零模型）
# ---------------------------------------------------------------------------

class _StoreShim:
    def __init__(self, owner_id, title_to_doc_id):
        self.owner_id = owner_id
        self.title_to_doc_id = title_to_doc_id

    async def seed(self, db, documents):       # owned_store=False，不会被调用
        raise AssertionError("注入 store 时不应触发真实 seed")

    async def cleanup(self, db):
        return None


@pytest.mark.asyncio
async def test_evaluate_no_judge_retrieval_is_reproducible(session_factory, monkeypatch):
    owner_id = 999
    # 直接往库里塞就绪文档 + 切片（StaticPool 跨 session 可见）
    async with session_factory() as s:
        d1 = Document(owner_id=owner_id, title="售后政策", source_type="txt",
                      raw_text="x", status="ready")
        d2 = Document(owner_id=owner_id, title="会员权益", source_type="txt",
                      raw_text="x", status="ready")
        s.add_all([d1, d2])
        await s.flush()
        s.add_all([
            Chunk(document_id=d1.id, owner_id=owner_id, seq=0,
                  content="退款需要 5-7 个工作日到账", char_count=15, embedding=None),
            Chunk(document_id=d2.id, owner_id=owner_id, seq=0,
                  content="黄金会员年费 88 元享 95 折", char_count=15, embedding=None),
        ])
        await s.commit()
        title_map = {"售后政策": d1.id, "会员权益": d2.id}

    monkeypatch.setattr(
        eval_qa, "_load_eval_data",
        lambda: (
            [],   # documents 不被 seed（owned_store=False）
            [
                {"id": "q_refund", "question": "退款 到账",
                 "expected_doc_titles": ["售后政策"], "tags": [], "difficulty": "easy"},
                {"id": "q_member", "question": "会员 年费",
                 "expected_doc_titles": ["会员权益"], "tags": [], "difficulty": "easy"},
            ],
        ),
    )

    res1, code1 = await evaluate(
        no_judge=True, mode="keyword", top_k=5, tag=None,
        store=_StoreShim(owner_id, title_map), session_factory=session_factory,
    )
    res2, code2 = await evaluate(
        no_judge=True, mode="keyword", top_k=5, tag=None,
        store=_StoreShim(owner_id, title_map), session_factory=session_factory,
    )
    assert code1 == 0 and code2 == 0
    # 两条用例都应被关键词命中 → hit@5 满、mrr=1.0，且两次完全一致（可复现）
    assert res1["retrieval"]["hit@5"] == 1.0
    assert res1["retrieval"]["mrr"] == 1.0
    assert res1["retrieval"] == res2["retrieval"]
    assert res1["judge_validated"] is None      # no-judge 不适用
