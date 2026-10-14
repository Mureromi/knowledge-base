"""全量评测：检索指标（零成本）+ 生成指标（LLM judge）（阶段 6，docs/项目实现文档-阶段6.md §5）。

用法：
    .venv/bin/python scripts/eval_qa.py                          # 混合检索 + judge
    .venv/bin/python scripts/eval_qa.py --no-judge               # 只跑检索指标（零 LLM 成本）
    .venv/bin/python scripts/eval_qa.py --mode keyword           # 纯关键词（阶段 3 行为，建基线对比）
    .venv/bin/python scripts/eval_qa.py --tag 同义改写            # 只跑某类
    .venv/bin/python scripts/eval_qa.py --json out.json          # 机器可读输出（§5.2）
    .venv/bin/python scripts/eval_qa.py --fail-under-hit5 0.90   # 低于阈值非零退出（CI 拦退化，§5.4）

★ 与生产问答流水线的关系：**绕开缓存与会话层**，直接组合 search → build_messages →
  llm.chat → judge。原因：
  1. 答案缓存会让 rerank on/off 两趟返回同一个缓存答案，A/B 对比直接失效（阶段 4 同款教训）。
  2. 会话/限流/落库与"测量检索+生成质量"无关，评测要的是可复现的最小闭环。
  prompt 与生成参数则**复用 qa.py 的同一实现**（单一事实源，评的才是线上真正跑的）。

★ judge_validated（§5.2）：跑正式评测前先用已知坏答案自检裁判（§1 的教训——
  未经验证的指标比没有指标更危险）。自检不过 → 结果标记不可信并**非零退出**。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from sqlalchemy import delete

from app.clients.embedding import embedding_client
from app.clients.llm import LLMClient, llm_client
from app.clients.rerank import rerank_client
from app.core.config import settings
from app.db.session import SessionLocal, engine
from app.evals.judge import Judge, validate_judge
from app.evals.metrics_retrieval import hit_at_k, recall_at_k, reciprocal_rank
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.user import User
from app.services import retrieval
from app.services.qa import NO_RESULT_ANSWER, build_messages

logger = logging.getLogger("eval_qa")

EVAL_USERNAME = "__eval_qa__"
# 拒答判定：System prompt 要求回"根据现有资料无法回答"，检索为空则回 NO_RESULT_ANSWER。
REFUSAL_MARKERS = ("无法回答", "没有找到", "无法提供")


@dataclass
class CaseOutcome:
    id: str
    question: str
    tags: list[str]
    difficulty: str
    retrieval: dict[str, Any]                    # hit@k / recall@k / rr
    generation: dict[str, Any] = field(default_factory=dict)
    passed: bool = True
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 语料装载（与 eval_rerank.py 同一隔离策略：独立评测用户，级联删除）
# ---------------------------------------------------------------------------

class CorpusStore:
    """把 qa_cases.yaml 里的 documents 灌入库（带真实向量），并维护标题→id 映射。"""

    def __init__(self) -> None:
        self.owner_id: int | None = None
        self.title_to_doc_id: dict[str, int] = {}

    async def seed(self, db, documents: list[dict]) -> None:
        user = User(username=EVAL_USERNAME, password_hash="eval-only-not-a-real-login")
        db.add(user)
        await db.flush()
        self.owner_id = int(user.id)

        texts = [c for doc in documents for c in doc["chunks"]]
        vectors = await embedding_client.embed_documents(texts)
        it = iter(vectors)
        for doc in documents:
            row = Document(
                owner_id=self.owner_id, title=doc["title"], source_type="txt",
                raw_text="\n".join(doc["chunks"]), status="ready",
            )
            db.add(row)
            await db.flush()
            self.title_to_doc_id[doc["title"]] = int(row.id)
            for seq, content in enumerate(doc["chunks"]):
                db.add(Chunk(
                    document_id=row.id, owner_id=self.owner_id, seq=seq,
                    content=content, char_count=len(content), embedding=next(it),
                ))
        await db.commit()

    async def cleanup(self, db) -> None:
        if self.owner_id is not None:
            # 删评测用户即可，documents/chunks 靠 ON DELETE CASCADE 级联清理
            await db.execute(delete(User).where(User.id == self.owner_id))
            await db.commit()
            self.owner_id = None


# ---------------------------------------------------------------------------
# 评测主流程（store / llm / judge 可注入假件 → test_eval_runner 确定性测试）
# ---------------------------------------------------------------------------

def _load_eval_data() -> tuple[list[dict], list[dict]]:
    data = yaml.safe_load(Path(settings.eval_cases_path).read_text(encoding="utf-8"))
    return data["documents"], data["cases"]


def _is_refusal(answer: str) -> bool:
    return any(m in answer for m in REFUSAL_MARKERS) or answer == NO_RESULT_ANSWER


def _build_result(
    outcomes: list[CaseOutcome],
    *,
    mode: str,
    top_k: int,
    tag: str | None,
    no_judge: bool,
    cost_yuan: float,
    judge: Any | None,
) -> dict[str, Any]:
    """把逐例结果聚合成结果 JSON（§5.2）。

    ★ 抽成纯函数是为 test_eval_runner 能**不连库、不跑模型**就确定性地验证聚合数学
      （hit@k 命中率、recall、MRR 均值、faithfulness 均值、拒答准确率）。
      读 settings 是可接受的：测试用 monkeypatch 固定即可复现。
    """
    def _mean(vals: list[float | None]) -> float | None:
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def _rate(key: str) -> float | None:
        vals = [o.retrieval[key] for o in outcomes if o.retrieval.get(key) is not None]
        return round(sum(1 for v in vals if v) / len(vals), 4) if vals else None

    kk = max(top_k, 5)
    gen_vals = [o.generation for o in outcomes if o.generation]
    refusal_cases = [o for o in outcomes if "refused" in o.generation]
    return {
        "config": {
            "mode": mode, "rerank_enabled": settings.rerank_enabled,
            "embedding_enabled": settings.embedding_enabled, "top_k": top_k,
            "query_rewrite_enabled": settings.query_rewrite_enabled,
            "case_count": len(outcomes), "tag": tag,
        },
        "retrieval": {
            f"hit@{kk}": _rate(f"hit@{kk}"),
            f"recall@{kk}": _rate(f"recall@{kk}"),
            "mrr": _mean([o.retrieval["rr"] for o in outcomes
                          if o.retrieval.get("rr") is not None]),
        },
        "generation": {
            "faithfulness": _mean([g.get("faithfulness") for g in gen_vals
                                   if g.get("faithfulness") is not None]),
            "completeness": _mean([g.get("completeness") for g in gen_vals
                                   if "completeness" in g and g["completeness"] is not None]),
            "refusal_accuracy": (
                round(sum(1 for o in refusal_cases if o.generation["refused"]) / len(refusal_cases), 4)
                if refusal_cases else None
            ),
        },
        "cases": [
            {"id": o.id, "passed": o.passed,
             "retrieval": o.retrieval, "generation": o.generation,
             "details": o.details}
            for o in outcomes
        ],
        "cost_yuan": round(cost_yuan + getattr(judge, "cost_yuan", 0.0), 6),
        # ★ false 时整份结果不应被采信（§5.2）；--no-judge 没有生成指标，视为不适用
        "judge_validated": None if no_judge else True,
    }


async def evaluate(
    *,
    no_judge: bool,
    mode: str,
    top_k: int,
    tag: str | None,
    store: Any | None = None,      # 可注入假件（test_eval_runner 确定性测试）
    session_factory: Any | None = None,   # 同上：测试注入内存 SQLite，不碰真实 PG
    llm: Any | None = None,
    judge: Any | None = None,
) -> tuple[dict, int]:
    """返回 (结果 JSON, 进程退出码)。退出码语义见 §5.4。"""
    documents, cases = _load_eval_data()
    if tag:
        cases = [c for c in cases if tag in (c.get("tags") or [])]
        if not cases:
            print(f"没有匹配 tag={tag!r} 的用例", file=sys.stderr)
            return {}, 2

    owned_store = store is None
    store = store or CorpusStore()
    llm = llm or llm_client
    # 真跑用全局 SessionLocal（PG）；测试注入自己的工厂，绝碰真实库
    db_factory = session_factory or SessionLocal
    outcomes: list[CaseOutcome] = []
    judge_obj = None
    exit_code = 0

    async with db_factory() as db:
        if owned_store:
            if not embedding_client.is_ready:
                embedding_client.startup()
            await store.seed(db, documents)
        if not no_judge and judge is None:
            judge_obj = Judge(llm=llm)      # 测试注入的假 LLM 同样喂给裁判
            # ★ 先证明尺子是准的（§1）：自检不过，整份结果不可信
            ratio = await validate_judge(judge_obj)
            if ratio < 1.0:
                print(f"❌ judge 自检未通过（{ratio:.0%}），评测结果不可信", file=sys.stderr)
                return {"judge_validated": False, "judge_self_check": ratio}, 1
        judge = judge or judge_obj

        reranker = rerank_client if (settings.rerank_enabled and rerank_client.is_ready) else None
        embedder = embedding_client if mode == "hybrid" else None
        cost_yuan = 0.0

        for case in cases:
            question = case["question"]
            # 多轮用例用解析好的自包含检索句，隔离"实时改写"对检索指标的干扰（见 yaml 注释）
            rq = case.get("retrieval_query") or question
            expected_docs = {
                store.title_to_doc_id[t] for t in case.get("expected_doc_titles") or []
            }
            chunks = await retrieval.search(
                db, owner_id=store.owner_id, question=rq, top_k=top_k,
                embedder=embedder, reranker=reranker,
            )
            ranked_doc_ids = [c.document_id for c in chunks]
            k = max(top_k, 5)
            outcome = CaseOutcome(
                id=case["id"], question=question, tags=case.get("tags") or [],
                difficulty=case.get("difficulty", "easy"),
                retrieval={
                    f"hit@{k}": hit_at_k(ranked_doc_ids, expected_docs, k) if expected_docs else None,
                    f"recall@{k}": recall_at_k(ranked_doc_ids, expected_docs, k) if expected_docs else None,
                    "rr": reciprocal_rank(ranked_doc_ids, expected_docs) if expected_docs else None,
                },
                details={
                    "retrieved": [
                        {"document_title": c.document_title, "chunk_id": c.chunk_id,
                         "keyword_rank": c.keyword_rank, "vector_rank": c.vector_rank}
                        for c in chunks
                    ],
                },
            )

            if not no_judge and not case.get("expect_refusal"):
                messages = build_messages(question, chunks, [])
                result = await llm.chat(messages, temperature=0.2, user_id="eval-qa")
                cost_yuan += result.cost_yuan
                contexts = [c.content for c in chunks]
                faith = await judge.faithfulness(result.content, contexts)
                comp = (await judge.completeness(result.content, case["key_points"])
                        if case.get("key_points") else None)
                outcome.generation = {
                    "faithfulness": faith["score"],
                    "completeness": comp["score"] if comp else None,
                }
                outcome.details["answer"] = result.content[:300]
                outcome.details["judge_reason"] = faith["reason"]
                if comp:
                    outcome.details["missing_points"] = comp["missing"]
                # 通过判定：期望文档必须进 Top-1（有期望时），且忠实
                outcome.passed = (
                    faith["score"] == 1.0
                    and (not expected_docs or hit_at_k(ranked_doc_ids, expected_docs, 1))
                )
            elif case.get("expect_refusal"):
                # 拒答类：不生成也先记判定所需信息；judge 开着才真正考察"模型会不会编"
                if not no_judge:
                    messages = build_messages(question, chunks, [])
                    result = await llm.chat(messages, temperature=0.2, user_id="eval-qa")
                    cost_yuan += result.cost_yuan
                    refused = _is_refusal(result.content)
                    outcome.generation = {"refused": refused}
                    outcome.details["answer"] = result.content[:300]
                    outcome.passed = refused
            else:
                outcome.passed = (
                    bool(expected_docs) and hit_at_k(ranked_doc_ids, expected_docs, k)
                ) if expected_docs else True

            outcomes.append(outcome)

    # ---------- 聚合 ----------
    result_json = _build_result(
        outcomes, mode=mode, top_k=top_k, tag=tag,
        no_judge=no_judge, cost_yuan=cost_yuan, judge=judge,
    )

    # ---------- 打印 ----------
    print(f"\n{'=' * 78}")
    print(f"评测 {len(outcomes)} 条 ｜ 模式 {mode} ｜ rerank={'开' if settings.rerank_enabled else '关'}"
          f" ｜ judge={'跳过' if no_judge else '启用'}")
    print("-" * 78)
    for o in outcomes:
        mark = "✅" if o.passed else "❌"
        rr = o.retrieval.get("rr")
        rr_s = f"rr={rr:.2f}" if rr is not None else "rr= - "
        gen = ""
        if o.generation:
            gen = ("拒答✅" if o.generation.get("refused")
                   else f"忠实={o.generation.get('faithfulness', '-')}"
                        f" 完整={o.generation.get('completeness', '-')}")
        print(f"{mark} {o.id:<26} {rr_s:<8} {gen}")
    print("-" * 78)
    r = result_json["retrieval"]
    g = result_json["generation"]
    kk = f"hit@{max(top_k, 5)}"
    print(f"检索：{kk}={r[kk]}  recall@{max(top_k, 5)}={r[f'recall@{max(top_k, 5)}']}  mrr={r['mrr']}")
    if not no_judge:
        print(f"生成：faithfulness={g['faithfulness']}  completeness={g['completeness']}"
              f"  拒答准确率={g['refusal_accuracy']}  生成成本≈{result_json['cost_yuan']}元")
    print("=" * 78)

    if owned_store:
        async with db_factory() as db:
            await store.cleanup(db)
        print("🧹 已清理评测数据（独立评测用户级联删除）")

    return result_json, exit_code


def _apply_fail_under(result: dict, args: argparse.Namespace) -> int:
    """§5.4：低于阈值 → 非零退出码，让 CI 能拦住"效果变差"（测试只保证不崩，不保证不变差）。"""
    kk = f"hit@{max(args.top_k, 5)}"
    checks = [
        (args.fail_under_hit5, result["retrieval"].get(kk), f"{kk}"),
        (args.fail_under_recall, result["retrieval"].get(f"recall@{max(args.top_k, 5)}"), "recall"),
        (args.fail_under_mrr, result["retrieval"].get("mrr"), "mrr"),
        (args.fail_under_faithfulness, result["generation"].get("faithfulness"), "faithfulness"),
    ]
    for floor, value, name in checks:
        if floor is None:
            continue
        if value is None:
            print(f"❌ 阈值 {name}≥{floor} 无法校验：本次评测没有该指标（--no-judge？）", file=sys.stderr)
            return 1
        if value < floor:
            print(f"❌ {name}={value} 低于阈值 {floor}", file=sys.stderr)
            return 1
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="全量评测（阶段 6 §5）")
    ap.add_argument("--no-judge", action="store_true", help="只跑检索指标（零 LLM 成本）")
    ap.add_argument("--mode", choices=["hybrid", "keyword"], default="hybrid",
                    help="keyword=纯关键词（阶段 3 行为基线）")
    ap.add_argument("--top-k", type=int, default=settings.final_top_k)
    ap.add_argument("--tag", default=None, help="只跑某类用例（如 同义改写）")
    ap.add_argument("--json", dest="json_out", default=None, help="结果写入 JSON 文件")
    ap.add_argument("--fail-under-hit5", type=float, default=None)
    ap.add_argument("--fail-under-recall", type=float, default=None)
    ap.add_argument("--fail-under-mrr", type=float, default=None)
    ap.add_argument("--fail-under-faithfulness", type=float, default=None)
    args = ap.parse_args()

    if args.mode == "keyword":
        # 只影响本次进程：embedder 不注入 → search 自然退回关键词（渐进降级同一通路）
        settings.embedding_enabled = False

    async def _run() -> int:
        try:
            if not args.no_judge or args.mode == "hybrid":
                # judge 需要 LLM 客户端已 startup；hybrid 需要 embedder
                if not args.no_judge:
                    await llm_client.startup()
                if args.mode == "hybrid" and not embedding_client.is_ready:
                    embedding_client.startup()
                if settings.rerank_enabled:
                    rerank_client.startup()
            result, code = await evaluate(
                no_judge=args.no_judge, mode=args.mode, top_k=args.top_k, tag=args.tag,
            )
            if not result:
                return code or 1
            if args.json_out:
                Path(args.json_out).write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"📄 结果已写入 {args.json_out}")
            if code != 0:
                return code
            return _apply_fail_under(result, args)
        finally:
            await llm_client.shutdown()
            await engine.dispose()

    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
