"""Agent 增量评测（阶段 5 §9）：用阶段 6 的评测体系量 Agent 这条新路径。

用法：
    .venv/bin/python scripts/eval_agent.py                       # 真实 LLM + PG 语料，建基线
    .venv/bin/python scripts/eval_agent.py --json out.json        # 机器可读输出
    .venv/bin/python scripts/eval_agent.py --fail-under-tool 0.8  # 工具选择准确率低于阈值非零退出
    .venv/bin/python scripts/eval_agent.py --fast-path-cost 0.0005 # 给定快速路径均价，算成本倍率

★ 语料复用阶段 6 的 qa_cases.yaml documents（独立评测用户，跑完级联删除）；
  用例来自 app/evals/agent_cases.yaml（计算 / 相对时间 / 多跳——快速路径量不到的类型）。

★ 绕开缓存、会话、限流：直接驱动 AgentRunner，评的是"工具选择 + 多步推理"本身。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from app.clients.embedding import embedding_client
from app.clients.llm import llm_client
from app.clients.rerank import rerank_client
from app.core.config import settings
from app.db.session import SessionLocal, engine
from app.evals.metrics_agent import build_agent_result
from app.services.agent import AgentRunner
from scripts.eval_qa import CorpusStore

logger = logging.getLogger("eval_agent")

DEFAULT_AGENT_CASES = "app/evals/agent_cases.yaml"


async def evaluate(
    *,
    llm: Any | None = None,
    embedder: Any | None = None,
    reranker: Any | None = None,
    store: CorpusStore | None = None,
    session_factory: Any | None = None,
    cases: list[dict] | None = None,
    documents: list[dict] | None = None,
    fast_path_cost: float | None = None,
) -> tuple[dict, int]:
    """返回 (结果 JSON, 进程退出码)。

    可注入假件（store / session_factory / llm）→ test_eval_agent 不连库不跑模型即可测聚合。
    """
    llm = llm or llm_client
    embedder = embedder if embedder is not None else embedding_client
    reranker = reranker if reranker is not None else rerank_client
    session_factory = session_factory or SessionLocal

    if cases is None or documents is None:
        cases, documents = _load_data()

    # 建语料（未注入 store 时自己灌库 + 建真实向量）
    owned_store = store is None
    if owned_store:
        store = CorpusStore()
        if not embedding_client.is_ready and settings.embedding_enabled:
            embedding_client.startup()
        async with session_factory() as db:
            await store.seed(db, documents)

    outcomes: list[dict] = []
    try:
        async with session_factory() as db:
            for case in cases:
                runner = AgentRunner(
                    db, llm, owner_id=store.owner_id,
                    embedder=embedder, reranker=reranker,
                )
                run = await runner.run(case["question"])
                outcomes.append({
                    "id": case["id"],
                    "question": case["question"],
                    "expect_tool": case.get("expect_tool"),
                    "tools": list(run.tool_counts.keys()),
                    "steps": len(run.steps),
                    "stopped_reason": run.stopped_reason,
                    "cost_yuan": run.cost_yuan,
                })
                logger.info(
                    "%s: expect=%s called=%s steps=%s stopped=%s",
                    case["id"], case.get("expect_tool"),
                    list(run.tool_counts.keys()), len(run.steps), run.stopped_reason,
                )
    finally:
        if owned_store:
            async with session_factory() as db:
                await store.cleanup(db)

    result = build_agent_result(outcomes, fast_path_cost_yuan=fast_path_cost)
    return result, 0


def _load_data() -> tuple[list[dict], list[dict]]:
    """语料来自 qa_cases.yaml documents（单一事实源），用例来自 agent_cases.yaml。"""
    qa = yaml.safe_load(Path(settings.eval_cases_path).read_text(encoding="utf-8"))
    documents = qa["documents"]
    agent = yaml.safe_load(Path(DEFAULT_AGENT_CASES).read_text(encoding="utf-8"))
    return agent["cases"], documents


def _apply_fail_under(result: dict, args: argparse.Namespace) -> int:
    acc = result["metrics"]["tool_selection_accuracy"]
    if args.fail_under_tool is not None and (acc is None or acc < args.fail_under_tool):
        return 1
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description="Agent 增量评测（阶段 5 §9）")
    parser.add_argument("--json", dest="json_out", default=None, help="结果写入 JSON 文件")
    parser.add_argument("--fail-under-tool", type=float, default=None,
                        help="工具选择准确率低于此值则非零退出")
    parser.add_argument("--fast-path-cost", type=float, default=None,
                        help="快速路径平均成本（元），用于计算成本倍率")
    parser.add_argument("--hybrid", action="store_true",
                        help="启用向量检索（默认按 settings.embedding_enabled）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.hybrid:
        settings.embedding_enabled = True

    if settings.embedding_enabled:
        # search_kb 走 retrieval.search，需要 embedder；未 ready 则 startup
        if not embedding_client.is_ready:
            embedding_client.startup()
        if settings.rerank_enabled and not rerank_client.is_ready:
            rerank_client.startup()

    try:
        await llm_client.startup()           # 脚本独立运行，不走 lifespan → 必须手动启动 LLM 客户端
        result, code = await evaluate(fast_path_cost=args.fast_path_cost)
    finally:
        await llm_client.shutdown()
        await engine.dispose()

    m = result["metrics"]
    print("=== Agent 评测（阶段 5 §9.3）===")
    print(f"用例数：{result['config']['case_count']}")
    print(f"工具选择准确率：{m['tool_selection_accuracy']}")
    print(f"平均步数：{m['avg_steps']}")
    print(f"保护触发率：{m['protection_trigger_rate']}")
    print(f"Agent 平均成本：{m['avg_cost_yuan']} 元")
    print(f"成本倍率（vs 快速路径）：{m['cost_multiple_vs_fast_path']}")
    for c in result["cases"]:
        flag = "✅" if c["tool_match"] else "❌"
        print(f"  {flag} {c['id']}: expect={c['expect_tool']} called={c['tools']}"
              f" steps={c['steps']} stopped={c['stopped_reason']}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"结果已写入 {args.json_out}")

    code = _apply_fail_under(result, args) if code == 0 else code
    return code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
