"""Agent 循环（阶段 5 核心）：LLM ↔ 工具，直到模型给出最终回答或触发保护条件。

★ 这不是"把 LangChain 换个写法"，而是刻意保持**完全透明**（文档 §1.5）：
  每一步的输入输出、每次 LLM 调用的 token 与成本、每次工具调用的耗时，
  都在你能读到的代码里，都能写进 chat_logs（阶段 6 的评估体系需要它）。

四条实测得出的设计（文档 §1）：
- §1.1 回传 reasoning_content 是质量优化、不是硬约束（不回传也不报 400）。
- §1.2 工具错误当结果返回、绝不抛异常冒泡；工具描述互斥。
- §1.3 连"我加了保护"都要实测——calc 的 OverflowError 就是无效保护的例子。
- §1.4 每轮都显式关闭思考模式（多轮 + 思考 = 成本成倍放大）。
"""

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.embedding import EmbeddingClient
from app.clients.llm import LLMClient
from app.clients.rerank import RerankClient
from app.clients.tracing import trace_span, update_span
from app.core.config import settings
from app.models.chunk import Chunk
from app.models.document import Document
from app.services.retrieval import RetrievedChunk
from app.tools import build_tools
from app.tools.base import Tool

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """你是知识库问答助手，可以调用工具来完成任务。

工作方式：
1. 先判断问题需要什么：涉及用户文档内容 → 调 search_kb；需要算数 → 调 calc；
   涉及「今天/最近」等相对时间 → 调 now。
2. 拿到工具结果后判断是否足够回答。不够就再调一次别的工具，但**不要用同样的参数重复调用**。
3. 资料不足时直接说明"根据现有资料无法回答"，**绝不编造**。
4. 最终回答要简洁，并在引用资料处标注 [1] [2] 这样的编号。
"""


@dataclass
class AgentStep:
    """一次工具调用记录，用于 trace 与 chat_logs。"""

    step: int
    tool: str
    arguments: dict
    result_preview: str
    is_error: bool
    latency_ms: int


@dataclass
class AgentRun:
    """一次 Agent 运行的完整结果。"""

    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    citations: list[RetrievedChunk] = field(default_factory=list)
    # 累计 LLM 使用量（多轮之和，这是 Agent 成本统计的关键）
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_yuan: float = 0.0
    llm_calls: int = 0
    latency_ms: int = 0
    # 保护条件是否触发（触发时要如实告知用户，而不是假装正常）
    stopped_reason: str | None = None
    tool_counts: dict[str, int] = field(default_factory=dict)


class AgentRunner:
    def __init__(
        self,
        db: AsyncSession,
        llm: LLMClient,
        *,
        owner_id: int,
        embedder: EmbeddingClient | None = None,
        reranker: RerankClient | None = None,
    ) -> None:
        self._db = db
        self._llm = llm
        self._owner_id = owner_id
        self._tools: dict[str, Tool] = build_tools(
            db=db, embedder=embedder, reranker=reranker, owner_id=owner_id
        )
        self._schemas = [t.to_openai_schema() for t in self._tools.values()]

    async def run(
        self,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
        trace: Any = None,          # 可选的 Langfuse trace（阶段 6，未启用时全 no-op）
    ) -> AgentRun:
        started = time.perf_counter()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            *(history or []),
            {"role": "user", "content": question},
        ]
        run = AgentRun(answer="")
        call_signatures: Counter[str] = Counter()

        for step in range(1, settings.agent_max_steps + 1):
            # ---------- 保护 3：总超时 ----------
            elapsed = time.perf_counter() - started
            if elapsed > settings.agent_timeout_seconds:
                run.stopped_reason = "timeout"
                break

            # ---------- 调 LLM（★ 显式关闭思考模式，见 §1.4）----------
            with trace_span(
                "agent_llm", as_type="generation", attach_to=trace, step=step
            ) as llm_span:
                result = await self._llm.chat(
                    messages,
                    tools=self._schemas,          # ★ LLMClient 已支持 tools 透传（§5.3）
                    temperature=0.2,
                    thinking=False,
                    user_id=f"u{self._owner_id}",
                )
                update_span(
                    llm_span,
                    output=result.content,
                    usage_details={
                        "input": result.prompt_tokens,
                        "output": result.completion_tokens,
                    },
                    metadata={"cost_yuan": result.cost_yuan},
                )
            run.llm_calls += 1
            run.prompt_tokens += result.prompt_tokens
            run.completion_tokens += result.completion_tokens
            run.cached_tokens += result.cached_tokens
            run.cost_yuan += result.cost_yuan

            # ---------- 保护 2：成本上限 ----------
            if run.cost_yuan > settings.agent_max_cost_yuan:
                run.stopped_reason = "cost_limit"
                run.answer = result.content or "（已达单次成本上限，未能给出完整回答）"
                break

            tool_calls = result.raw.get("tool_calls") or []
            if not tool_calls:
                # 没有工具调用 = 模型给出最终回答
                run.answer = result.content
                break

            # ---------- 把 assistant 的 tool_calls 追加进对话 ----------
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": result.content or "",
                "tool_calls": tool_calls,
            }
            # ★ 回传 reasoning_content：当前 API 不强制（§1.1 实测），
            #   但能让模型复用上一轮推理，属于质量优化。关闭思考模式时该字段为空。
            reasoning = result.raw.get("reasoning_content")
            if reasoning:
                assistant_msg["reasoning_content"] = reasoning
            messages.append(assistant_msg)

            # ---------- 逐个执行工具（★ 错误也当结果返回，§1.2）----------
            repeat_hit = False
            for call in tool_calls:
                cid = call.get("id", "")
                fn = call.get("function", {})
                name = fn.get("name", "")
                raw_args = fn.get("arguments") or "{}"
                t0 = time.perf_counter()
                args: dict = {}

                try:
                    args = json.loads(raw_args)
                    if not isinstance(args, dict):
                        args = {}
                except json.JSONDecodeError:
                    content = f"错误：参数不是合法 JSON（收到 {raw_args[:80]}）"
                    is_error = True
                else:
                    # ---------- 保护 4：重复调用检测 ----------
                    signature = (
                        f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
                    )
                    call_signatures[signature] += 1
                    if call_signatures[signature] > settings.agent_repeat_call_limit:
                        content = (
                            "错误：你已经用完全相同的参数调用过这个工具多次，"
                            "请换一个思路或直接基于已有信息回答。"
                        )
                        is_error = True
                        # ★ 被拒的调用**不算一次执行**：不写进 steps/agent_tools，
                        #   只把可读提示回灌给模型（否则 tool_counts 会被未执行的尝试虚高）。
                        messages.append({
                            "role": "tool", "tool_call_id": cid, "content": content,
                        })
                        run.stopped_reason = "repeat_call"
                        repeat_hit = True
                        break

                    tool = self._tools.get(name)
                    if tool is None:
                        content = f"错误：不存在的工具 {name}"
                        is_error = True
                    else:
                        with trace_span(
                            f"tool:{name}", attach_to=trace, step=step
                        ) as tool_span:
                            tr = await tool.run(**args)
                            content = tr.content
                            is_error = tr.is_error
                            update_span(
                                tool_span,
                                output=content[:200],
                                metadata={
                                    "args": args, "is_error": is_error,
                                    "meta": tr.meta,
                                },
                            )
                latency_ms = int((time.perf_counter() - t0) * 1000)

                run.steps.append(AgentStep(
                    step=step, tool=name, arguments=args,
                    result_preview=content[:200], is_error=is_error,
                    latency_ms=latency_ms,
                ))
                messages.append({
                    "role": "tool", "tool_call_id": cid, "content": content,
                })

            if repeat_hit:
                break   # 外层也停：重复调用说明已陷入死循环，继续只是烧钱
        else:
            # for 循环正常走完（用尽轮次）——不是 break 出来的
            if run.stopped_reason is None:
                run.stopped_reason = "max_steps"

        # ---------- 兜底：超时/用尽轮次且没有回答时，让模型"就现有信息作答"（§5.2）----------
        if not run.answer:
            messages.append({
                "role": "user",
                "content": "请立即基于目前已有的信息给出最终回答，不要再调用任何工具。"
                           "如果信息不足，请明确说明无法回答。",
            })
            with trace_span(
                "agent_final", as_type="generation", attach_to=trace
            ) as final_span:
                final = await self._llm.chat(
                    messages, temperature=0.2, thinking=False,
                    user_id=f"u{self._owner_id}",
                )
                update_span(final_span, output=final.content)
            run.llm_calls += 1
            run.prompt_tokens += final.prompt_tokens
            run.completion_tokens += final.completion_tokens
            run.cached_tokens += final.cached_tokens
            run.cost_yuan += final.cost_yuan
            run.answer = final.content or "抱歉，我没能完成这个任务。"

        # ---------- 汇总引用：从 search_kb 工具收集 chunk_ids 再回表取内容 ----------
        search_tool = self._tools.get("search_kb")
        if search_tool is not None and getattr(search_tool, "last_chunk_ids", None):
            run.citations = await self._load_citations(search_tool.last_chunk_ids)

        run.tool_counts = dict(Counter(s.tool for s in run.steps))
        run.latency_ms = int((time.perf_counter() - started) * 1000)

        logger.info(
            '{"event":"agent_run","user":%s,"steps":%s,"llm_calls":%s,'
            '"tools":"%s","cost_yuan":%s,"latency_ms":%s,"stopped":%s}',
            self._owner_id, len(run.steps), run.llm_calls,
            json.dumps(run.tool_counts, ensure_ascii=False),
            round(run.cost_yuan, 6), run.latency_ms, run.stopped_reason,
        )
        return run

    async def _load_citations(self, chunk_ids: list[int]) -> list[RetrievedChunk]:
        """按 chunk_id 回表取切片（复用阶段 4 的取数逻辑，带 owner 校验）。"""
        stmt = (
            select(Chunk, Document.title)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.id.in_(chunk_ids), Chunk.owner_id == self._owner_id)  # ★ 越权防线
            .where(Document.status == "ready")
        )
        rows = {c.id: (c, title) for c, title in (await self._db.execute(stmt)).all()}
        out: list[RetrievedChunk] = []
        for cid in chunk_ids:
            item = rows.get(cid)
            if item is None:
                continue
            c, title = item
            out.append(RetrievedChunk(
                chunk_id=c.id, document_id=c.document_id, document_title=title,
                seq=c.seq, content=c.content, score=0.0,
            ))
        return out
