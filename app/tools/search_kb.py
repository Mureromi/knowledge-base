"""知识库检索工具（阶段 5 §4.2）。

包装阶段 4 的 retrieval.search，签名与行为不变——owner_id 隔离、只查 ready 文档、
混合检索 + rerank 全都复用，工具只是把 RetrievedChunk 列表转成给模型看的编号文本。

★ 关键：检索内部抛任何异常都转成可读 ToolResult(is_error=True)，绝不冒泡（§1.2）。
"""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.retrieval import search
from app.tools.base import ToolResult, openai_schema


class SearchKBTool:
    name = "search_kb"
    description = (
        "在用户的知识库中检索资料。当问题涉及用户上传的文档内容（政策、说明、条款、流程等）时使用。"
        "不要用它回答常识性问题或做数学计算——计算请用 calc。"
        "如果一次没检索到，可以换一个检索词再试一次，但不要反复用同样的词。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索词。用完整的问句或关键短语，不要只用一个字。",
            }
        },
        "required": ["query"],
    }

    def __init__(
        self,
        db: AsyncSession,
        embedder: Any | None,
        reranker: Any | None,
        owner_id: int,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._reranker = reranker
        self._owner_id = owner_id
        # ★ 记录本次工具调用召回了哪些切片，供引用回填（tool 结果里没有 chunk_id，
        #   因为模型不需要看到 id，但接口的 citations 需要）
        self.last_chunk_ids: list[int] = []

    def to_openai_schema(self) -> dict:
        return openai_schema(self)

    async def run(self, **kwargs: Any) -> ToolResult:
        query = (kwargs.get("query") or "").strip()
        if not query:
            return ToolResult(content="错误：缺少必填参数 query（应为非空字符串）", is_error=True)

        try:
            # ★ 复用阶段 4 的检索（owner_id 隔离、只查 ready 文档、混合检索 + rerank）
            chunks = await search(
                self._db,
                owner_id=self._owner_id,
                question=query,
                embedder=self._embedder,
                reranker=self._reranker,
            )
        except Exception as exc:                     # noqa: BLE001
            # ★ 不抛异常：转成可读错误，模型会自己换词重试（§1.2 实测）
            return ToolResult(content=f"检索失败：{type(exc).__name__}", is_error=True)

        # ★ 累加而非覆盖：多跳问答（先检索 A 再检索 B）时 citations 应含两次召回。
        #   去重且保留先后顺序（citations 的 [1][2] 编号与此顺序对齐）。
        for cid in (c.chunk_id for c in chunks):
            if cid not in self.last_chunk_ids:
                self.last_chunk_ids.append(cid)

        if not chunks:
            return ToolResult(
                content="知识库中没有找到与该检索词相关的内容。可以换一种说法再试。",
                meta={"hits": 0},
            )

        # ★ 给模型的格式：带编号 + 出处，便于它在回答里引用
        lines = [
            f"[{i}] 来源《{c.document_title}》第 {c.seq + 1} 段：{c.content}"
            for i, c in enumerate(chunks, start=1)
        ]
        text = "\n\n".join(lines)
        if len(text) > settings.agent_tool_result_max_chars:
            text = text[: settings.agent_tool_result_max_chars] + "\n…（已截断）"
        return ToolResult(content=text, meta={"hits": len(chunks)})
