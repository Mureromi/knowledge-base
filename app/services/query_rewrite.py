"""把多轮追问改写成自包含的检索句。

★ 为什么必须做（阶段 4 §4.1）：向量检索无法处理指代与省略。
   「那退货呢？」单看这句根本没法定向量，必须结合上文补全。

★ 为什么用 LLM 而不是拼接历史（对比见 §4.3）：拼接会稀释向量语义，
   指代也没被真正还原；LLM 改写质量最高，且结果被缓存后重复问题不再花钱。

★ 改写只影响**检索**，不影响**回答**：qa 里发给 LLM 的仍是原始问题 + 真实历史，
   改写句只用来去库里捞切片（§4.3 的坑，写成把改写句当提问发给模型会答非所问）。
"""

import hashlib
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.cache import CacheBackend
from app.clients.llm import LLMClient
from app.core.config import settings

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = """你是一个检索查询改写器。根据对话历史，把用户最新的一句话改写成一个**自包含的检索查询**。

规则：
1. 只输出改写后的查询语句本身，不要解释、不要加引号、不要加任何前缀。
2. 补全指代和省略的信息（「它」「那个」「那…呢」要还原成具体对象）。
3. 保留原句里的关键术语（型号、订单号、专有名词），不要替换成同义词。
4. 如果原句本身已经自包含，原样输出。
5. 改写后的查询不超过 60 字。

示例：
历史：用户问「退款要多久到账？」助手答「审核通过后 5-7 个工作日。」
输入：那退货呢？
输出：退货的条件和流程是什么
"""


async def rewrite_query(
    db: AsyncSession,
    llm: LLMClient,
    cache: CacheBackend,
    *,
    owner_id: int,
    session_id: int | None,
    question: str,
    window: list[dict[str, str]],
) -> str:
    """返回用于检索的查询句。无会话或无需改写时原样返回 question。"""
    if not settings.query_rewrite_enabled or not window:
        # 没有历史 = 首轮 = 问题本身自包含，直接返回（省一次 LLM 调用）
        return question

    history = window[-(settings.query_rewrite_max_history_turns * 2):]
    fingerprint = hashlib.sha256(
        (
            f"{owner_id}|{session_id}|{question}|"
            + "|".join(f"{m['role']}:{m['content']}" for m in history)
        ).encode()
    ).hexdigest()[:32]
    key = f"{settings.cache_key_prefix}:v2:rewrite:{fingerprint}"

    cached = await cache.get_json(key)
    if cached and cached.get("query"):
        return str(cached["query"])

    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": question},
    ]
    try:
        result = await llm.chat(messages, temperature=0.0, max_tokens=100)
        content = (result.content or "").strip()
        if not content:
            return question
        rewritten = content.strip("「」\"'").splitlines()[0][:200]
    except Exception as exc:                       # noqa: BLE001
        # ★ 改写失败不能让问答失败：退化为原问题（检索质量略降，但服务可用）
        logger.warning("查询改写失败，退化为原问题: %s", exc)
        return question

    if not rewritten:
        return question

    await cache.set_json(
        key, {"query": rewritten}, ttl=settings.query_rewrite_cache_ttl
    )
    logger.info(
        '{"event":"query_rewrite","user":%s,"raw_len":%s,"rewritten":"%s"}',
        owner_id, len(question), rewritten[:60],
    )
    return rewritten
