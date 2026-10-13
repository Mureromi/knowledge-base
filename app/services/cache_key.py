"""缓存键构造（阶段 1 §5.3 / 阶段 2 §4 升级）。

正确性关键：
1. 必须含 owner_id，否则 A 用户的问题会命中 B 用户的答案 → 数据泄露。
2. 必须含检索结果指纹，否则"上传新文档 / 删除文档"后仍返回旧答案。
3. 阶段 2：必须含**窗口内**历史指纹，否则多轮下"那要多久？"会命中另一个会话的答案。
   ⚠️ 只能用窗口内历史，不能用整会话：后者会让指纹每轮都变，缓存永不命中（§4.2）。
4. 归一化只做"表层"（大小写、空白、末尾标点），不做语义改写——那是阶段 4 的活。
"""

import hashlib
import re

from app.core.config import settings
from app.services.retrieval import RetrievedChunk


def normalize_question(question: str) -> str:
    """归一化：去首尾空白、压缩内部空白、转小写、去掉结尾标点。

    目的：让 "退款要多久？" 和 "退款要多久" 命中同一条缓存。
    ⚠️ 不要做更激进的归一化（如去停用词、同义词替换），那是语义缓存的活，
      在阶段 1 做会带来"看起来相同其实想问的不一样"的风险。
    """
    q = question.strip()
    q = re.sub(r"\s+", " ", q)
    q = q.lower()
    q = re.sub(r"[？?！!。.，,、；;：:]+$", "", q)
    return q


def context_fingerprint(chunks: list[RetrievedChunk]) -> str:
    """本次检索结果的指纹：切片 id + 内容哈希。

    内容哈希而非只用 id：文档被更新（同 id 内容变了）时指纹必须变。
    """
    h = hashlib.sha256()
    for c in chunks:
        h.update(f"{c.chunk_id}:".encode())
        h.update(hashlib.sha256(c.content.encode("utf-8")).hexdigest().encode())
        h.update(b"|")
    return h.hexdigest()


def answer_cache_key(
    *,
    owner_id: int,
    question: str,
    chunks: list[RetrievedChunk],
    window_messages: list[dict[str, str]] | None = None,
) -> str:
    """★ 缓存键 = owner_id + 归一化问题 + 窗口历史指纹 + 检索结果指纹。

    为什么必须含 owner_id：否则 A 用户的问题命中 B 用户的答案 = 数据泄露。
    window_messages 必须与真正发给 LLM 的历史完全一致（见阶段 2 文档 §4.2），
    且必须在"写入本轮消息之前"读窗口（§3.2），否则指纹含本轮问题，永不命中。
    """
    from app.services.session import history_fingerprint

    hist = history_fingerprint(window_messages or [])
    raw = (
        f"{owner_id}|{normalize_question(question)}|{hist}|"
        f"{context_fingerprint(chunks)}"
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    # 版本号 v1 → v2：键结构变了，旧缓存自然失效不命中，靠 TTL 自然过期，
    # 不需要写任何清理脚本——这就是阶段 1 "前缀带版本号"设计的兑现。
    return f"{settings.cache_key_prefix}:v2:answer:{digest}"
