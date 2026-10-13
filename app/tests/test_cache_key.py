"""缓存键构造测试（阶段 1 §10.2 + 阶段 2 §4）。

防回归的关键断言：
- **不同用户同问题 → 键不同**（数据泄露的防线）
- **检索内容变了 → 键不同**（"改了文档答案不变" bug 的防线）
- **历史窗口变了 → 键不同**（阶段 2：多轮下"那要多久？"不能串会话）
"""

from app.services.cache_key import (
    answer_cache_key,
    context_fingerprint,
    normalize_question,
)
from app.services.retrieval import RetrievedChunk


def _chunk(cid: int, content: str, doc_id: int = 1, seq: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id=doc_id,
        document_title="doc",
        seq=seq,
        content=content,
        score=1.0,
    )


# ---- normalize_question ----

def test_normalize_trims_and_lowercases():
    assert normalize_question("  Hello World  ") == "hello world"


def test_normalize_collapses_internal_whitespace():
    assert normalize_question("退款   要\t多久") == "退款 要 多久"


def test_normalize_strips_trailing_punctuation():
    # "退款要多久？" 与 "退款要多久" 必须同键
    assert normalize_question("退款要多久？") == normalize_question("退款要多久")
    assert normalize_question("How long?") == normalize_question("how long")


# ---- 键的三条关键回归 ----

def test_same_owner_same_question_same_chunks_yields_same_key():
    chunks = [_chunk(1, "内容 A"), _chunk(2, "内容 B")]
    k1 = answer_cache_key(owner_id=1, question="退款要多久？", chunks=chunks)
    k2 = answer_cache_key(owner_id=1, question="退款要多久", chunks=chunks)
    assert k1 == k2, "表层归一化后同问题应命中同一条缓存"


def test_different_users_same_question_yields_different_keys():
    """★ 防跨用户数据泄露（§10.2 最重要断言）。"""
    chunks = [_chunk(1, "内容 A")]
    k_alice = answer_cache_key(owner_id=1, question="退款要多久", chunks=chunks)
    k_bob = answer_cache_key(owner_id=2, question="退款要多久", chunks=chunks)
    assert k_alice != k_bob


def test_different_chunks_yield_different_keys():
    """检索内容变了（比如用户上传了新文档）→ 键必须变。"""
    k1 = answer_cache_key(owner_id=1, question="退款", chunks=[_chunk(1, "旧")])
    k2 = answer_cache_key(owner_id=1, question="退款", chunks=[_chunk(1, "旧"), _chunk(2, "新")])
    assert k1 != k2


def test_same_chunk_id_but_different_content_yields_different_keys():
    """同 id 内容变了（文档被更新）→ 指纹必须变。"""
    k1 = answer_cache_key(owner_id=1, question="q", chunks=[_chunk(1, "v1")])
    k2 = answer_cache_key(owner_id=1, question="q", chunks=[_chunk(1, "v2")])
    assert k1 != k2


def test_empty_chunks_fingerprint_is_stable():
    assert context_fingerprint([]) == context_fingerprint([])


def test_key_has_version_prefix():
    """键名规范 §4.3：阶段 2 结构升级 v1 → v2，旧缓存自然失效不命中，
    靠 TTL 自然过期，不需要写任何清理脚本。"""
    k = answer_cache_key(owner_id=1, question="q", chunks=[_chunk(1, "c")])
    assert k.startswith("kb:v2:answer:")


# ---- 阶段 2：窗口历史指纹 ----

def test_different_history_yields_different_keys():
    """同问题同检索，但历史不同 → 键不同（防串会话）。"""
    chunks = [_chunk(1, "内容")]
    k1 = answer_cache_key(
        owner_id=1, question="那要多久", chunks=chunks, window_messages=[]
    )
    k2 = answer_cache_key(
        owner_id=1,
        question="那要多久",
        chunks=chunks,
        window_messages=[{"role": "user", "content": "退款流程"}],
    )
    assert k1 != k2


def test_same_history_yields_same_key():
    """窗口内容完全一致 → 同键（§4.2：窗口稳定后重复问仍能命中）。"""
    chunks = [_chunk(1, "内容")]
    win = [
        {"role": "user", "content": "退款"},
        {"role": "assistant", "content": "审核 3 天"},
    ]
    k1 = answer_cache_key(
        owner_id=1, question="那退货呢", chunks=chunks, window_messages=win
    )
    k2 = answer_cache_key(
        owner_id=1, question="那退货呢", chunks=chunks, window_messages=list(win)
    )
    assert k1 == k2


def test_history_order_matters():
    """历史顺序不同（影响 LLM 输出）→ 键必须不同。"""
    chunks = [_chunk(1, "内容")]
    a = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "y"},
    ]
    b = [
        {"role": "assistant", "content": "y"},
        {"role": "user", "content": "x"},
    ]
    k1 = answer_cache_key(owner_id=1, question="q", chunks=chunks, window_messages=a)
    k2 = answer_cache_key(owner_id=1, question="q", chunks=chunks, window_messages=b)
    assert k1 != k2


def test_none_window_equals_empty_window():
    """首问（无会话）传 None 与传 [] 必须同键，否则单轮缓存全部失配。"""
    chunks = [_chunk(1, "c")]
    k1 = answer_cache_key(owner_id=1, question="q", chunks=chunks)
    k2 = answer_cache_key(
        owner_id=1, question="q", chunks=chunks, window_messages=[]
    )
    assert k1 == k2
