"""文档解析与切片。阶段 0 只支持 txt/md，按字符切分。

对应 §7.1。split_text 是纯函数，必须有单测覆盖。
"""

import re
from dataclasses import dataclass

from app.core.config import settings


@dataclass
class ChunkDraft:
    seq: int
    content: str


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(
    text: str, chunk_size: int | None = None, overlap: int | None = None
) -> list[ChunkDraft]:
    """按段落聚合切片，超长段落内部再切，相邻切片保留 overlap 重叠。

    为什么按段落而不是硬切 N 个字符：
    硬切会把一句话、一张表拦腰截断，检索到的片段语义不完整，模型容易答错。
    阶段 4 会在此基础上加"按标题层级切分"，但段落优先的原则不变。
    """
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    text = normalize(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    drafts: list[ChunkDraft] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            drafts.append(ChunkDraft(seq=len(drafts), content=buf.strip()))
        buf = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            flush()
            start = 0
            while start < len(para):
                piece = para[start:start + chunk_size]
                drafts.append(ChunkDraft(seq=len(drafts), content=piece))
                if start + chunk_size >= len(para):
                    break
                start += chunk_size - overlap
            continue

        if len(buf) + len(para) + 2 <= chunk_size:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            flush()
            buf = para

    flush()

    # 段落聚合产生的相邻切片之间补重叠，避免跨片语义断裂
    if overlap > 0 and len(drafts) > 1:
        for i in range(1, len(drafts)):
            prev_tail = drafts[i - 1].content[-overlap:]
            if prev_tail and not drafts[i].content.startswith(prev_tail):
                drafts[i] = ChunkDraft(
                    seq=drafts[i].seq, content=f"{prev_tail}\n\n{drafts[i].content}"
                )

    return drafts


# 支持的文件扩展名：阶段 0 只 txt/md，阶段 3 加 pdf/docx（异步解析）
ALLOWED_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}

# 快路径：本就是纯文本，上传时直接同步解析，不进 worker 队列（§5.3）
SYNC_SOURCE_TYPES = frozenset({"txt", "md"})
# 异步路径：需要 CPU 解析，投递给 worker
ASYNC_SOURCE_TYPES = frozenset({"pdf", "docx"})


def detect_source_type(filename: str) -> str | None:
    """返回 'txt'/'md'/'pdf'/'docx'，不支持的后缀返回 None（由 endpoints 报 415）。"""
    lowered = filename.lower()
    for suffix in ALLOWED_SUFFIXES:
        if lowered.endswith(suffix):
            return suffix.lstrip(".")
    return None


def decode_bytes(raw: bytes) -> str:
    """utf-8 优先，兼容常见编码；全部失败则按 utf-8 replace。"""
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")
