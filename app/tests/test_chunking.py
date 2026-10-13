"""切片纯函数的单测：最容易测、也最该测（§9）。"""

from app.services.ingest import normalize, split_text


def test_empty_returns_nothing():
    assert split_text("") == []
    assert split_text("   \n\n  ") == []


def test_normalize_handles_crlf_and_blank_lines():
    out = normalize("a\r\n\r\n\r\n\r\nb")
    assert out == "a\n\nb"


def test_short_paragraphs_get_merged():
    # 两个短段落，chunk_size 够大 → 应合成 1 片
    text = "段落一。\n\n段落二。"
    drafts = split_text(text, chunk_size=100, overlap=0)
    assert len(drafts) == 1
    assert "段落一" in drafts[0].content
    assert "段落二" in drafts[0].content


def test_seq_is_continuous():
    text = "para-" + "\n\npara-".join(str(i) for i in range(10))
    drafts = split_text(text, chunk_size=15, overlap=0)
    assert [d.seq for d in drafts] == list(range(len(drafts)))


def test_long_paragraph_internally_split():
    long = "字" * 1200
    drafts = split_text(long, chunk_size=500, overlap=80)
    assert len(drafts) >= 3
    # 每片长度 <= chunk_size（内部硬切，不再补 overlap）
    assert all(len(d.content) <= 500 for d in drafts)


def test_overlap_prepended_between_paragraph_drafts():
    text = "A" * 60 + "\n\n" + "B" * 60
    drafts = split_text(text, chunk_size=60, overlap=10)
    assert len(drafts) == 2
    # 第二片应以第一片末尾 10 个 A 开头
    assert drafts[1].content.startswith("A" * 10)


def test_chunk_size_boundary():
    """恰好等于 chunk_size 的段落，应只产生一片。"""
    text = "x" * 100
    drafts = split_text(text, chunk_size=100, overlap=0)
    assert len(drafts) == 1
    assert drafts[0].content == "x" * 100
