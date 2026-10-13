"""解析层测试（阶段 3 §9.3）。重点覆盖两个"静默失败"陷阱：
扫描件返回空串、损坏/加密 PDF 抛裸异常——都必须转成受控异常。"""

import pytest

from app.services.parsing import (
    EmptyDocumentError,
    UnsupportedDocumentError,
    extract_docx_text,
    extract_pdf_text,
    extract_text,
)


def test_pdf_文字层提取成功(make_pdf):
    raw = make_pdf("Refund policy: submit a claim and it is reviewed in 3 workdays.")
    text = extract_pdf_text(raw, max_pages=500)
    assert "Refund" in text and "3 workdays" in text


def test_pdf_扫描件无文字层抛空文档错误(make_pdf):
    """★ §2.2：只有图形、无文字层的 PDF，extract 会"成功返回空串"。
    必须显式抛 EmptyDocumentError，否则会以 ready/0 切片静默入库。"""
    raw = make_pdf(None)  # 无文字层
    with pytest.raises(EmptyDocumentError) as exc:
        extract_pdf_text(raw, max_pages=500)
    assert "扫描件" in str(exc.value) or "文字层" in str(exc.value)


def test_pdf_损坏抛不支持错误而非裸异常(make_pdf):
    """§2.3：损坏内容不能抛出裸 PdfStreamError，必须转 UnsupportedDocumentError。"""
    with pytest.raises(UnsupportedDocumentError):
        extract_pdf_text(b"%PDF-1.4 this is definitely not a real pdf", max_pages=500)


def test_pdf_加密抛不支持错误(make_pdf):
    raw = make_pdf("Secret content that should not be readable.", encrypted=True)
    with pytest.raises(UnsupportedDocumentError) as exc:
        extract_pdf_text(raw, max_pages=500)
    assert "加密" in str(exc.value)


def test_pdf_超页数限制抛不支持错误(make_pdf):
    raw = make_pdf("Page filler text for limit test.", pages=8)
    with pytest.raises(UnsupportedDocumentError) as exc:
        extract_pdf_text(raw, max_pages=5)
    assert "超过上限" in str(exc.value)


def test_pdf_多页全部提取且保序(make_pdf):
    raw = make_pdf("Line A\nLine B", pages=3)
    text = extract_pdf_text(raw, max_pages=500)
    # 每页都有 Line A/Line B，且页间用空行分隔拼接
    assert text.count("Line A") == 3
    assert text.index("Line A") < text.index("Line B")


def test_docx_段落与表格都被提取(make_docx):
    raw = make_docx(
        ["售后政策说明书", "一、退款流程"],
        [["项目", "时限"], ["审核", "3 个工作日"]],
    )
    text = extract_docx_text(raw)
    assert "退款流程" in text
    assert "项目 | 时限" in text       # 表格行以竖线拼接
    assert "3 个工作日" in text


def test_docx_空内容抛空文档错误(make_docx):
    raw = make_docx([])  # 没有段落
    with pytest.raises(EmptyDocumentError):
        extract_docx_text(raw)


def test_extract_text_按类型分派_txt_和_md():
    assert extract_text("你好世界".encode(), "txt", 500) == "你好世界"
    assert "md 内容" in extract_text("# md 内容".encode(), "md", 500)


def test_extract_text_未知类型抛不支持错误():
    with pytest.raises(UnsupportedDocumentError):
        extract_text(b"whatever", "xlsx", 500)
