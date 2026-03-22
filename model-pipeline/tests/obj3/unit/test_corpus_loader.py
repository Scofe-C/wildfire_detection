"""Unit tests for corpus_loader.py — §5.1 test_corpus_loader."""

from __future__ import annotations

import pytest

from src.models.obj3_gemini.corpus_loader import (
    CorpusDocument,
    CorpusLoadError,
    estimate_corpus_tokens,
    get_corpus_as_text,
    load_corpus_texts,
)


class TestLoadCorpusTexts:
    def test_load_corpus_valid_dir(self, tmp_path):
        """A directory with 2 PDFs and 1 TXT returns 3 CorpusDocuments."""
        v1 = tmp_path / "v1"
        v1.mkdir()
        (v1 / "doc_a.pdf").write_bytes(b"%PDF-1.4 fake")
        (v1 / "doc_b.pdf").write_bytes(b"%PDF-1.4 also fake")
        (v1 / "notes.txt").write_text("Some reference notes", encoding="utf-8")

        docs = load_corpus_texts(tmp_path, "v1")
        assert len(docs) == 3
        assert all(isinstance(d, CorpusDocument) for d in docs)

        mimes = {d.mime_type for d in docs}
        assert "application/pdf" in mimes
        assert "text/plain" in mimes

    def test_load_corpus_empty_dir(self, tmp_path):
        v1 = tmp_path / "v1"
        v1.mkdir()
        with pytest.raises(CorpusLoadError, match="empty"):
            load_corpus_texts(tmp_path, "v1")

    def test_load_corpus_missing_dir(self, tmp_path):
        with pytest.raises(CorpusLoadError, match="does not exist"):
            load_corpus_texts(tmp_path, "nonexistent")


class TestEstimateCorpusTokens:
    def test_token_estimate_reasonable(self):
        # 100 KB of text → should be 20,000–30,000 tokens
        text = b"a" * 100_000
        docs = [CorpusDocument("test.txt", text, "text/plain")]
        tokens = estimate_corpus_tokens(docs)
        assert 20_000 <= tokens <= 30_000


class TestGetCorpusAsText:
    def test_corpus_as_text_basic(self):
        docs = [
            CorpusDocument("doc.txt", b"Hello world", "text/plain"),
        ]
        text = get_corpus_as_text(docs)
        assert "Hello world" in text
        assert "doc.txt" in text

    def test_corpus_as_text_truncation(self):
        docs = [
            CorpusDocument("big.txt", b"x" * 10_000, "text/plain"),
        ]
        text = get_corpus_as_text(docs, max_corpus_chars=100)
        assert len(text) <= 120  # 100 + "[TRUNCATED]" suffix
        assert "[TRUNCATED]" in text

    def test_corpus_pdf_placeholder(self):
        docs = [
            CorpusDocument("report.pdf", b"%PDF-fake", "application/pdf"),
        ]
        text = get_corpus_as_text(docs)
        assert "report.pdf" in text
        assert "binary" in text
