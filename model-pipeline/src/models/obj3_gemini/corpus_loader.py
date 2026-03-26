"""Corpus loader — load RAG reference documents for context injection.

Phase 1/2: loads corpus files and concatenates as plain text.
Phase 3 (Vertex AI): creates/reuses a context cache.

Supports three file types:
  - ``.json`` — extracted chunks (prefers ``distilled_content`` if available,
    falls back to ``content``)
  - ``.txt`` — plain text files
  - ``.pdf`` — binary PDF files (for Vertex AI cache or placeholder reference)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CorpusDocument:
    """A single loaded corpus file."""

    filename: str
    content_bytes: bytes
    mime_type: str


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CorpusLoadError(Exception):
    """Raised when corpus directory is missing or empty."""


class CacheCreationError(Exception):
    """Raised when Vertex AI cache creation fails (Phase 3)."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_corpus_texts(corpus_dir: Path, version: str) -> list[CorpusDocument]:
    """Load corpus files from ``corpus/{version}/`` (recursive).

    Supports ``.json`` (extracted chunks with ``content`` / ``distilled_content``
    field), ``.txt``, and ``.pdf`` files.  Subdirectories are walked recursively.

    Parameters
    ----------
    corpus_dir:
        Root corpus directory (e.g. ``corpus/``).
    version:
        Corpus version folder name (e.g. ``"processed"``).

    Returns
    -------
    list[CorpusDocument]

    Raises
    ------
    CorpusLoadError
        If the versioned directory does not exist or contains no documents.
    """
    import json as _json

    target = Path(corpus_dir) / version
    if not target.is_dir():
        raise CorpusLoadError(f"Corpus directory does not exist: {target}")

    docs: list[CorpusDocument] = []
    distilled_count = 0

    for path in sorted(target.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()

        if suffix == ".json":
            # Extracted chunk — prefer distilled_content, fall back to content
            try:
                raw = _json.loads(path.read_bytes().decode("utf-8", errors="replace"))
                content_text = raw.get("distilled_content") or raw.get("content", "")
                used_distilled = "distilled_content" in raw and raw["distilled_content"]
                if not content_text:
                    logger.warning("Corpus JSON has no content: %s", path.name)
                    continue
                # Prefix with source metadata for LLM context
                source_name = raw.get("source_name", path.stem)
                header = f"[{source_name}]"
                full_text = f"{header}\n{content_text}"
                docs.append(CorpusDocument(
                    filename=path.name,
                    content_bytes=full_text.encode("utf-8"),
                    mime_type="text/plain",
                ))
                if used_distilled:
                    distilled_count += 1
            except Exception as exc:
                logger.warning("Failed to parse corpus JSON %s: %s", path.name, exc)
        elif suffix == ".txt":
            docs.append(CorpusDocument(
                filename=path.name,
                content_bytes=path.read_bytes(),
                mime_type="text/plain",
            ))
        elif suffix == ".pdf":
            docs.append(CorpusDocument(
                filename=path.name,
                content_bytes=path.read_bytes(),
                mime_type="application/pdf",
            ))

    if not docs:
        raise CorpusLoadError(f"Corpus directory is empty (no .json/.pdf/.txt files): {target}")

    total_size = sum(len(d.content_bytes) for d in docs)
    logger.info(
        "Loaded %d corpus documents (%.1f KB, %d distilled) from %s",
        len(docs), total_size / 1024, distilled_count, target,
    )
    return docs


def estimate_corpus_tokens(corpus_docs: list[CorpusDocument]) -> int:
    """Rough token estimate: total bytes ÷ 4 (≈1 token per 4 bytes of English)."""
    total_bytes = sum(len(d.content_bytes) for d in corpus_docs)
    return total_bytes // 4


def get_corpus_as_text(
    corpus_docs: list[CorpusDocument],
    max_corpus_chars: int = 500_000,
) -> str:
    """Concatenate corpus contents as a single text string (Phase 1/2 fallback).

    PDF files are included by filename reference only — full PDF parsing
    requires additional libraries (e.g. ``pymupdf``) and is deferred.
    Text files are decoded as UTF-8.

    If total length exceeds *max_corpus_chars*, the output is truncated.
    """
    parts: list[str] = []
    for doc in corpus_docs:
        if doc.mime_type == "text/plain":
            try:
                text = doc.content_bytes.decode("utf-8", errors="replace")
            except Exception:
                text = f"[Could not decode {doc.filename}]"
            parts.append(f"--- {doc.filename} ---\n{text}")
        else:
            # PDF: include as a reference placeholder
            parts.append(f"--- {doc.filename} (binary, {len(doc.content_bytes)} bytes) ---")

    combined = "\n\n".join(parts)
    if len(combined) > max_corpus_chars:
        combined = combined[:max_corpus_chars] + "\n[TRUNCATED]"
    return combined


# ---------------------------------------------------------------------------
# Phase 3 stubs (Vertex AI context caching)
# ---------------------------------------------------------------------------

def get_or_create_cache(
    client: object,
    model: str,
    corpus_docs: list[CorpusDocument],
    system_prompt: str,
    ttl_seconds: int,
) -> str:
    """Create or reuse a Vertex AI context cache.

    Checks if a named cache ``wildfire-rag-corpus-v{version}`` already exists
    and is not expired.  If valid, returns cache name ("Cache hit").
    Otherwise creates a new cache ("Cache created").

    Parameters
    ----------
    client:
        A ``google.genai.Client`` instance (Vertex AI mode).
    model:
        Vertex AI model name (e.g. ``"gemini-2.5-flash"``).
    corpus_docs:
        Loaded corpus documents to cache.
    system_prompt:
        System prompt to include in cache.
    ttl_seconds:
        Cache time-to-live in seconds.

    Returns
    -------
    str
        The cache resource name.

    Raises
    ------
    CacheCreationError
        On API failure or if corpus is too small (< 2 048 tokens).
    """
    try:
        from google.genai import types
    except ImportError as exc:
        raise CacheCreationError(
            "The 'google-genai' package is required for context caching. "
            "Install with: pip install google-genai"
        ) from exc

    # Minimum token check
    token_estimate = estimate_corpus_tokens(corpus_docs)
    min_tokens = 2_048
    if token_estimate < min_tokens:
        raise CacheCreationError(
            f"Corpus too small for caching ({token_estimate} tokens < {min_tokens} minimum). "
            "Use get_corpus_as_text() instead."
        )

    # Build a deterministic display name
    total_size = sum(len(d.content_bytes) for d in corpus_docs)
    cache_display_name = f"wildfire-rag-corpus-v{len(corpus_docs)}-{total_size}"

    # Check for existing cache — list all and filter client-side by display_name.
    # ListCachedContentsConfig has no server-side `filter` parameter in this SDK version.
    try:
        existing = client.caches.list(  # type: ignore[attr-defined]
            config=types.ListCachedContentsConfig(),
        )
        for cache in existing:
            if (
                hasattr(cache, "name") and cache.name
                and hasattr(cache, "display_name")
                and cache.display_name == cache_display_name
            ):
                logger.info("Cache hit: %s", cache.name)
                return cache.name
    except Exception as exc:
        logger.warning("Failed to query existing caches: %s", exc)

    # Build corpus parts
    corpus_parts: list = []
    for doc in corpus_docs:
        if doc.mime_type == "text/plain":
            text = doc.content_bytes.decode("utf-8", errors="replace")
            corpus_parts.append(types.Part(text=f"--- {doc.filename} ---\n{text}"))
        else:
            corpus_parts.append(types.Part.from_bytes(
                data=doc.content_bytes, mime_type=doc.mime_type,
            ))

    # Create new cache — `model` is a top-level arg to create(), not part of the config.
    try:
        cache = client.caches.create(  # type: ignore[attr-defined]
            model=model,
            config=types.CreateCachedContentConfig(
                display_name=cache_display_name,
                system_instruction=system_prompt,
                contents=types.Content(parts=corpus_parts, role="user"),
                ttl=f"{ttl_seconds}s",
            ),
        )
        logger.info("Cache created: %s (TTL=%ds)", cache.name, ttl_seconds)
        return cache.name
    except Exception as exc:
        raise CacheCreationError(f"Failed to create Vertex AI cache: {exc}") from exc
