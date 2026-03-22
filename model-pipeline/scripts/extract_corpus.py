"""
extract_corpus.py — Corpus Extraction Script for OBJ-3 RAG Corpus

Extracts targeted text and table content from 6 reference PDFs into structured
JSON chunks for use by corpus_loader.py and Vertex AI context caching.

Strategy (hybrid):
  - If chunk has page_override → extract from that exact page range
  - If chunk has page_override: null → scan all pages for keyword matches

Usage:
    # Full extraction
    python scripts/extract_corpus.py --config configs/corpus_extraction.yaml

    # Single document only
    python scripts/extract_corpus.py --config configs/corpus_extraction.yaml --doc ics209

    # Dry run — prints what would be extracted, writes nothing
    python scripts/extract_corpus.py --config configs/corpus_extraction.yaml --dry-run

Dependencies:
    pdfplumber>=0.11
    pyyaml>=6.0

Output:
    corpus/processed/{geography}/{doc_id}_{chunk_id}.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pdfplumber
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("extract_corpus")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConfigValidationError(Exception):
    """Raised when corpus_extraction.yaml is missing required keys."""


class PDFExtractionError(Exception):
    """Raised when pdfplumber fails to open or read a PDF."""


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ChunkResult:
    """Result of extracting a single chunk from a PDF."""
    doc_id: str
    chunk_id: str
    geography: str
    source_name: str
    source_org: str
    source_pdf: str
    pages_extracted: list[int]
    extraction_method: str          # "page_override" or "keyword"
    extracted_at: str
    char_count: int
    tables: list[list[str]]
    content: str
    was_truncated: bool = False
    was_skipped: bool = False
    skip_reason: str = ""


@dataclass
class ExtractionSummary:
    """Summary reported after run_extraction() completes."""
    documents_processed: int = 0
    chunks_written: int = 0
    chunks_skipped: int = 0
    chunks_truncated: int = 0
    output_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def print_report(self) -> None:
        """Print human-readable summary to stdout."""
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"  Documents processed  : {self.documents_processed}")
        print(f"  Chunks written       : {self.chunks_written}")
        print(f"  Skipped (too short)  : {self.chunks_skipped}")
        print(f"  Truncated (too long) : {self.chunks_truncated}")
        print("\n  Output locations:")
        for geo, count in sorted(self.output_counts.items()):
            print(f"    corpus/processed/{geo:20s} → {count:3d} files")
        if self.warnings:
            print("\n  ⚠  Warnings (manual page_override recommended):")
            for w in self.warnings:
                print(f"    - {w}")
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Config Loading & Validation
# ---------------------------------------------------------------------------

REQUIRED_TOP_KEYS = {"extraction", "documents"}
REQUIRED_EXTRACTION_KEYS = {"input_dir", "output_dir", "min_chunk_chars",
                             "max_chunk_chars", "overlap_lines"}
REQUIRED_DOC_KEYS = {"path", "source_name", "source_org", "geography", "chunks"}
REQUIRED_CHUNK_KEYS = {"id", "keywords", "page_override", "extract_tables"}


def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load and validate corpus_extraction.yaml.

    Raises:
        ConfigValidationError: if required keys are missing at any level.
        FileNotFoundError: if config_path does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Top-level keys
    missing_top = REQUIRED_TOP_KEYS - set(config.keys())
    if missing_top:
        raise ConfigValidationError(f"Config missing top-level keys: {missing_top}")

    # extraction sub-keys
    missing_extraction = REQUIRED_EXTRACTION_KEYS - set(config["extraction"].keys())
    if missing_extraction:
        raise ConfigValidationError(f"Config missing extraction keys: {missing_extraction}")

    # per-document keys
    for doc_id, doc_cfg in config["documents"].items():
        missing_doc = REQUIRED_DOC_KEYS - set(doc_cfg.keys())
        if missing_doc:
            raise ConfigValidationError(
                f"Document '{doc_id}' missing keys: {missing_doc}"
            )
        for chunk in doc_cfg["chunks"]:
            missing_chunk = REQUIRED_CHUNK_KEYS - set(chunk.keys())
            if missing_chunk:
                raise ConfigValidationError(
                    f"Chunk '{chunk.get('id', '?')}' in '{doc_id}' missing keys: {missing_chunk}"
                )

    logger.info("Config loaded and validated: %s", config_path)
    return config


# ---------------------------------------------------------------------------
# PDF Extraction Helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """
    Normalize extracted PDF text.
    Collapses excessive whitespace, strips trailing spaces per line,
    removes null bytes that pdfplumber occasionally emits.
    """
    if not text:
        return ""
    lines = text.replace("\x00", "").splitlines()
    cleaned = [line.rstrip() for line in lines]
    # Collapse runs of more than 2 blank lines into a single blank line
    result: list[str] = []
    blank_run = 0
    for line in cleaned:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                result.append(line)
        else:
            blank_run = 0
            result.append(line)
    return "\n".join(result).strip()


def _extract_tables_from_page(page: Any) -> list[list[str]]:
    """
    Extract all tables from a single pdfplumber page object.
    Returns list of tables; each table is a list of rows (list of strings).
    Handles None cells by converting to empty string.
    """
    tables = []
    raw_tables = page.extract_tables()
    if not raw_tables:
        return tables
    for table in raw_tables:
        clean_table = [
            [str(cell) if cell is not None else "" for cell in row]
            for row in table
            if any(cell for cell in row)   # skip fully empty rows
        ]
        if clean_table:
            tables.append(clean_table)
    return tables


def extract_by_page_range(
    pdf_path: Path,
    pages: list[int],
    extract_tables: bool,
) -> tuple[str, list[list[str]]]:
    """
    Extract text and optionally tables from a specific page range.

    Args:
        pdf_path:      Absolute path to the PDF file.
        pages:         [start, end] inclusive, 1-indexed.
        extract_tables: Whether to extract tables in addition to text.

    Returns:
        (text_content, tables) — tables is [] if extract_tables=False.

    Raises:
        PDFExtractionError: if the PDF cannot be opened or pages are out of range.
    """
    if len(pages) != 2 or pages[0] > pages[1]:
        raise PDFExtractionError(
            f"Invalid page_override {pages} — must be [start, end] with start <= end"
        )

    start_page, end_page = pages[0] - 1, pages[1]  # pdfplumber is 0-indexed

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            if start_page < 0 or end_page > total_pages:
                raise PDFExtractionError(
                    f"Page range {pages} out of bounds for {pdf_path.name} "
                    f"({total_pages} pages)"
                )

            text_parts: list[str] = []
            all_tables: list[list[str]] = []

            for page in pdf.pages[start_page:end_page]:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
                if extract_tables:
                    all_tables.extend(_extract_tables_from_page(page))

    except pdfplumber.exceptions.PDFSyntaxError as e:
        raise PDFExtractionError(f"Failed to open {pdf_path.name}: {e}") from e

    return _clean_text("\n".join(text_parts)), all_tables


def extract_by_keywords(
    pdf_path: Path,
    keywords: list[str],
    overlap_lines: int,
    extract_tables: bool,
) -> tuple[str, list[list[str]], list[int]]:
    """
    Scan all pages for keyword matches (case-insensitive).
    Collect matched pages + overlap_lines of context from adjacent pages.
    Merge contiguous page groups. Extract text + tables from matched pages.

    Args:
        pdf_path:      Absolute path to the PDF file.
        keywords:      List of keyword strings to match (any match → include page).
        overlap_lines: Number of lines to carry over from adjacent pages for context.
        extract_tables: Whether to extract tables.

    Returns:
        (text_content, tables, matched_page_numbers)  — page numbers are 1-indexed.

    Raises:
        PDFExtractionError: if the PDF cannot be opened.
    """
    lower_keywords = [kw.lower() for kw in keywords]

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

            # Step 1: identify which pages contain any keyword
            matched_indices: set[int] = set()
            for i, page in enumerate(pdf.pages):
                text = (page.extract_text() or "").lower()
                if any(kw in text for kw in lower_keywords):
                    matched_indices.add(i)

            if not matched_indices:
                return "", [], []

            # Step 2: expand with overlap (±1 page for context continuity)
            expanded: set[int] = set()
            for idx in matched_indices:
                expanded.add(idx)
                if idx > 0:
                    expanded.add(idx - 1)
                if idx < total_pages - 1:
                    expanded.add(idx + 1)

            sorted_indices = sorted(expanded)

            # Step 3: extract text + tables from matched pages
            text_parts: list[str] = []
            all_tables: list[list[str]] = []

            for idx in sorted_indices:
                page = pdf.pages[idx]
                page_text = page.extract_text() or ""

                # For overlap pages (not directly matched), take only overlap_lines
                if idx not in matched_indices and overlap_lines > 0:
                    lines = page_text.splitlines()
                    if idx < min(matched_indices):
                        page_text = "\n".join(lines[-overlap_lines:])
                    else:
                        page_text = "\n".join(lines[:overlap_lines])

                text_parts.append(page_text)

                if extract_tables and idx in matched_indices:
                    all_tables.extend(_extract_tables_from_page(page))

    except pdfplumber.exceptions.PDFSyntaxError as e:
        raise PDFExtractionError(f"Failed to open {pdf_path.name}: {e}") from e

    matched_page_numbers = [i + 1 for i in sorted_indices]  # convert to 1-indexed
    return _clean_text("\n".join(text_parts)), all_tables, matched_page_numbers


# ---------------------------------------------------------------------------
# Chunk Assembly
# ---------------------------------------------------------------------------

def build_chunk(
    doc_id: str,
    chunk_id: str,
    geography: str,
    source_name: str,
    source_org: str,
    source_pdf: str,
    pages_extracted: list[int],
    extraction_method: str,
    text: str,
    tables: list[list[str]],
    min_chunk_chars: int,
    max_chunk_chars: int,
) -> ChunkResult:
    """
    Assemble a ChunkResult from extracted content.

    Applies min/max character bounds:
      - Below min_chunk_chars → was_skipped = True
      - Above max_chunk_chars → truncate + was_truncated = True
    """
    full_id = f"{doc_id}_{chunk_id}"
    was_truncated = False
    was_skipped = False
    skip_reason = ""

    if len(text) < min_chunk_chars:
        was_skipped = True
        skip_reason = (
            f"Content too short ({len(text)} chars < min {min_chunk_chars}). "
            f"Add page_override to corpus_extraction.yaml for chunk '{full_id}'."
        )

    if len(text) > max_chunk_chars:
        text = text[:max_chunk_chars]
        was_truncated = True

    return ChunkResult(
        doc_id=doc_id,
        chunk_id=chunk_id,
        geography=geography,
        source_name=source_name,
        source_org=source_org,
        source_pdf=source_pdf,
        pages_extracted=pages_extracted,
        extraction_method=extraction_method,
        extracted_at=datetime.now(timezone.utc).isoformat(),
        char_count=len(text),
        tables=tables,
        content=text,
        was_truncated=was_truncated,
        was_skipped=was_skipped,
        skip_reason=skip_reason,
    )


def chunk_result_to_dict(chunk: ChunkResult) -> dict[str, Any]:
    """Serialize a ChunkResult to the canonical .json output schema."""
    return {
        "chunk_id": f"{chunk.doc_id}_{chunk.chunk_id}",
        "doc_id": chunk.doc_id,
        "source_name": chunk.source_name,
        "source_org": chunk.source_org,
        "source_pdf": chunk.source_pdf,
        "geography": chunk.geography,
        "pages_extracted": chunk.pages_extracted,
        "extraction_method": chunk.extraction_method,
        "extracted_at": chunk.extracted_at,
        "char_count": chunk.char_count,
        "tables": chunk.tables,
        "content": chunk.content,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_extraction(
    config_path: Path,
    dry_run: bool = False,
    filter_doc: str | None = None,
) -> ExtractionSummary:
    """
    Main orchestrator. Iterates all documents → all chunks.

    For each chunk:
      1. Resolves page_override vs keyword strategy.
      2. Calls appropriate extraction function.
      3. Builds ChunkResult, validates bounds.
      4. Saves to corpus/processed/{geography}/{doc_id}_{chunk_id}.json

    Args:
        config_path: Path to corpus_extraction.yaml.
        dry_run:     If True, print what would be extracted but write nothing.
        filter_doc:  If set, only process the named document ID.

    Returns:
        ExtractionSummary with counts and warnings.
    """
    config = load_config(config_path)
    extraction_cfg = config["extraction"]
    project_root = config_path.parent.parent   # configs/ → project root

    input_dir = project_root / extraction_cfg["input_dir"]
    output_dir = project_root / extraction_cfg["output_dir"]
    min_chars = extraction_cfg["min_chunk_chars"]
    max_chars = extraction_cfg["max_chunk_chars"]
    overlap_lines = extraction_cfg["overlap_lines"]

    summary = ExtractionSummary()

    for doc_id, doc_cfg in config["documents"].items():

        # Honour --doc filter
        if filter_doc and doc_id != filter_doc:
            continue

        pdf_path = input_dir / doc_cfg["path"]
        if not pdf_path.exists():
            logger.warning(
                "PDF not found, skipping document '%s': %s", doc_id, pdf_path
            )
            summary.warnings.append(
                f"PDF not found for '{doc_id}': {pdf_path}"
            )
            continue

        logger.info("Processing document: %s (%s)", doc_id, pdf_path.name)
        summary.documents_processed += 1

        for chunk_cfg in doc_cfg["chunks"]:
            chunk_id: str = chunk_cfg["id"]
            keywords: list[str] = chunk_cfg["keywords"]
            page_override: list[int] | None = chunk_cfg.get("page_override")
            extract_tables: bool = chunk_cfg.get("extract_tables", False)
            # chunk-level geography override
            geography: str = chunk_cfg.get("output_geography", doc_cfg["geography"])

            full_id = f"{doc_id}_{chunk_id}"
            logger.info("  Extracting chunk: %s", full_id)

            # -----------------------------------------------------------------
            # Extraction — page override takes priority
            # -----------------------------------------------------------------
            try:
                if page_override is not None:
                    text, tables = extract_by_page_range(
                        pdf_path, page_override, extract_tables
                    )
                    pages_used = list(range(page_override[0], page_override[1] + 1))
                    method = "page_override"
                else:
                    text, tables, pages_used = extract_by_keywords(
                        pdf_path, keywords, overlap_lines, extract_tables
                    )
                    method = "keyword"

            except PDFExtractionError as e:
                logger.error("    FAILED: %s", e)
                summary.warnings.append(f"Extraction failed for '{full_id}': {e}")
                continue

            # -----------------------------------------------------------------
            # Build and validate chunk
            # -----------------------------------------------------------------
            chunk = build_chunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                geography=geography,
                source_name=doc_cfg["source_name"],
                source_org=doc_cfg["source_org"],
                source_pdf=str(pdf_path.relative_to(project_root)),
                pages_extracted=pages_used,
                extraction_method=method,
                text=text,
                tables=tables,
                min_chunk_chars=min_chars,
                max_chunk_chars=max_chars,
            )

            if chunk.was_skipped:
                logger.warning("    SKIPPED: %s", chunk.skip_reason)
                summary.chunks_skipped += 1
                summary.warnings.append(chunk.skip_reason)
                continue

            if chunk.was_truncated:
                logger.warning(
                    "    TRUNCATED: '%s' exceeded %d chars, truncated.", full_id, max_chars
                )
                summary.chunks_truncated += 1

            # -----------------------------------------------------------------
            # Dry run — print summary, skip write
            # -----------------------------------------------------------------
            if dry_run:
                print(
                    f"[DRY RUN] {full_id} → "
                    f"processed/{geography}/{full_id}.json "
                    f"({chunk.char_count} chars, "
                    f"{len(chunk.tables)} tables, "
                    f"method={method}, "
                    f"pages={pages_used[:5]}{'...' if len(pages_used) > 5 else ''})"
                )
                summary.chunks_written += 1
                summary.output_counts[geography] = (
                    summary.output_counts.get(geography, 0) + 1
                )
                continue

            # -----------------------------------------------------------------
            # Write output
            # -----------------------------------------------------------------
            geo_dir = output_dir / geography
            geo_dir.mkdir(parents=True, exist_ok=True)
            output_path = geo_dir / f"{full_id}.json"

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunk_result_to_dict(chunk), f, indent=2, ensure_ascii=False)

            logger.info(
                "    Written: %s (%d chars, %d tables)",
                output_path.relative_to(project_root),
                chunk.char_count,
                len(chunk.tables),
            )

            summary.chunks_written += 1
            summary.output_counts[geography] = (
                summary.output_counts.get(geography, 0) + 1
            )

    return summary


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract targeted content from corpus PDFs into structured JSON chunks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full extraction
  python scripts/extract_corpus.py --config configs/corpus_extraction.yaml

  # Single document only
  python scripts/extract_corpus.py --config configs/corpus_extraction.yaml --doc ics209

  # Dry run — prints what would be extracted, writes nothing
  python scripts/extract_corpus.py --config configs/corpus_extraction.yaml --dry-run
        """,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to corpus_extraction.yaml",
    )
    parser.add_argument(
        "--doc",
        type=str,
        default=None,
        help="Process only this document ID (e.g. ics209, scott_burgan)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print extraction plan without writing any files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        logger.info("DRY RUN MODE — no files will be written")

    try:
        summary = run_extraction(
            config_path=args.config,
            dry_run=args.dry_run,
            filter_doc=args.doc,
        )
    except (ConfigValidationError, FileNotFoundError) as e:
        logger.error("Startup failure: %s", e)
        sys.exit(1)

    summary.print_report()

    # Exit with non-zero code if any chunks were skipped — useful for CI gates
    if summary.chunks_skipped > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()