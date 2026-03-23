"""
audit_corpus_pages.py — Corpus PDF Page Auditor

Prints a page-by-page preview of each PDF in corpus/raw/ so you can
identify the correct page ranges to set as page_override in
corpus_extraction.yaml.

Usage:
    # Audit all 6 PDFs
    python scripts/audit_corpus_pages.py

    # Audit a single PDF
    python scripts/audit_corpus_pages.py --pdf corpus/raw/nwcg/pms461.pdf

    # Save output to file for easier browsing
    python scripts/audit_corpus_pages.py --pdf corpus/raw/usfs/rmrs_gtr153.pdf > audit_scott_burgan.txt

    # Show more characters per page (default 300)
    python scripts/audit_corpus_pages.py --pdf corpus/raw/nwcg/pms461.pdf --chars 500

Output format per page:
    Page  5 | [T] [2 tables] | LCES — Lookouts, Communications, Escape Routes...
             └─ [T] = has text layer  [F] = form fields detected  [I] = image only

After running:
    1. Scan the output for pages where your target content starts.
    2. Note the start/end page numbers (1-indexed, inclusive).
    3. Set page_override: [start, end] in corpus_extraction.yaml.
    4. Re-run extract_corpus.py --doc <doc_id> to regenerate the chunk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("ERROR: pdfplumber not installed. Run: pip install pdfplumber --break-system-packages")
    sys.exit(1)


# ---------------------------------------------------------------------------
# PDF registry — maps doc_id to relative path from project root
# ---------------------------------------------------------------------------

PDF_REGISTRY = {
    "ics209":       "corpus/raw/ics/ICS 209 Fillable PDF Form.pdf",
    "nims_booklet": "corpus/raw/ics/nims ics forms booklet (v3).pdf",
    "irpg":         "corpus/raw/nwcg/pms461.pdf",
    "imsr_guide":   "corpus/raw/nifc/Understanding the IMSR 2024.pdf",
    "fema_nri":     "corpus/raw/fema/fema_national-risk-index_technical-documentation.pdf",
    "scott_burgan": "corpus/raw/usfs/rmrs_gtr153.pdf",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_form_fields(page) -> int:
    """Return number of form field annotations on a page (0 if none)."""
    try:
        annots = page.annots or []
        return sum(1 for a in annots if a.get("data", {}).get("Subtype") in ("Widget",))
    except Exception:
        return 0


def _detect_tables(page) -> int:
    """Return number of tables detected on a page."""
    try:
        tables = page.extract_tables()
        return len(tables) if tables else 0
    except Exception:
        return 0


def _page_summary(page, max_chars: int) -> str:
    """Return a one-line summary of a page's content."""
    text = (page.extract_text() or "").strip()
    text = " ".join(text.split())  # collapse whitespace

    if not text:
        return "[NO TEXT — likely image/scan or form fields only]"

    snippet = text[:max_chars]
    if len(text) > max_chars:
        snippet += "…"
    return snippet


def audit_pdf(pdf_path: Path, max_chars: int = 300) -> None:
    """Print a page-by-page audit of a single PDF."""
    if not pdf_path.exists():
        print(f"\n[SKIP] File not found: {pdf_path}")
        return

    print(f"\n{'=' * 80}")
    print(f"PDF: {pdf_path}")
    print(f"{'=' * 80}")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
            print(f"Total pages: {total}\n")

            for i, page in enumerate(pdf.pages):
                page_num = i + 1  # 1-indexed, matches page_override format

                text = (page.extract_text() or "").strip()
                tables = _detect_tables(page)
                form_fields = _detect_form_fields(page)

                # Build flags
                flags = []
                if text:
                    flags.append("T")       # has text layer
                if tables:
                    flags.append(f"{tables}tbl")  # number of tables
                if form_fields:
                    flags.append(f"{form_fields}fld")  # form fields
                if not text and not form_fields:
                    flags.append("IMG")     # probably scanned image

                flag_str = f"[{' '.join(flags)}]" if flags else "[???]"

                # Content preview
                summary = _page_summary(page, max_chars)

                print(f"Page {page_num:4d} | {flag_str:<16} | {summary}")

    except Exception as e:
        print(f"[ERROR] Could not open {pdf_path}: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit corpus PDFs — print page-by-page previews to find correct page ranges.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Audit all 6 PDFs (may take 30-60 seconds)
  python scripts/audit_corpus_pages.py

  # Audit a single document by ID
  python scripts/audit_corpus_pages.py --doc irpg

  # Audit a specific PDF file
  python scripts/audit_corpus_pages.py --pdf corpus/raw/usfs/rmrs_gtr153.pdf

  # Save output to file for easier review
  python scripts/audit_corpus_pages.py --doc irpg > audit_irpg.txt
  python scripts/audit_corpus_pages.py --doc fema_nri > audit_fema_nri.txt

Available doc IDs: ics209, nims_booklet, irpg, imsr_guide, fema_nri, scott_burgan
        """,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--doc",
        type=str,
        choices=list(PDF_REGISTRY.keys()),
        help="Audit a single document by its doc_id",
    )
    group.add_argument(
        "--pdf",
        type=Path,
        help="Audit a specific PDF file path",
    )
    parser.add_argument(
        "--chars",
        type=int,
        default=300,
        help="Number of characters to show per page preview (default: 300)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root directory (default: current directory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.pdf:
        # Single explicit file
        audit_pdf(args.pdf, max_chars=args.chars)

    elif args.doc:
        # Single doc by ID
        rel_path = PDF_REGISTRY[args.doc]
        pdf_path = args.project_root / rel_path
        audit_pdf(pdf_path, max_chars=args.chars)

    else:
        # All 6 PDFs
        print("Auditing all 6 corpus PDFs...")
        print("Tip: pipe output to a file for easier review:")
        print("  python scripts/audit_corpus_pages.py > full_audit.txt\n")

        for doc_id, rel_path in PDF_REGISTRY.items():
            pdf_path = args.project_root / rel_path
            audit_pdf(pdf_path, max_chars=args.chars)

    print("\nDone. Use the page numbers above to set page_override in corpus_extraction.yaml.")
    print("Format: page_override: [start, end]  (1-indexed, inclusive)")


if __name__ == "__main__":
    main()