"""
distill_corpus.py — Corpus Distillation for OBJ-3 RAG Context

Reads extracted JSON chunks from corpus/processed/ and uses the Gemini Dev
API to compress each chunk's ``content`` field into a shorter
``distilled_content`` field, written back in-place.

The distilled content is what gets injected into the LLM context window,
reducing token usage while preserving the most decision-relevant information.

Locked parameters (from Session 1):
  - Target: 8,000 chars per chunk
  - Scope: all chunks
  - Output: ``distilled_content`` field added in-place to each JSON file

Usage:
    # Distill all chunks
    python scripts/distill_corpus.py

    # Distill a single file
    python scripts/distill_corpus.py --file corpus/processed/shared/irpg_risk_management.json

    # Dry run — show what would be distilled, no writes
    python scripts/distill_corpus.py --dry-run

    # Force re-distill chunks that already have distilled_content
    python scripts/distill_corpus.py --force

Dependencies:
    google-genai, python-dotenv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("distill_corpus")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_CHARS = 8_000
GEMINI_MODEL = "gemini-2.5-flash"
# Free tier: 10 RPM, so pace at ~7s between calls to stay safe
RATE_LIMIT_DELAY = 7.0

DISTILL_PROMPT = """\
You are a technical editor for a wildfire emergency management system.
Your task is to distill the following reference document into a shorter
version that preserves ALL decision-relevant information for automated
disaster report generation.

RULES:
1. Target length: {target_chars} characters (hard limit: {hard_limit} characters).
2. Preserve ALL: numerical thresholds, definitions, field names, ICS form
   references, risk categories, resource types, and procedural steps.
3. Remove: redundant examples, verbose introductions, repeated context,
   formatting artifacts, and non-actionable background.
4. Keep the original structure (headings, lists) where it aids comprehension.
5. Do NOT add commentary, opinions, or information not in the source.
6. Output ONLY the distilled text — no preamble, no markdown fences.

SOURCE DOCUMENT ({source_name}, {char_count} chars):
---
{content}
---

Distill this to ~{target_chars} characters while preserving all
decision-relevant information.
"""


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def get_gemini_client():
    """Lazy-initialise and return a google.genai.Client."""
    try:
        from google import genai
    except ImportError:
        logger.error(
            "google-genai package not installed. "
            "Install with: pip install google-genai"
        )
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.error(
            "GEMINI_API_KEY env var not set. "
            "Get a free key from https://aistudio.google.com/apikey"
        )
        sys.exit(1)

    return genai.Client(api_key=api_key)


def distill_chunk(client, content: str, source_name: str) -> str:
    """Send content to Gemini and return the distilled text."""
    from google.genai import types

    prompt = DISTILL_PROMPT.format(
        target_chars=TARGET_CHARS,
        hard_limit=TARGET_CHARS + 2000,  # Allow some overflow
        source_name=source_name,
        char_count=len(content),
        content=content,
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
        ),
    )

    result = response.text.strip()
    if not result:
        raise RuntimeError("Gemini returned empty response")
    return result


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def find_chunks(corpus_dir: Path) -> list[Path]:
    """Recursively find all .json chunk files under corpus_dir."""
    return sorted(corpus_dir.rglob("*.json"))


def process_chunk(
    client,
    chunk_path: Path,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """Distill a single chunk file. Returns a summary dict."""
    with open(chunk_path, encoding="utf-8") as f:
        data = json.load(f)

    source_name = data.get("source_name", chunk_path.stem)
    content = data.get("content", "")

    if not content:
        return {"file": chunk_path.name, "status": "SKIPPED", "reason": "no content"}

    if data.get("distilled_content") and not force:
        return {
            "file": chunk_path.name,
            "status": "SKIPPED",
            "reason": "already distilled (use --force to re-distill)",
        }

    original_chars = len(content)

    if original_chars <= TARGET_CHARS:
        # Already short enough — copy content as-is
        if not dry_run:
            data["distilled_content"] = content
            data["distilled_chars"] = original_chars
            data["distill_ratio"] = 1.0
            with open(chunk_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        return {
            "file": chunk_path.name,
            "status": "COPIED",
            "original": original_chars,
            "distilled": original_chars,
            "ratio": 1.0,
        }

    if dry_run:
        return {
            "file": chunk_path.name,
            "status": "WOULD_DISTILL",
            "original": original_chars,
            "target": TARGET_CHARS,
        }

    # Call Gemini
    logger.info(
        "Distilling %s (%d chars → ~%d target)...",
        chunk_path.name, original_chars, TARGET_CHARS,
    )

    distilled = distill_chunk(client, content, source_name)
    distilled_chars = len(distilled)
    ratio = distilled_chars / original_chars

    # Write back in-place
    data["distilled_content"] = distilled
    data["distilled_chars"] = distilled_chars
    data["distill_ratio"] = round(ratio, 3)
    with open(chunk_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(
        "  → %d chars (%.0f%% of original)",
        distilled_chars, ratio * 100,
    )

    return {
        "file": chunk_path.name,
        "status": "DISTILLED",
        "original": original_chars,
        "distilled": distilled_chars,
        "ratio": round(ratio, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill corpus chunks using Gemini Dev API",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Distill a single JSON file instead of all chunks",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("corpus/processed"),
        help="Root directory of extracted chunks (default: corpus/processed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be distilled without writing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-distill chunks that already have distilled_content",
    )
    args = parser.parse_args()

    # Resolve paths
    if args.file:
        if not args.file.exists():
            logger.error("File not found: %s", args.file)
            sys.exit(1)
        chunks = [args.file]
    else:
        if not args.corpus_dir.is_dir():
            logger.error("Corpus directory not found: %s", args.corpus_dir)
            sys.exit(1)
        chunks = find_chunks(args.corpus_dir)

    if not chunks:
        logger.warning("No JSON files found.")
        sys.exit(0)

    logger.info("Found %d chunk(s) to process", len(chunks))

    if args.dry_run:
        logger.info("DRY RUN — no files will be modified")
        client = None
    else:
        client = get_gemini_client()

    # Process
    results = []
    distilled_count = 0
    for i, chunk_path in enumerate(chunks):
        result = process_chunk(
            client, chunk_path, dry_run=args.dry_run, force=args.force,
        )
        results.append(result)

        if result["status"] == "DISTILLED":
            distilled_count += 1
            # Rate limit — only after actual API calls
            if i < len(chunks) - 1:
                logger.info(
                    "  Rate limit pause (%.0fs) — %d/%d done",
                    RATE_LIMIT_DELAY, i + 1, len(chunks),
                )
                time.sleep(RATE_LIMIT_DELAY)

    # Summary
    print("\n" + "=" * 70)
    print("DISTILLATION SUMMARY")
    print("=" * 70)

    total_original = 0
    total_distilled = 0

    for r in results:
        status = r["status"]
        name = r["file"]
        if status == "DISTILLED":
            print(
                f"  ✓ {name}: {r['original']:,} → {r['distilled']:,} chars "
                f"({r['ratio']:.0%})"
            )
            total_original += r["original"]
            total_distilled += r["distilled"]
        elif status == "COPIED":
            print(f"  = {name}: {r['original']:,} chars (already under target)")
            total_original += r["original"]
            total_distilled += r["distilled"]
        elif status == "WOULD_DISTILL":
            print(f"  ? {name}: {r['original']:,} chars → ~{r['target']:,} target")
            total_original += r["original"]
        else:
            print(f"  - {name}: SKIPPED ({r.get('reason', 'unknown')})")

    print(f"\n  Chunks processed: {len(results)}")
    print(f"  API calls made:  {distilled_count}")
    if total_original > 0 and not args.dry_run:
        print(f"  Total original:  {total_original:,} chars")
        print(f"  Total distilled: {total_distilled:,} chars")
        print(f"  Overall ratio:   {total_distilled / total_original:.0%}")
    print("=" * 70)


if __name__ == "__main__":
    main()