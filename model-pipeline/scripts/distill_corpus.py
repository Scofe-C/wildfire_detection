"""
distill_corpus.py — Corpus Distillation for OBJ-3 RAG Context

Reads extracted JSON chunks from corpus/processed/ and uses an LLM backend
(Ollama local or Gemini Dev API) to compress each chunk's ``content`` field
into a shorter ``distilled_content`` field, written back in-place.

The distilled content is what gets injected into the LLM context window,
reducing token usage while preserving the most decision-relevant information.

Locked parameters (from Session 1):
  - Target: 8,000 chars per chunk
  - Scope: all chunks
  - Output: ``distilled_content`` field added in-place to each JSON file

Usage:
    # Distill all chunks using local Ollama (default)
    python scripts/distill_corpus.py

    # Distill using Gemini Dev API
    python scripts/distill_corpus.py --backend gemini

    # Distill a single file
    python scripts/distill_corpus.py --file corpus/processed/shared/irpg_risk_management.json

    # Dry run — show what would be distilled, no writes
    python scripts/distill_corpus.py --dry-run

    # Force re-distill chunks that already have distilled_content
    python scripts/distill_corpus.py --force

Dependencies:
    ollama (default) or google-genai + python-dotenv (--backend gemini)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

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
MIN_OUTPUT_CHARS = 4_000  # Reject outputs shorter than this and retry
GEMINI_MODEL = "gemini-2.5-flash"
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_BASE_URL = "http://localhost:11434"
# Gemini free tier: 10 RPM, so pace at ~7s between calls to stay safe.
# Ollama is local — 1s delay is enough to avoid overwhelming the GPU.
RATE_LIMIT_DELAY_GEMINI = 7.0
RATE_LIMIT_DELAY_OLLAMA = 1.0

# Max input chars per LLM call — Qwen3:8b has ~32K token context (~100K chars).
# Leave room for prompt + output by capping input at 60K chars per segment.
OLLAMA_MAX_INPUT_CHARS = 60_000
GEMINI_MAX_INPUT_CHARS = 800_000  # Gemini 2.5 Flash has 1M token context

DISTILL_PROMPT = """\
You are a technical editor for a wildfire emergency management system.
Your task is to distill the following reference document into a shorter
version that preserves ALL decision-relevant information for automated
disaster report generation.

CRITICAL LENGTH REQUIREMENTS:
- Your output MUST be between {min_chars} and {hard_limit} characters.
- Target: ~{target_chars} characters.
- If the source is dense with data, KEEP MORE rather than less.
- An output under {min_chars} characters is TOO SHORT and will be rejected.

WHAT TO PRESERVE (mandatory — do NOT omit any of these):
- ALL numerical thresholds, scores, formulas, and ranges
- ALL definitions and field names
- ALL ICS form references and resource type codes
- ALL risk categories, severity levels, and classification criteria
- ALL procedural steps and decision criteria
- ALL table data (preserve as structured lists)

WHAT TO REMOVE:
- Redundant examples that repeat the same pattern
- Verbose introductions and background context
- Repeated headers and formatting artifacts
- Non-actionable disclaimers and boilerplate

FORMATTING:
- Keep headings and lists for structure
- Output ONLY the distilled text
- No preamble like "Here is the distilled version"
- No markdown code fences wrapping the output

SOURCE DOCUMENT ({source_name}, {char_count} chars):
---
{content}
---

Distill this to ~{target_chars} characters. Output MUST be at least {min_chars} characters.
"""

MERGE_PROMPT = """\
You are a technical editor. Below are {segment_count} distilled segments
from the same reference document "{source_name}". Merge them into a single
coherent distilled document.

RULES:
- Target length: ~{target_chars} characters (minimum {min_chars}).
- Remove duplicates across segments but preserve ALL unique information.
- Keep the structure (headings, lists).
- Output ONLY the merged text — no preamble, no code fences.

SEGMENTS:
---
{segments}
---

Merge into a single document of ~{target_chars} characters.
"""


# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

def get_gemini_client():
    """Lazy-initialise and return a google.genai.Client."""
    from dotenv import load_dotenv

    load_dotenv()

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


def get_ollama_client():
    """Lazy-initialise and return an ollama.Client."""
    try:
        import ollama as ollama_lib
    except ImportError:
        logger.error(
            "ollama package not installed. "
            "Install with: pip install ollama"
        )
        sys.exit(1)

    client = ollama_lib.Client(host=OLLAMA_BASE_URL)
    # Verify the model is available
    try:
        model_list = client.list()
        available = [m.model for m in model_list.models]
        if not any(OLLAMA_MODEL in name for name in available):
            logger.error(
                "Model %s not found in Ollama. Available: %s. "
                "Pull with: ollama pull %s",
                OLLAMA_MODEL, available, OLLAMA_MODEL,
            )
            sys.exit(1)
    except Exception as exc:
        logger.error("Cannot connect to Ollama at %s: %s", OLLAMA_BASE_URL, exc)
        sys.exit(1)

    return client


def _build_distill_prompt(content: str, source_name: str, target: int = TARGET_CHARS) -> str:
    """Build the distillation prompt for any backend."""
    min_chars = max(target // 2, MIN_OUTPUT_CHARS)
    return DISTILL_PROMPT.format(
        target_chars=target,
        min_chars=min_chars,
        hard_limit=target + 2000,
        source_name=source_name,
        char_count=len(content),
        content=content,
    )


def _build_merge_prompt(segments: list[str], source_name: str) -> str:
    """Build the merge prompt for combining distilled segments."""
    joined = "\n\n---SEGMENT BREAK---\n\n".join(segments)
    min_chars = max(TARGET_CHARS // 2, MIN_OUTPUT_CHARS)
    return MERGE_PROMPT.format(
        segment_count=len(segments),
        source_name=source_name,
        target_chars=TARGET_CHARS,
        min_chars=min_chars,
        segments=joined,
    )


def _split_into_segments(content: str, max_chars: int) -> list[str]:
    """Split content into segments that fit within the LLM context window.

    Splits on paragraph boundaries (double newline) to avoid cutting mid-sentence.
    """
    if len(content) <= max_chars:
        return [content]

    paragraphs = content.split("\n\n")
    segments: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # +2 for the \n\n separator
        if current_len + para_len > max_chars and current:
            segments.append("\n\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        segments.append("\n\n".join(current))

    return segments


def _call_ollama(client, system: str, user: str) -> str:
    """Single Ollama chat call."""
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.0, "num_ctx": 32768},
    )
    result = response.message.content.strip()
    if not result:
        raise RuntimeError("Ollama returned empty response")
    # Strip thinking tags if present (Qwen3 sometimes outputs <think>...</think>)
    import re
    result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
    return result


def _call_gemini(client, prompt: str) -> str:
    """Single Gemini generate call."""
    from google.genai import types

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True,
            ),
        ),
    )
    result = response.text.strip()
    if not result:
        raise RuntimeError("Gemini returned empty response")
    return result


def distill_chunk_gemini(client, content: str, source_name: str) -> str:
    """Send content to Gemini and return the distilled text.

    Gemini has a large context window, so chunking is rarely needed.
    """
    segments = _split_into_segments(content, GEMINI_MAX_INPUT_CHARS)

    if len(segments) == 1:
        prompt = _build_distill_prompt(content, source_name)
        return _call_gemini(client, prompt)

    # Multi-segment: distill each, then merge
    distilled_parts = []
    per_segment_target = max(TARGET_CHARS, TARGET_CHARS * 2 // len(segments))
    for i, seg in enumerate(segments):
        logger.info("    Gemini segment %d/%d (%d chars)...", i + 1, len(segments), len(seg))
        prompt = _build_distill_prompt(seg, f"{source_name} [part {i+1}/{len(segments)}]", per_segment_target)
        distilled_parts.append(_call_gemini(client, prompt))

    merge_prompt = _build_merge_prompt(distilled_parts, source_name)
    return _call_gemini(client, merge_prompt)


def distill_chunk_ollama(client, content: str, source_name: str) -> str:
    """Send content to local Ollama and return the distilled text.

    For documents exceeding OLLAMA_MAX_INPUT_CHARS, splits into segments,
    distills each one, then merges into a final distilled document.
    """
    system_msg = (
        "You are a technical editor for wildfire emergency management. "
        "Output ONLY the distilled text — no preamble, no thinking, "
        "no markdown fences, no commentary. Be thorough and detailed."
    )

    segments = _split_into_segments(content, OLLAMA_MAX_INPUT_CHARS)

    if len(segments) == 1:
        prompt = _build_distill_prompt(content, source_name)
        result = _call_ollama(client, system_msg, prompt)
        # Retry once if output is too short
        if len(result) < MIN_OUTPUT_CHARS:
            logger.warning(
                "    Output too short (%d chars < %d min), retrying with emphasis...",
                len(result), MIN_OUTPUT_CHARS,
            )
            prompt += (
                f"\n\nWARNING: Your previous attempt was only {len(result)} characters. "
                f"That is far too short. You MUST output at least {MIN_OUTPUT_CHARS} characters. "
                "Include ALL data tables, thresholds, and definitions."
            )
            result = _call_ollama(client, system_msg, prompt)
        return result

    # Multi-segment: distill each segment, then merge
    logger.info("    Large document — splitting into %d segments", len(segments))
    distilled_parts: list[str] = []
    # Each segment gets a proportional target so the merge stays near TARGET_CHARS
    per_segment_target = max(TARGET_CHARS, (TARGET_CHARS * 2) // len(segments))

    for i, seg in enumerate(segments):
        logger.info(
            "    Segment %d/%d (%d chars → ~%d target)...",
            i + 1, len(segments), len(seg), per_segment_target,
        )
        prompt = _build_distill_prompt(
            seg,
            f"{source_name} [part {i+1}/{len(segments)}]",
            per_segment_target,
        )
        part = _call_ollama(client, system_msg, prompt)
        distilled_parts.append(part)
        time.sleep(RATE_LIMIT_DELAY_OLLAMA)

    # Merge step
    logger.info("    Merging %d distilled segments...", len(distilled_parts))
    merge_prompt = _build_merge_prompt(distilled_parts, source_name)
    merged = _call_ollama(client, system_msg, merge_prompt)
    return merged


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
    backend: str = "ollama",
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

    # Call LLM backend
    logger.info(
        "Distilling %s (%d chars → ~%d target) via %s...",
        chunk_path.name, original_chars, TARGET_CHARS, backend,
    )

    max_input = OLLAMA_MAX_INPUT_CHARS if backend == "ollama" else GEMINI_MAX_INPUT_CHARS
    if original_chars > max_input:
        n_segments = (original_chars // max_input) + 1
        logger.info(
            "  Document exceeds %dK context — will split into ~%d segments",
            max_input // 1000, n_segments,
        )

    distill_fn = distill_chunk_ollama if backend == "ollama" else distill_chunk_gemini
    distilled = distill_fn(client, content, source_name)
    distilled_chars = len(distilled)
    ratio = distilled_chars / original_chars

    # Quality gate: warn if suspiciously short
    if distilled_chars < MIN_OUTPUT_CHARS:
        logger.warning(
            "  ⚠ Output is only %d chars (target: %d, min: %d) — may have lost info",
            distilled_chars, TARGET_CHARS, MIN_OUTPUT_CHARS,
        )

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
        description="Distill corpus chunks using Ollama (default) or Gemini Dev API",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "gemini", "hybrid"],
        default="hybrid",
        help=(
            "LLM backend: 'hybrid' (default) uses Gemini for large docs > 60K chars, "
            "Ollama for the rest. 'ollama' or 'gemini' forces one backend for all."
        ),
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

    logger.info("Found %d chunk(s) to process (backend=%s)", len(chunks), args.backend)

    if args.dry_run:
        logger.info("DRY RUN — no files will be modified")
        ollama_client = None
        gemini_client = None
    else:
        ollama_client = None
        gemini_client = None
        if args.backend in ("ollama", "hybrid"):
            ollama_client = get_ollama_client()
        if args.backend in ("gemini", "hybrid"):
            gemini_client = get_gemini_client()

    # Process
    results = []
    distilled_count = 0
    gemini_calls = 0
    ollama_calls = 0

    for i, chunk_path in enumerate(chunks):
        # In hybrid mode, pick backend based on content size
        effective_backend = args.backend
        if args.backend == "hybrid" and not args.dry_run:
            with open(chunk_path, encoding="utf-8") as f:
                peek = json.load(f)
            content_len = len(peek.get("content", ""))
            if content_len > OLLAMA_MAX_INPUT_CHARS:
                effective_backend = "gemini"
                logger.info(
                    "  [hybrid] %s (%dK chars) → Gemini (too large for Ollama)",
                    chunk_path.name, content_len // 1000,
                )
            else:
                effective_backend = "ollama"

        client = gemini_client if effective_backend == "gemini" else ollama_client

        result = process_chunk(
            client, chunk_path,
            backend=effective_backend, dry_run=args.dry_run, force=args.force,
        )
        results.append(result)

        if result["status"] == "DISTILLED":
            distilled_count += 1
            if effective_backend == "gemini":
                gemini_calls += 1
            else:
                ollama_calls += 1
            # Rate limit — longer pause for Gemini free tier
            rate_delay = (
                RATE_LIMIT_DELAY_GEMINI if effective_backend == "gemini"
                else RATE_LIMIT_DELAY_OLLAMA
            )
            if i < len(chunks) - 1:
                logger.info(
                    "  Rate limit pause (%.0fs) — %d/%d done",
                    rate_delay, i + 1, len(chunks),
                )
                time.sleep(rate_delay)

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
    if args.backend == "hybrid":
        print(f"    Gemini calls:  {gemini_calls} (free tier: 500/day)")
        print(f"    Ollama calls:  {ollama_calls} (local, unlimited)")
    if total_original > 0 and not args.dry_run:
        print(f"  Total original:  {total_original:,} chars")
        print(f"  Total distilled: {total_distilled:,} chars")
        print(f"  Overall ratio:   {total_distilled / total_original:.0%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
