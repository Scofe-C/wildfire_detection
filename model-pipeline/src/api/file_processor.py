"""file_processor.py — Smart file routing based on LLM backend capabilities.

Each backend has different context limits and vision support:

  ollama      — text only, small budget (no vision on qwen3:8b text model)
  gemini_dev  — vision + large text budget (inline base64)
  vertex_ai   — vision + very large budget (GCS upload)

Files are processed at upload time and stored as UploadedFile objects.
Each adapter then decides how to pass them to the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-backend capabilities
# ---------------------------------------------------------------------------

BACKEND_CAPS: dict[str, dict[str, Any]] = {
    "ollama": {
        "vision": False,
        "max_chars_per_file": 3_000,
        "max_total_chars": 8_000,
        "max_images": 0,
        "note": "Vision not available — image content described as metadata only.",
    },
    "gemini_dev": {
        "vision": True,
        "max_chars_per_file": 60_000,
        "max_total_chars": 300_000,
        "max_images": 10,
        "note": "",
    },
    "vertex_ai": {
        "vision": True,
        "max_chars_per_file": 100_000,
        "max_total_chars": 600_000,
        "max_images": 20,
        "note": "",
    },
}


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def extract_pdf_text(content: bytes, max_chars: int) -> str:
    """Extract plain text from a PDF using pypdf (best-effort)."""
    try:
        import io  # noqa: PLC0415

        import pypdf  # noqa: PLC0415
        reader = pypdf.PdfReader(io.BytesIO(content))
        pages = []
        total = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
            total += len(text)
            if total >= max_chars:
                break
        full = "\n".join(pages)
        return full[:max_chars]
    except ImportError:
        logger.warning("pypdf not installed — PDF text extraction unavailable. pip install pypdf")
        return "[PDF content — install pypdf for text extraction]"
    except Exception as exc:
        logger.warning("PDF extraction failed: %s", exc)
        return f"[PDF extraction failed: {exc}]"


def extract_text_content(
    filename: str,
    content: bytes,
    mime_type: str,
    max_chars: int,
) -> str:
    """Extract text from any supported file type."""
    if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
        return extract_pdf_text(content, max_chars)

    if mime_type.startswith("text/") or filename.lower().endswith(
        (".txt", ".csv", ".json", ".geojson", ".md", ".xml", ".log")
    ):
        try:
            return content.decode("utf-8", errors="replace")[:max_chars]
        except Exception:
            return content.decode("latin-1", errors="replace")[:max_chars]

    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ProcessedFile:
    """A file processed for LLM injection."""
    filename: str
    mime_type: str
    # For text injection (all backends)
    text_content: str
    # For vision injection (Gemini/Vertex only)
    image_bytes: bytes | None
    is_image: bool


def process_files(
    raw_files: list[tuple[str, bytes, str]],  # (filename, content, mime_type)
    backend: str,
) -> list[ProcessedFile]:
    """Process uploaded files according to backend capabilities.

    Parameters
    ----------
    raw_files:
        List of (filename, content_bytes, mime_type) tuples.
    backend:
        Active LLM backend name.

    Returns
    -------
    List of ProcessedFile ready for adapter injection.
    """
    caps = BACKEND_CAPS.get(backend, BACKEND_CAPS["ollama"])
    max_per = caps["max_chars_per_file"]
    max_total = caps["max_total_chars"]
    max_images = caps["max_images"]
    vision = caps["vision"]

    processed: list[ProcessedFile] = []
    total_chars = 0
    image_count = 0

    for filename, content, mime_type in raw_files:
        is_image = mime_type.startswith("image/") or filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff")
        )

        if is_image:
            if vision and image_count < max_images:
                processed.append(ProcessedFile(
                    filename=filename,
                    mime_type=mime_type,
                    text_content=f"[Image: {filename}]",
                    image_bytes=content,
                    is_image=True,
                ))
                image_count += 1
                logger.info("File %s → vision (image #%d)", filename, image_count)
            else:
                reason = (
                    f"max {max_images} images reached" if image_count >= max_images
                    else caps["note"]
                )
                processed.append(ProcessedFile(
                    filename=filename,
                    mime_type=mime_type,
                    text_content=(
                        f"[Image: {filename} — {reason}. "
                        f"File size: {len(content):,} bytes. "
                        f"Use Gemini backend to enable visual analysis.]"
                    ),
                    image_bytes=None,
                    is_image=True,
                ))
                logger.info("File %s → text placeholder (no vision)", filename)
        else:
            # Text/PDF/data file
            remaining = min(max_per, max_total - total_chars)
            if remaining <= 0:
                processed.append(ProcessedFile(
                    filename=filename,
                    mime_type=mime_type,
                    text_content=f"[{filename} — text budget exhausted, file not included]",
                    image_bytes=None,
                    is_image=False,
                ))
                continue

            text = extract_text_content(filename, content, mime_type, remaining)
            total_chars += len(text)
            processed.append(ProcessedFile(
                filename=filename,
                mime_type=mime_type,
                text_content=text,
                image_bytes=None,
                is_image=False,
            ))
            logger.info(
                "File %s → text (%d chars, %s)",
                filename, len(text), mime_type,
            )

    return processed
