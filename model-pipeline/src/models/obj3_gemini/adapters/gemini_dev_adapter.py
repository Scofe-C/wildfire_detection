"""Gemini Developer API adapter — Phase 2 (free-tier API key).

Uses the ``google-genai`` SDK with an API key from env var
``GEMINI_API_KEY``.  No context caching — corpus is injected inline.

Smart rate-limit handling:
  - Per-minute limit (429 + "minute") → pause 60s, retry same model
  - Per-day / quota exhausted (429 + "day" | "quota") → switch to next model
  - Model priority list is configurable in ``reporting_config.yaml``

Free-tier limits (defaults, check https://ai.google.dev/pricing):
  - gemini-2.5-flash:       10 RPM / 500 RPD / 250K TPM
  - gemini-2.5-flash-lite:  30 RPM / 1500 RPD / 250K TPM
  - gemini-2.0-flash:       15 RPM / 1500 RPD / 1M TPM
  - gemini-2.0-flash-lite:  30 RPM / 1500 RPD / 1M TPM
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv

from src.models.obj3_gemini.adapters.base_adapter import LLMAdapter, LLMGenerationError
from src.models.obj3_gemini.context_builder import ContextBundle

load_dotenv()
logger = logging.getLogger(__name__)

# Default fallback order (best quality first, highest free-tier quota last)
_DEFAULT_MODEL_PRIORITY: list[str] = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

# How long to wait on a per-minute rate limit before retrying
_MINUTE_LIMIT_PAUSE_SECONDS = 60

# Max per-minute retries before switching model
_MAX_MINUTE_RETRIES = 2


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if exception is a 429 / RESOURCE_EXHAUSTED error."""
    exc_str = str(exc).lower()
    return "429" in exc_str or "resource_exhausted" in exc_str or "resourceexhausted" in exc_str


def _is_per_minute_limit(exc: Exception) -> bool:
    """Detect per-minute rate limit (vs per-day / full quota).

    Google's 429 errors usually contain 'minute' or 'rpm' for per-minute,
    and 'day' or 'rpd' or 'quota' for per-day limits.
    If unclear, assume per-minute (safer — try pausing first).
    """
    exc_str = str(exc).lower()
    # Explicit per-day indicators → NOT per-minute
    if any(kw in exc_str for kw in ("per day", "per_day", "rpd", "daily")):
        return False
    # Explicit per-minute indicators → IS per-minute
    if any(kw in exc_str for kw in ("per minute", "per_minute", "rpm")):
        return True
    # Ambiguous — default to per-minute (will pause and retry once)
    return True


class GeminiDevAdapter(LLMAdapter):
    """Phase 2 adapter — calls Gemini Developer API with API key.

    Set ``llm_backend: "gemini_dev"`` in ``reporting_config.yaml``
    and export ``GEMINI_API_KEY`` env var before use.

    Model fallback priority is configurable via ``gemini_dev.model_priority``
    in the config. The ``gemini_dev.model`` field sets the primary model
    (first in the list). Any models in ``model_priority`` are appended
    after the primary if not already present.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        gemini_cfg = config.get("gemini_dev", {})
        primary_model: str = gemini_cfg.get("model", "gemini-3-flash-preview")
        configured_priority: list[str] = gemini_cfg.get("model_priority", [])

        # Build final priority: primary first, then configured extras, then defaults
        self._model_priority = self._build_priority(primary_model, configured_priority)
        self._api_key: str = os.environ.get("GEMINI_API_KEY", "")
        self._client: Any = None  # lazy-init genai.Client
        # Track models exhausted for the day (reset on next calendar day)
        self._exhausted_models: set[str] = set()
        self._active_model_index: int = 0

    @staticmethod
    def _build_priority(primary: str, configured: list[str]) -> list[str]:
        """Build deduplicated priority list: primary → configured → defaults."""
        seen: set[str] = set()
        result: list[str] = []
        for model in [primary] + configured + _DEFAULT_MODEL_PRIORITY:
            if model not in seen:
                seen.add(model)
                result.append(model)
        return result

    @property
    def active_model(self) -> str:
        """The model currently being used (after any fallbacks)."""
        idx = min(self._active_model_index, len(self._model_priority) - 1)
        return self._model_priority[idx]

    def _ensure_client(self) -> None:
        """Lazy-initialise the Google Gen AI client."""
        if self._client is not None:
            return

        try:
            from google import genai
        except ImportError as exc:
            raise LLMGenerationError(
                "The 'google-genai' package is required for Phase 2. "
                "Install with: pip install google-genai"
            ) from exc

        if not self._api_key:
            raise LLMGenerationError(
                "GEMINI_API_KEY env var is not set. "
                "Get a free API key from https://aistudio.google.com/apikey"
            )

        self._client = genai.Client(api_key=self._api_key)
        logger.info(
            "Gemini Dev client initialised (primary=%s, priority=%s)",
            self._model_priority[0],
            self._model_priority,
        )

    def _next_available_model(self) -> str | None:
        """Advance to the next non-exhausted model. Returns None if all exhausted."""
        while self._active_model_index < len(self._model_priority):
            model = self._model_priority[self._active_model_index]
            if model not in self._exhausted_models:
                return model
            self._active_model_index += 1
        return None

    def _call_model(
        self,
        model_name: str,
        content_parts: list[Any],
        gen_config: Any,
    ) -> str:
        """Single model call. Returns raw JSON string."""
        response = self._client.models.generate_content(
            model=model_name,
            contents=content_parts,
            config=gen_config,
        )
        raw = response.text
        if not raw or not raw.strip():
            raise LLMGenerationError(f"Gemini Dev ({model_name}) returned empty response")
        # Validate parseable JSON
        try:
            json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMGenerationError(
                f"Gemini Dev ({model_name}) returned invalid JSON: {exc}"
            ) from exc
        return raw

    def generate(self, context_bundle: ContextBundle, schema: dict[str, Any]) -> str:
        """Send context to Gemini Developer API and return raw JSON string.

        Rate limit handling:
          1. Per-minute limit → pause 60s and retry (up to 2 times)
          2. Per-day limit → mark model as exhausted, switch to next in priority
          3. All models exhausted → raise LLMGenerationError
        """
        from google.genai import types

        self._ensure_client()

        # Build the system instruction
        system_instruction = context_bundle.system_prompt
        if context_bundle.corpus_text:
            system_instruction += "\n\n--- REFERENCE CORPUS ---\n" + context_bundle.corpus_text

        # Build content parts — text blocks first, then uploaded files (vision)
        user_text_parts: list[str] = [context_bundle.ml_block, context_bundle.data_block]
        if context_bundle.human_block:
            user_text_parts.append("--- OPERATOR INPUT ---\n" + context_bundle.human_block)

        # Inject uploaded files: images as vision Parts, text files inline
        content_parts: list[Any] = []
        if context_bundle.uploaded_files:
            for pf in context_bundle.uploaded_files:
                if pf.is_image and pf.image_bytes:
                    content_parts.append(
                        types.Part.from_bytes(
                            data=pf.image_bytes,
                            mime_type=pf.mime_type,
                        )
                    )
                    user_text_parts.append(f"[Image above: {pf.filename}]")
                else:
                    header = f"=== Uploaded File: {pf.filename} ==="
                    user_text_parts.append(header + "\n" + pf.text_content)

        user_text_parts.append(context_bundle.instruction)
        content_parts.insert(0, "\n\n".join(user_text_parts))

        gen_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_json_schema=schema,
            temperature=0.0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True,
            ),
        )

        # --- Smart fallback loop ---
        while True:
            current_model = self._next_available_model()
            if current_model is None:
                raise LLMGenerationError(
                    f"All Gemini models exhausted for the day. "
                    f"Tried: {self._model_priority}. "
                    f"Exhausted: {sorted(self._exhausted_models)}. "
                    f"Wait for daily quota reset or upgrade to paid tier."
                )

            minute_retries = 0
            while minute_retries <= _MAX_MINUTE_RETRIES:
                try:
                    raw = self._call_model(current_model, content_parts, gen_config)
                    logger.info("Gemini generation succeeded (model=%s)", current_model)
                    return raw
                except LLMGenerationError:
                    # Non-rate-limit errors (empty response, bad JSON) — propagate
                    raise
                except Exception as exc:
                    if not _is_rate_limit_error(exc):
                        raise LLMGenerationError(
                            f"Gemini Dev API error ({current_model}): {exc}"
                        ) from exc

                    if _is_per_minute_limit(exc) and minute_retries < _MAX_MINUTE_RETRIES:
                        minute_retries += 1
                        logger.warning(
                            "Per-minute rate limit on %s (attempt %d/%d). "
                            "Pausing %ds before retry...",
                            current_model,
                            minute_retries,
                            _MAX_MINUTE_RETRIES,
                            _MINUTE_LIMIT_PAUSE_SECONDS,
                        )
                        time.sleep(_MINUTE_LIMIT_PAUSE_SECONDS)
                        continue
                    else:
                        # Per-day limit or exhausted minute retries → switch model
                        reason = "per-day limit" if not _is_per_minute_limit(exc) else "minute retries exhausted"
                        logger.warning(
                            "Rate-limited on %s (%s). Marking exhausted, switching to next model.",
                            current_model, reason,
                        )
                        self._exhausted_models.add(current_model)
                        self._active_model_index += 1
                        break  # → outer while loop picks next model

    def list_available_models(self) -> list[dict[str, str]]:
        """List all Gemini models available with the current API key.

        Returns list of dicts with 'name' and 'display_name' keys.
        Only models supporting generateContent are included.
        """
        self._ensure_client()
        results: list[dict[str, str]] = []
        for m in self._client.models.list():
            actions = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
            if "generateContent" in actions:
                results.append({
                    "name": m.name,
                    "display_name": getattr(m, "display_name", m.name),
                })
        return results

    def is_available(self) -> bool:
        """Check if Gemini Developer API is accessible and the primary model exists."""
        try:
            from google import genai  # noqa: F811
        except ImportError:
            logger.warning("google-genai package not installed")
            return False

        if not self._api_key:
            logger.warning("GEMINI_API_KEY env var not set")
            return False

        try:
            client = genai.Client(api_key=self._api_key)
            available_names: list[str] = []
            for m in client.models.list():
                actions = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
                if "generateContent" in actions:
                    available_names.append(m.name)
            primary = self._model_priority[0]
            target = f"models/{primary}"
            found = any(target in name for name in available_names)
            if not found:
                logger.warning(
                    "Model %s not found. Available: %s",
                    primary, available_names[:5],
                )
            return found
        except Exception as exc:
            logger.warning("Gemini Dev health check failed: %s", exc)
            return False