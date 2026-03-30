"""Gemini Developer API adapter — Phase 2 (free-tier API key).

Uses the ``google-genai`` SDK with an API key from env var
``GEMINI_API_KEY``.  No context caching — corpus is injected inline.

Free-tier limits (Gemini 2.5 Flash):
  - 10 requests per minute (RPM)
  - 500 requests per day (RPD)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv

from src.models.obj3_gemini.adapters.base_adapter import LLMAdapter, LLMGenerationError
from src.models.obj3_gemini.context_builder import ContextBundle

load_dotenv()
logger = logging.getLogger(__name__)


class GeminiDevAdapter(LLMAdapter):
    """Phase 2 adapter — calls Gemini Developer API with API key.

    Set ``llm_backend: "gemini_dev"`` in ``reporting_config.yaml``
    and export ``GEMINI_API_KEY`` env var before use.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        gemini_cfg = config.get("gemini_dev", {})
        self._model_name: str = gemini_cfg.get("model", "gemini-2.5-flash")
        self._api_key: str = os.environ.get("GEMINI_API_KEY", "")
        self._client: Any = None  # lazy-init genai.Client

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
        logger.info("Gemini Dev client initialised (model=%s)", self._model_name)

    def generate(self, context_bundle: ContextBundle, schema: dict[str, Any]) -> str:
        """Send context to Gemini Developer API and return raw JSON string.

        Uses ``response_mime_type="application/json"`` and ``response_json_schema``
        in generation config for structured output.
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
                    # Vision: pass raw bytes as inline_data Part
                    content_parts.append(
                        types.Part.from_bytes(
                            data=pf.image_bytes,
                            mime_type=pf.mime_type,
                        )
                    )
                    user_text_parts.append(f"[Image above: {pf.filename}]")
                else:
                    # Text/PDF: inject content inline
                    header = f"=== Uploaded File: {pf.filename} ==="
                    user_text_parts.append(header + "\n" + pf.text_content)

        user_text_parts.append(context_bundle.instruction)
        # Prepend the combined text as the first content part
        content_parts.insert(0, "\n\n".join(user_text_parts))

        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=content_parts,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_json_schema=schema,
                    temperature=0.0,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True,
                    ),
                ),
            )
            raw = response.text
        except Exception as exc:
            raise LLMGenerationError(f"Gemini Dev API error: {exc}") from exc

        if not raw or not raw.strip():
            raise LLMGenerationError("Gemini Dev returned empty response")

        # Validate it's parseable JSON
        try:
            json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMGenerationError(
                f"Gemini Dev returned invalid JSON: {exc}"
            ) from exc

        return raw

    def is_available(self) -> bool:
        """Check if Gemini Developer API is accessible and the target model exists."""
        try:
            from google import genai
        except ImportError:
            logger.warning("google-genai package not installed")
            return False

        if not self._api_key:
            logger.warning("GEMINI_API_KEY env var not set")
            return False

        try:
            client = genai.Client(api_key=self._api_key)
            models = [m.name for m in client.models.list()]
            # Model names are like "models/gemini-2.5-flash"
            target = f"models/{self._model_name}"
            available = any(target in name for name in models)
            if not available:
                logger.warning(
                    "Model %s not found. Available: %s",
                    self._model_name, models[:5],
                )
            return available
        except Exception as exc:
            logger.warning("Gemini Dev health check failed: %s", exc)
            return False
