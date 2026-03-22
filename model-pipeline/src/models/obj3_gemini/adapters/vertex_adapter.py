"""Vertex AI adapter — Phase 3 (GCP project, context caching).

Uses the ``google-genai`` SDK (unified Gemini SDK) with Vertex AI backend.
Enables context caching for the RAG corpus to reduce cost on repeated calls.

Prerequisites:
  - GCP project with billing or Vertex AI Express Mode (90-day free)
  - ``GOOGLE_CLOUD_PROJECT`` env var set
  - ``gcloud auth application-default login`` completed
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from src.models.obj3_gemini.adapters.base_adapter import LLMAdapter, LLMGenerationError
from src.models.obj3_gemini.context_builder import ContextBundle
from src.models.obj3_gemini.corpus_loader import (
    CacheCreationError,
    CorpusDocument,
    estimate_corpus_tokens,
)

logger = logging.getLogger(__name__)

# Minimum token count for Vertex AI context caching (API requirement)
_MIN_CACHE_TOKENS = 2_048


class VertexAdapter(LLMAdapter):
    """Phase 3 adapter — calls Vertex AI with optional context caching.

    Set ``llm_backend: "vertex_ai"`` in ``reporting_config.yaml`` and
    configure GCP project settings.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        vertex_cfg = config.get("vertex_ai", {})
        self._model_name: str = vertex_cfg.get("model", "gemini-2.5-flash")
        self._project_id: str = vertex_cfg.get(
            "project_id", ""
        ) or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self._location: str = vertex_cfg.get("location", "us-central1")
        self._ttl_seconds: int = int(vertex_cfg.get("corpus_cache_ttl_seconds", 3600))
        self._client: Any = None  # lazy-init genai.Client
        self._cache_name: str | None = None

    def _ensure_client(self) -> None:
        """Lazy-initialise the google-genai Client with Vertex AI backend."""
        if self._client is not None:
            return

        try:
            from google import genai
        except ImportError as exc:
            raise LLMGenerationError(
                "The 'google-genai' package is required for Phase 3. "
                "Install with: pip install google-genai"
            ) from exc

        if not self._project_id:
            raise LLMGenerationError(
                "GCP project ID is not set. Set vertex_ai.project_id in "
                "reporting_config.yaml or export GOOGLE_CLOUD_PROJECT."
            )

        self._client = genai.Client(
            vertexai=True,
            project=self._project_id,
            location=self._location,
        )
        logger.info(
            "Vertex AI client initialised (project=%s, location=%s, model=%s)",
            self._project_id, self._location, self._model_name,
        )

    def generate(self, context_bundle: ContextBundle, schema: dict[str, Any]) -> str:
        """Send context to Vertex AI and return raw JSON string.

        If ``corpus_ref`` is set in the bundle (from a prior ``load_corpus_cache``
        call), uses ``cached_content`` to avoid re-transmitting the corpus.
        """
        try:
            from google import genai  # noqa: F401
            from google.genai import types
        except ImportError as exc:
            raise LLMGenerationError(
                "The 'google-genai' package is required for Phase 3. "
                "Install with: pip install google-genai"
            ) from exc

        self._ensure_client()

        # Build system instruction (includes schema rules)
        system_instruction = context_bundle.system_prompt

        # Build user content from remaining bundle sections
        user_parts: list[str] = [context_bundle.ml_block, context_bundle.data_block]
        if context_bundle.human_block:
            user_parts.append("--- OPERATOR INPUT ---\n" + context_bundle.human_block)
        user_parts.append(context_bundle.instruction)
        user_content = "\n\n".join(user_parts)

        # If no cache, inject corpus inline
        if not context_bundle.corpus_ref and context_bundle.corpus_text:
            system_instruction += "\n\n--- REFERENCE CORPUS ---\n" + context_bundle.corpus_text

        # Build generation config
        gen_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.0,
            cached_content=context_bundle.corpus_ref,  # None if no cache
        )

        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=user_content,
                config=gen_config,
            )
            raw = response.text
        except Exception as exc:
            raise LLMGenerationError(f"Vertex AI API error: {exc}") from exc

        if not raw or not raw.strip():
            raise LLMGenerationError("Vertex AI returned empty response")

        # Validate JSON
        try:
            json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMGenerationError(
                f"Vertex AI returned invalid JSON: {exc}"
            ) from exc

        return raw

    def is_available(self) -> bool:
        """Check if Vertex AI endpoint is accessible and the model is listed."""
        try:
            from google import genai  # noqa: F401
        except ImportError:
            logger.warning("google-genai package not installed")
            return False

        if not self._project_id:
            logger.warning("GCP project ID not configured")
            return False

        try:
            self._ensure_client()
            # List models to verify connectivity
            models = self._client.models.list()
            model_names = [m.name for m in models]
            available = any(self._model_name in name for name in model_names)
            if not available:
                logger.warning(
                    "Model %s not found in Vertex AI. Available (first 5): %s",
                    self._model_name, model_names[:5],
                )
            return available
        except Exception as exc:
            logger.warning("Vertex AI health check failed: %s", exc)
            return False

    def load_corpus_cache(
        self,
        corpus_docs: list[CorpusDocument],
        system_prompt: str,
        ttl: int | None = None,
    ) -> str | None:
        """Create or reuse a Vertex AI context cache for the RAG corpus.

        Parameters
        ----------
        corpus_docs:
            Loaded corpus documents.
        system_prompt:
            System prompt to cache alongside the corpus.
        ttl:
            Cache TTL in seconds. Defaults to config value.

        Returns
        -------
        str | None
            Cache resource name, or None if corpus is too small for caching.
        """
        try:
            from google import genai  # noqa: F401
            from google.genai import types
        except ImportError as exc:
            raise CacheCreationError(
                "The 'google-genai' package is required. "
                "Install with: pip install google-genai"
            ) from exc

        self._ensure_client()
        ttl = ttl or self._ttl_seconds

        # Check minimum token count
        token_estimate = estimate_corpus_tokens(corpus_docs)
        if token_estimate < _MIN_CACHE_TOKENS:
            logger.warning(
                "Corpus too small for caching (%d tokens < %d minimum). "
                "Will inject inline instead.",
                token_estimate, _MIN_CACHE_TOKENS,
            )
            self._cache_name = None
            return None

        # Check for existing cache
        cache_display_name = f"wildfire-rag-corpus-{self._get_corpus_version(corpus_docs)}"
        try:
            existing_caches = self._client.caches.list(
                config=types.ListCachedContentsConfig(
                    filter=f'displayName="{cache_display_name}"',
                ),
            )
            for cache in existing_caches:
                # Check if the cache is still valid (not expired)
                if hasattr(cache, "name") and cache.name:
                    logger.info("Cache hit: %s", cache.name)
                    self._cache_name = cache.name
                    return cache.name
        except Exception as exc:
            logger.warning("Failed to check existing caches: %s", exc)

        # Create new cache
        try:
            # Build corpus content for caching
            corpus_parts: list[types.Part] = []
            for doc in corpus_docs:
                if doc.mime_type == "text/plain":
                    text = doc.content_bytes.decode("utf-8", errors="replace")
                    corpus_parts.append(types.Part.from_text(
                        f"--- {doc.filename} ---\n{text}"
                    ))
                else:
                    # Binary files (PDFs) as inline data
                    corpus_parts.append(types.Part.from_bytes(
                        data=doc.content_bytes,
                        mime_type=doc.mime_type,
                    ))

            cache = self._client.caches.create(
                config=types.CreateCachedContentConfig(
                    model=self._model_name,
                    display_name=cache_display_name,
                    system_instruction=system_prompt,
                    contents=[types.Content(parts=corpus_parts, role="user")],
                    ttl=f"{ttl}s",
                ),
            )
            self._cache_name = cache.name
            logger.info("Cache created: %s (TTL=%ds)", cache.name, ttl)
            return cache.name

        except Exception as exc:
            raise CacheCreationError(
                f"Failed to create Vertex AI context cache: {exc}"
            ) from exc

    @staticmethod
    def _get_corpus_version(corpus_docs: list[CorpusDocument]) -> str:
        """Derive a version string from corpus document count + total size."""
        total_size = sum(len(d.content_bytes) for d in corpus_docs)
        return f"v{len(corpus_docs)}-{total_size}"