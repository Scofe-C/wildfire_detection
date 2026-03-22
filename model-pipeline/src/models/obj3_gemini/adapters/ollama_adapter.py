"""Ollama adapter — Phase 1 local LLM backend.

Uses the ``ollama`` Python client to call a locally-running Ollama server.
Schema enforcement is done via the ``format`` parameter in ``ollama.chat()``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.models.obj3_gemini.adapters.base_adapter import LLMAdapter, LLMGenerationError
from src.models.obj3_gemini.context_builder import ContextBundle

logger = logging.getLogger(__name__)


class OllamaAdapter(LLMAdapter):
    """Phase 1 adapter — calls a local Ollama server."""

    def __init__(self, config: dict[str, Any]) -> None:
        ollama_cfg = config.get("ollama", {})
        self._model: str = ollama_cfg.get("model", "qwen2.5:7b")
        self._base_url: str = ollama_cfg.get("base_url", "http://localhost:11434")
        self._temperature: float = float(ollama_cfg.get("temperature", 0.0))
        self._max_retries: int = int(ollama_cfg.get("max_retries", 2))

    def generate(self, context_bundle: ContextBundle, schema: dict[str, Any]) -> str:
        """Send context to Ollama and return raw JSON string.

        Message construction order (matches plan injection order):
          1. system role: system_prompt + inline corpus_text
          2. user role: ml_block + data_block + human_block + instruction

        Retries up to ``max_retries`` on JSON decode failure.
        """
        try:
            import ollama as ollama_lib
        except ImportError as exc:
            raise LLMGenerationError(
                "The 'ollama' package is required for Phase 1. "
                "Install with: pip install ollama"
            ) from exc

        # Build message list
        system_content = context_bundle.system_prompt
        if context_bundle.corpus_text:
            system_content += "\n\n--- REFERENCE CORPUS ---\n" + context_bundle.corpus_text

        user_parts = [context_bundle.ml_block, context_bundle.data_block]
        if context_bundle.human_block:
            user_parts.append("--- OPERATOR INPUT ---\n" + context_bundle.human_block)
        user_parts.append(context_bundle.instruction)
        user_content = "\n\n".join(user_parts)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        client = ollama_lib.Client(host=self._base_url)
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 2):  # +2 → initial + retries
            try:
                response = client.chat(
                    model=self._model,
                    messages=messages,
                    format=schema,
                    options={"temperature": self._temperature},
                )
                raw = response.message.content
                if not raw or not raw.strip():
                    raise LLMGenerationError("Ollama returned empty response")

                # Validate it's parseable JSON before returning
                json.loads(raw)
                return raw

            except json.JSONDecodeError as exc:
                last_error = exc
                logger.warning(
                    "Ollama response is not valid JSON (attempt %d/%d): %s",
                    attempt, self._max_retries + 1, exc,
                )
            except LLMGenerationError:
                raise
            except Exception as exc:
                raise LLMGenerationError(f"Ollama API error: {exc}") from exc

        raise LLMGenerationError(
            f"Ollama failed to produce valid JSON after {self._max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )

    def is_available(self) -> bool:
        """Check if Ollama server is reachable and the configured model is pulled."""
        try:
            import ollama as ollama_lib
            client = ollama_lib.Client(host=self._base_url)
            model_list = client.list()
            available = [m.model for m in model_list.models]
            # Check both exact match and base name (e.g. "qwen2.5:7b" vs "qwen2.5:14b")
            return any(
                self._model in name or name.startswith(self._model.split(":")[0])
                for name in available
            )
        except Exception as exc:
            logger.warning("Ollama health check failed: %s", exc)
            return False
