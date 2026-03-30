"""Ollama adapter — Phase 1 local LLM backend.

Uses the ``ollama`` Python client to call a locally-running Ollama server.
Schema enforcement is done via the ``format`` parameter in ``ollama.chat()``.

Qwen3 note
----------
Qwen3 models emit ``<think>...</think>`` blocks before JSON when thinking mode
is active.  We suppress thinking mode via ``options={"think": False}`` and also
strip any residual thinking tags as a fallback, so the adapter always returns
clean JSON regardless of model version or option support.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.models.obj3_gemini.adapters.base_adapter import LLMAdapter, LLMGenerationError
from src.models.obj3_gemini.context_builder import ContextBundle

logger = logging.getLogger(__name__)

# Regex that matches <think>...</think> blocks (including multiline, non-greedy)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _extract_json(raw: str) -> str:
    """Strip Qwen3 thinking tags and extract the first JSON object or array.

    Steps
    -----
    1. Remove all ``<think>...</think>`` blocks.
    2. Strip leading/trailing whitespace.
    3. If the result starts with ``{`` or ``[`` it is already clean — return it.
    4. Otherwise locate the first ``{`` or ``[`` and return from there to the end.
       This handles models that write a short preamble before the JSON.

    Raises
    ------
    ValueError
        If no JSON start character can be found after cleaning.
    """
    cleaned = _THINK_RE.sub("", raw).strip()

    if cleaned.startswith(("{", "[")):
        return cleaned

    # Find first JSON start character
    for i, ch in enumerate(cleaned):
        if ch in ("{", "["):
            return cleaned[i:]

    raise ValueError(f"No JSON object found in response (first 200 chars): {cleaned[:200]!r}")


class OllamaAdapter(LLMAdapter):
    """Phase 1 adapter — calls a local Ollama server."""

    def __init__(self, config: dict[str, Any]) -> None:
        ollama_cfg = config.get("ollama", {})
        self._model: str = ollama_cfg.get("model", "qwen3:8b")
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

        # Inject uploaded files as text (Ollama has no vision for text-only models)
        if context_bundle.uploaded_files:
            file_parts: list[str] = []
            for pf in context_bundle.uploaded_files:
                header = f"=== Uploaded File: {pf.filename} ==="
                file_parts.append(header + "\n" + pf.text_content)
            user_parts.append("--- UPLOADED FILES ---\n" + "\n\n".join(file_parts))

        user_parts.append(context_bundle.instruction)
        user_content = "\n\n".join(user_parts)

        # Qwen3 thinking-mode suppression — prepend /no_think to the user turn.
        # This is the most reliable cross-version method: honoured directly by
        # Qwen3's chat template regardless of whether the Ollama version supports
        # the think: false option.  Non-Qwen3 models silently ignore it.
        if "qwen3" in self._model.lower():
            user_content = "/no_think\n\n" + user_content

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
                    options={
                        "temperature": self._temperature,
                        # Suppress Qwen3 chain-of-thought thinking blocks.
                        # Older models silently ignore unknown options — safe for all.
                        "think": False,
                    },
                )
                raw = response.message.content
                if not raw or not raw.strip():
                    raise LLMGenerationError("Ollama returned empty response")

                # Strip thinking tags + extract JSON (Qwen3 safety net)
                try:
                    cleaned = _extract_json(raw)
                except ValueError as exc:
                    raise json.JSONDecodeError(str(exc), raw, 0) from exc

                # Validate it's parseable JSON before returning
                json.loads(cleaned)
                return cleaned

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
        """Check if Ollama server is reachable and the exact configured model is pulled."""
        try:
            import ollama as ollama_lib
            client = ollama_lib.Client(host=self._base_url)
            model_list = client.list()
            available = [m.model for m in model_list.models]
            # Exact match: "qwen2.5:7b" must match "qwen2.5:7b", not "qwen2.5:14b".
            # Also accept ":latest" suffix for models pulled without explicit tag.
            return any(
                name == self._model
                or name == f"{self._model}:latest"
                or (":latest" in name and name.replace(":latest", "") == self._model)
                for name in available
            )
        except Exception as exc:
            logger.warning("Ollama health check failed: %s", exc)
            return False
