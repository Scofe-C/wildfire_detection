"""Abstract base adapter — all LLM backends must implement this interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.models.obj3_gemini.context_builder import ContextBundle


class LLMGenerationError(Exception):
    """Raised on LLM API failure or empty response."""


class LLMAdapter(ABC):
    """Abstract interface for LLM backends (Ollama, Gemini Dev, Vertex AI)."""

    @abstractmethod
    def generate(self, context_bundle: ContextBundle, schema: dict[str, Any]) -> str:
        """Send context to the LLM and return the raw JSON string (unparsed).

        Parameters
        ----------
        context_bundle:
            Complete context payload assembled by the context builder.
        schema:
            JSON schema dict that the response must conform to.

        Returns
        -------
        str
            Raw JSON string from the LLM.

        Raises
        ------
        LLMGenerationError
            On API failure, timeout, or empty response.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Health check — return True if the LLM backend is reachable.

        Used by ``reporter.load_model()`` to validate setup at init time.
        """
