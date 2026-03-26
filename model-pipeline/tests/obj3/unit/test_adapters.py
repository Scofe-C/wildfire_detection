"""Unit tests for LLM adapters.

All external calls are mocked — no Ollama server or Gemini API key required.
Tests run fully offline in CI.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.models.obj3_gemini.adapters.base_adapter import LLMGenerationError
from src.models.obj3_gemini.context_builder import ContextBundle

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VALID_JSON = json.dumps({
    "incident_id": "INC-001",
    "summary": "Test summary",
    "risk_level": "HIGH",
})

MINIMAL_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
}


def _make_bundle(**kwargs: Any) -> ContextBundle:
    defaults = dict(
        system_prompt="You are a wildfire analyst.",
        corpus_text="Reference corpus text.",
        corpus_ref=None,
        ml_block="ML block content.",
        data_block="Data block content.",
        human_block="",
        instruction="Generate a report.",
        report_type="daily",
        incident_id="INC-001",
    )
    defaults.update(kwargs)
    return ContextBundle(**defaults)


# ===========================================================================
# OllamaAdapter
# ===========================================================================

class TestOllamaAdapterInit:
    def test_default_config(self) -> None:
        from src.models.obj3_gemini.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter({})
        assert adapter._model == "qwen2.5:7b"
        assert adapter._base_url == "http://localhost:11434"
        assert adapter._temperature == 0.0
        assert adapter._max_retries == 2

    def test_custom_config(self) -> None:
        from src.models.obj3_gemini.adapters.ollama_adapter import OllamaAdapter
        cfg = {"ollama": {"model": "llama3:8b", "base_url": "http://myserver:11434",
                          "temperature": 0.1, "max_retries": 5}}
        adapter = OllamaAdapter(cfg)
        assert adapter._model == "llama3:8b"
        assert adapter._max_retries == 5


class TestOllamaAdapterGenerate:
    def _make_response(self, content: str) -> MagicMock:
        msg = MagicMock()
        msg.content = content
        resp = MagicMock()
        resp.message = msg
        return resp

    def test_returns_valid_json(self) -> None:
        from src.models.obj3_gemini.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter({})
        bundle = _make_bundle()

        mock_client = MagicMock()
        mock_client.chat.return_value = self._make_response(VALID_JSON)

        with patch.dict("sys.modules", {"ollama": MagicMock(Client=MagicMock(return_value=mock_client))}):
            result = adapter.generate(bundle, MINIMAL_SCHEMA)

        assert json.loads(result) == json.loads(VALID_JSON)

    def test_includes_corpus_in_system(self) -> None:
        from src.models.obj3_gemini.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter({})
        bundle = _make_bundle(corpus_text="CORPUS_MARKER")

        mock_client = MagicMock()
        mock_client.chat.return_value = self._make_response(VALID_JSON)

        with patch.dict("sys.modules", {"ollama": MagicMock(Client=MagicMock(return_value=mock_client))}):
            adapter.generate(bundle, MINIMAL_SCHEMA)

        call_args = mock_client.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[1] if call_args.args else call_args.kwargs["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert "CORPUS_MARKER" in system_msg["content"]

    def test_includes_human_block_when_present(self) -> None:
        from src.models.obj3_gemini.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter({})
        bundle = _make_bundle(human_block="OPERATOR_NOTE")

        mock_client = MagicMock()
        mock_client.chat.return_value = self._make_response(VALID_JSON)

        with patch.dict("sys.modules", {"ollama": MagicMock(Client=MagicMock(return_value=mock_client))}):
            adapter.generate(bundle, MINIMAL_SCHEMA)

        call_args = mock_client.chat.call_args
        messages = call_args.kwargs.get("messages", call_args.args[1] if call_args.args else [])
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "OPERATOR_NOTE" in user_msg["content"]

    def test_retries_on_invalid_json(self) -> None:
        from src.models.obj3_gemini.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter({"ollama": {"max_retries": 1}})
        bundle = _make_bundle()

        mock_client = MagicMock()
        # First call returns bad JSON, second returns valid
        mock_client.chat.side_effect = [
            self._make_response("not-json"),
            self._make_response(VALID_JSON),
        ]

        with patch.dict("sys.modules", {"ollama": MagicMock(Client=MagicMock(return_value=mock_client))}):
            result = adapter.generate(bundle, MINIMAL_SCHEMA)

        assert json.loads(result) == json.loads(VALID_JSON)
        assert mock_client.chat.call_count == 2

    def test_raises_after_all_retries_exhausted(self) -> None:
        from src.models.obj3_gemini.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter({"ollama": {"max_retries": 1}})
        bundle = _make_bundle()

        mock_client = MagicMock()
        mock_client.chat.return_value = self._make_response("not-json-ever")

        with patch.dict("sys.modules", {"ollama": MagicMock(Client=MagicMock(return_value=mock_client))}), \
             pytest.raises(LLMGenerationError, match="failed to produce valid JSON"):
            adapter.generate(bundle, MINIMAL_SCHEMA)

    def test_raises_on_empty_response(self) -> None:
        from src.models.obj3_gemini.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter({})
        bundle = _make_bundle()

        mock_client = MagicMock()
        mock_client.chat.return_value = self._make_response("")

        with patch.dict("sys.modules", {"ollama": MagicMock(Client=MagicMock(return_value=mock_client))}), \
             pytest.raises(LLMGenerationError, match="empty response"):
            adapter.generate(bundle, MINIMAL_SCHEMA)

    def test_raises_on_import_error(self) -> None:
        from src.models.obj3_gemini.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter({})
        bundle = _make_bundle()

        with patch.dict("sys.modules", {"ollama": None}), \
             pytest.raises(LLMGenerationError, match="'ollama' package is required"):
            adapter.generate(bundle, MINIMAL_SCHEMA)

    def test_raises_on_api_error(self) -> None:
        from src.models.obj3_gemini.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter({})
        bundle = _make_bundle()

        mock_client = MagicMock()
        mock_client.chat.side_effect = ConnectionError("refused")

        with patch.dict("sys.modules", {"ollama": MagicMock(Client=MagicMock(return_value=mock_client))}), \
             pytest.raises(LLMGenerationError, match="Ollama API error"):
            adapter.generate(bundle, MINIMAL_SCHEMA)


class TestOllamaAdapterIsAvailable:
    def test_returns_true_when_model_available(self) -> None:
        from src.models.obj3_gemini.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter({"ollama": {"model": "qwen2.5:7b"}})

        mock_model = MagicMock()
        mock_model.model = "qwen2.5:7b"
        mock_client = MagicMock()
        mock_client.list.return_value = MagicMock(models=[mock_model])

        with patch.dict("sys.modules", {"ollama": MagicMock(Client=MagicMock(return_value=mock_client))}):
            assert adapter.is_available() is True

    def test_returns_false_on_connection_error(self) -> None:
        from src.models.obj3_gemini.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter({})

        mock_client = MagicMock()
        mock_client.list.side_effect = ConnectionError("refused")

        with patch.dict("sys.modules", {"ollama": MagicMock(Client=MagicMock(return_value=mock_client))}):
            assert adapter.is_available() is False

    def test_returns_false_when_import_fails(self) -> None:
        from src.models.obj3_gemini.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter({})
        with patch.dict("sys.modules", {"ollama": None}):
            assert adapter.is_available() is False


# ===========================================================================
# GeminiDevAdapter
# ===========================================================================

class TestGeminiDevAdapterInit:
    def test_default_config(self) -> None:
        from src.models.obj3_gemini.adapters.gemini_dev_adapter import GeminiDevAdapter
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            adapter = GeminiDevAdapter({})
        assert adapter._model_name == "gemini-2.5-flash"
        assert adapter._client is None  # lazy init

    def test_custom_model(self) -> None:
        from src.models.obj3_gemini.adapters.gemini_dev_adapter import GeminiDevAdapter
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            adapter = GeminiDevAdapter({"gemini_dev": {"model": "gemini-2.0-flash"}})
        assert adapter._model_name == "gemini-2.0-flash"


class TestGeminiDevAdapterGenerate:
    def _make_adapter(self) -> Any:
        from src.models.obj3_gemini.adapters.gemini_dev_adapter import GeminiDevAdapter
        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
            return GeminiDevAdapter({})

    def _mock_genai(self, response_text: str) -> MagicMock:
        mock_response = MagicMock()
        mock_response.text = response_text

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_types = MagicMock()
        mock_types.GenerateContentConfig = MagicMock(return_value={})

        return mock_genai, mock_types, mock_client

    def test_returns_valid_json(self) -> None:
        adapter = self._make_adapter()
        bundle = _make_bundle()
        mock_genai, mock_types, mock_client = self._mock_genai(VALID_JSON)

        with patch.dict("sys.modules", {
            "google": MagicMock(genai=mock_genai),
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }):
            adapter._client = mock_client
            result = adapter.generate(bundle, MINIMAL_SCHEMA)

        assert json.loads(result) == json.loads(VALID_JSON)

    def test_raises_on_empty_response(self) -> None:
        adapter = self._make_adapter()
        bundle = _make_bundle()
        mock_genai, mock_types, mock_client = self._mock_genai("")

        with patch.dict("sys.modules", {
            "google": MagicMock(genai=mock_genai),
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }), pytest.raises(LLMGenerationError, match="empty response"):
            adapter._client = mock_client
            adapter.generate(bundle, MINIMAL_SCHEMA)

    def test_raises_on_invalid_json(self) -> None:
        adapter = self._make_adapter()
        bundle = _make_bundle()
        mock_genai, mock_types, mock_client = self._mock_genai("not valid json {{")

        with patch.dict("sys.modules", {
            "google": MagicMock(genai=mock_genai),
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }), pytest.raises(LLMGenerationError, match="invalid JSON"):
            adapter._client = mock_client
            adapter.generate(bundle, MINIMAL_SCHEMA)

    def test_raises_when_no_api_key(self) -> None:
        from src.models.obj3_gemini.adapters.gemini_dev_adapter import GeminiDevAdapter
        with patch.dict("os.environ", {}, clear=True):
            adapter = GeminiDevAdapter({})

        bundle = _make_bundle()
        mock_genai = MagicMock()

        with patch.dict("sys.modules", {
            "google": MagicMock(genai=mock_genai),
            "google.genai": mock_genai,
            "google.genai.types": MagicMock(),
        }), pytest.raises(LLMGenerationError, match="GEMINI_API_KEY"):
            adapter.generate(bundle, MINIMAL_SCHEMA)

    def test_raises_on_api_error(self) -> None:
        adapter = self._make_adapter()
        bundle = _make_bundle()

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("quota exceeded")
        mock_genai = MagicMock()
        mock_types = MagicMock()
        mock_types.GenerateContentConfig = MagicMock(return_value={})

        with patch.dict("sys.modules", {
            "google": MagicMock(genai=mock_genai),
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }), pytest.raises(LLMGenerationError, match="Gemini Dev API error"):
            adapter._client = mock_client
            adapter.generate(bundle, MINIMAL_SCHEMA)


class TestGeminiDevAdapterIsAvailable:
    def test_returns_false_when_no_api_key(self) -> None:
        from src.models.obj3_gemini.adapters.gemini_dev_adapter import GeminiDevAdapter
        with patch.dict("os.environ", {}, clear=True):
            adapter = GeminiDevAdapter({})
        mock_genai = MagicMock()
        with patch.dict("sys.modules", {
            "google": MagicMock(genai=mock_genai),
            "google.genai": mock_genai,
        }):
            assert adapter.is_available() is False

    def test_returns_false_when_import_fails(self) -> None:
        from src.models.obj3_gemini.adapters.gemini_dev_adapter import GeminiDevAdapter
        with patch.dict("os.environ", {"GEMINI_API_KEY": "key"}):
            adapter = GeminiDevAdapter({})
        with patch.dict("sys.modules", {"google": None, "google.genai": None}):
            assert adapter.is_available() is False

    def test_returns_false_on_api_error(self) -> None:
        from src.models.obj3_gemini.adapters.gemini_dev_adapter import GeminiDevAdapter
        with patch.dict("os.environ", {"GEMINI_API_KEY": "key"}):
            adapter = GeminiDevAdapter({})

        mock_genai = MagicMock()
        mock_genai.Client.side_effect = Exception("network error")
        with patch.dict("sys.modules", {
            "google": MagicMock(genai=mock_genai),
            "google.genai": mock_genai,
        }):
            assert adapter.is_available() is False
