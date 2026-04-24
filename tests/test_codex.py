"""Tests for codex utility — JSON extraction from various output formats."""

import pytest

from inference_agent.utils.codex import _extract_json


class TestExtractJson:
    def test_plain_json(self):
        result = _extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_whitespace(self):
        result = _extract_json('  \n  {"key": "value"}  \n  ')
        assert result == {"key": "value"}

    def test_markdown_json_fence(self):
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_markdown_plain_fence(self):
        text = 'Here is the result:\n```\n{"key": 42}\n```'
        result = _extract_json(text)
        assert result == {"key": 42}

    def test_json_embedded_in_text(self):
        text = 'The answer is {"engine": "vllm", "tp": 2} as described.'
        result = _extract_json(text)
        assert result["engine"] == "vllm"

    def test_nested_json(self):
        text = '{"outer": {"inner": [1, 2, 3]}}'
        result = _extract_json(text)
        assert result["outer"]["inner"] == [1, 2, 3]

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not extract JSON"):
            _extract_json("this is not json at all")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _extract_json("")

    def test_complex_codex_output(self):
        """Simulate codex wrapping JSON in explanation text."""
        text = """I'll configure a baseline vLLM experiment.

```json
{
  "engine": "vllm",
  "tensor_parallel_size": 1,
  "max_model_len": 32768,
  "rationale": "Baseline config"
}
```

This should work well."""
        result = _extract_json(text)
        assert result["engine"] == "vllm"
        assert result["max_model_len"] == 32768
