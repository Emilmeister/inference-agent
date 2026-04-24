"""Tests for codex utility — JSON extraction from various output formats."""

import pytest

from inference_agent.utils.codex import _extract_json, _find_after_banner


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

    def test_empty_string_extracts_nothing(self):
        with pytest.raises(ValueError):
            _extract_json("   ")


class TestFindAfterBanner:
    def test_no_banner(self):
        assert _find_after_banner("just an error message") == 0

    def test_single_separator(self):
        stderr = "OpenAI Codex v0.124.0\n--------\nsome content"
        assert _find_after_banner(stderr) == 0

    def test_full_banner_with_error(self):
        stderr = (
            "OpenAI Codex v0.124.0 (research preview)\n"
            "--------\n"
            "workdir: /home/user1/project\n"
            "model: gpt-5.4\n"
            "provider: openai\n"
            "session id: abc-123\n"
            "--------\n"
            "user\n"
            "Some prompt text here...\n"
            "\n"
            "Error: API request failed with status 401\n"
        )
        pos = _find_after_banner(stderr)
        remaining = stderr[pos:]
        assert "Error: API request failed" in remaining

    def test_banner_no_error_marker(self):
        stderr = (
            "OpenAI Codex v0.124.0\n"
            "--------\n"
            "workdir: /tmp\n"
            "--------\n"
            "user\n"
            "prompt content only\n"
        )
        pos = _find_after_banner(stderr)
        # Returns position after second --------
        assert pos > 0
        assert "user" in stderr[pos:]

    def test_error_marker_denied(self):
        stderr = (
            "OpenAI Codex v0.124.0\n"
            "--------\n"
            "info: stuff\n"
            "--------\n"
            "user\n"
            "long prompt...\n"
            "permission denied: /tmp/result.json\n"
        )
        pos = _find_after_banner(stderr)
        remaining = stderr[pos:]
        assert "denied" in remaining

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
