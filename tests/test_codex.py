"""Tests for claude CLI utility — JSON extraction fallback and backward compat."""

import pytest

from inference_agent.utils.codex import _extract_json, claude_structured_output, codex_structured_output


class TestExtractJson:
    """_extract_json is a fallback for when structured_output is not available."""

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

    def test_complex_output(self):
        """Simulate LLM wrapping JSON in explanation text."""
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


class TestBackwardCompat:
    """Verify that the old import name still works."""

    def test_codex_structured_output_is_alias(self):
        assert codex_structured_output is claude_structured_output


class TestPydanticSchemas:
    """Verify that Pydantic schemas are valid JSON Schema for claude --json-schema."""

    def test_planner_output_schema(self):
        """PlannerOutput schema should be valid JSON Schema."""
        from inference_agent.models import PlannerOutput

        schema = PlannerOutput.model_json_schema()

        # Must be a valid JSON-serializable dict
        json_str = __import__("json").dumps(schema)
        parsed = __import__("json").loads(json_str)
        assert parsed["type"] == "object"
        assert "properties" in parsed
        assert "engine" in parsed["properties"]
        assert "rationale" in parsed["properties"]

    def test_analyzer_output_schema(self):
        """AnalyzerOutput schema should be valid JSON Schema."""
        from inference_agent.models import AnalyzerOutput

        schema = AnalyzerOutput.model_json_schema()

        json_str = __import__("json").dumps(schema)
        parsed = __import__("json").loads(json_str)
        assert parsed["type"] == "object"
        assert "commentary" in parsed["properties"]
        assert "decision" in parsed["properties"]

    def test_planner_schema_has_extra_env(self):
        """extra_env (dict[str, str]) should be in the schema — claude handles it natively."""
        from inference_agent.models import PlannerOutput

        schema = PlannerOutput.model_json_schema()
        # Unlike OpenAI strict mode, claude handles free-form dicts
        assert "extra_env" in schema["properties"]

    def test_planner_schema_has_extra_engine_args(self):
        from inference_agent.models import PlannerOutput

        schema = PlannerOutput.model_json_schema()
        assert "extra_engine_args" in schema["properties"]
