"""Tests for the OpenAI-compatible LLM utility — schema prep and JSON extraction."""

import json

import pytest

from inference_agent.utils.llm import _extract_json, _strictify_schema


class TestExtractJson:
    def test_plain_json(self):
        assert _extract_json('{"key": "value"}') == {"key": "value"}

    def test_json_with_whitespace(self):
        assert _extract_json('  \n  {"key": "value"}  \n  ') == {"key": "value"}

    def test_markdown_json_fence(self):
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        assert _extract_json(text) == {"key": "value"}

    def test_markdown_plain_fence(self):
        text = 'Here is the result:\n```\n{"key": 42}\n```'
        assert _extract_json(text) == {"key": 42}

    def test_json_embedded_in_text(self):
        text = 'The answer is {"engine": "vllm", "tp": 2} as described.'
        result = _extract_json(text)
        assert result["engine"] == "vllm"

    def test_nested_json(self):
        text = '{"outer": {"inner": [1, 2, 3]}}'
        assert _extract_json(text)["outer"]["inner"] == [1, 2, 3]

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not extract JSON"):
            _extract_json("this is not json at all")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _extract_json("")


class TestStrictifySchema:
    def test_adds_additional_properties_false(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
        }
        result = _strictify_schema(schema)
        assert result["additionalProperties"] is False

    def test_marks_all_properties_required(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
            },
            "required": ["a"],
        }
        result = _strictify_schema(schema)
        assert set(result["required"]) == {"a", "b"}

    def test_recurses_into_nested_objects(self):
        schema = {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {"inner": {"type": "string"}},
                }
            },
        }
        result = _strictify_schema(schema)
        assert result["additionalProperties"] is False
        assert result["properties"]["outer"]["additionalProperties"] is False
        assert result["properties"]["outer"]["required"] == ["inner"]

    def test_planner_schema_strictifies(self):
        from inference_agent.models import PlannerOutput

        schema = json.loads(json.dumps(PlannerOutput.model_json_schema()))
        strict = _strictify_schema(schema)
        assert strict["additionalProperties"] is False
        assert "engine" in strict["required"]
        assert "rationale" in strict["required"]

    def test_analyzer_schema_strictifies(self):
        from inference_agent.models import AnalyzerOutput

        schema = json.loads(json.dumps(AnalyzerOutput.model_json_schema()))
        strict = _strictify_schema(schema)
        assert strict["additionalProperties"] is False
        assert "decision" in strict["required"]


class TestPydanticSchemas:
    def test_planner_output_schema(self):
        from inference_agent.models import PlannerOutput

        schema = PlannerOutput.model_json_schema()
        assert schema["type"] == "object"
        assert "engine" in schema["properties"]
        assert "rationale" in schema["properties"]

    def test_analyzer_output_schema(self):
        from inference_agent.models import AnalyzerOutput

        schema = AnalyzerOutput.model_json_schema()
        assert schema["type"] == "object"
        assert "commentary" in schema["properties"]
        assert "decision" in schema["properties"]
