"""Smoke tests for tool-calling and structured output."""

from __future__ import annotations

import json
import logging

import aiohttp

from inference_agent.models import SmokeTestResult

logger = logging.getLogger(__name__)


async def _chat_completion(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    payload_extra: dict,
) -> dict:
    """Send a chat completion request and return the parsed response."""
    payload = {"model": model, "max_tokens": 1024, **payload_extra}
    async with session.post(
        url, json=payload, timeout=aiohttp.ClientTimeout(total=60)
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {body[:300]}")
        return await resp.json()


async def test_tool_calling(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
) -> tuple[bool, str]:
    """Test that the model can produce a tool call."""
    try:
        data = await _chat_completion(session, url, model, {
            "messages": [
                {"role": "user", "content": "What's the weather in Moscow right now?"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "City name",
                                }
                            },
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        })

        message = data["choices"][0]["message"]
        tool_calls = message.get("tool_calls", [])
        if not tool_calls:
            return False, "No tool_calls in response"

        tc = tool_calls[0]
        func = tc.get("function", {})
        if func.get("name") != "get_weather":
            return False, f"Expected get_weather, got {func.get('name')}"

        args = json.loads(func.get("arguments", "{}"))
        if "city" not in args:
            return False, f"Missing 'city' in arguments: {args}"

        return True, f"PASS: tool_calls with get_weather(city={args['city']})"

    except Exception as e:
        return False, f"ERROR: {e}"


async def test_json_mode(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
) -> tuple[bool, str]:
    """Test that the model can produce valid JSON in JSON mode."""
    try:
        data = await _chat_completion(session, url, model, {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "List 3 programming languages with their year of creation. "
                        "Respond in JSON format. Do not think, just respond with JSON directly."
                    ),
                }
            ],
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
        })

        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        if not content:
            return False, "No content in response (reasoning model may not support JSON mode)"
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return False, f"Expected JSON object, got {type(parsed).__name__}"
        return True, f"PASS: valid JSON with keys {list(parsed.keys())[:5]}"

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"ERROR: {e}"


async def test_json_schema(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
) -> tuple[bool, str]:
    """Test structured output with a JSON schema."""
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "languages",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "languages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "year": {"type": "integer"},
                            },
                            "required": ["name", "year"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["languages"],
                "additionalProperties": False,
            },
        },
    }

    try:
        data = await _chat_completion(session, url, model, {
            "messages": [
                {
                    "role": "user",
                    "content": "List 3 programming languages with their year of creation. Do not think, just respond directly.",
                }
            ],
            "max_tokens": 1024,
            "response_format": schema,
        })

        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        if not content:
            return False, "No content in response (reasoning model may not support JSON schema)"
        parsed = json.loads(content)

        # Validate structure
        if "languages" not in parsed:
            return False, "Missing 'languages' key"
        langs = parsed["languages"]
        if not isinstance(langs, list) or len(langs) == 0:
            return False, f"Expected non-empty list, got {type(langs).__name__}"

        for lang in langs:
            if "name" not in lang or "year" not in lang:
                return False, f"Missing name/year in: {lang}"
            if not isinstance(lang["year"], int):
                return False, f"Year is not int: {lang['year']}"

        names = [l["name"] for l in langs]
        return True, f"PASS: {len(langs)} languages: {names}"

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"ERROR: {e}"


async def run_smoke_tests(api_base_url: str, model: str) -> SmokeTestResult:
    """Run all smoke tests against the running engine."""
    url = f"{api_base_url}/chat/completions"
    result = SmokeTestResult()

    async with aiohttp.ClientSession() as session:
        # Tool calling
        logger.info("Smoke test: tool calling...")
        result.tool_calling, result.tool_calling_detail = await test_tool_calling(
            session, url, model
        )
        logger.info("  %s", result.tool_calling_detail)

        # JSON mode
        logger.info("Smoke test: JSON mode...")
        result.json_mode, result.json_mode_detail = await test_json_mode(
            session, url, model
        )
        logger.info("  %s", result.json_mode_detail)

        # JSON schema
        logger.info("Smoke test: JSON schema...")
        result.json_schema, result.json_schema_detail = await test_json_schema(
            session, url, model
        )
        logger.info("  %s", result.json_schema_detail)

    return result
