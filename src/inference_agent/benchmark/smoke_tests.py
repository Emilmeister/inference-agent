"""Smoke tests for correctness gate — basic chat, tool-calling, structured output."""

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


async def test_basic_chat(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
) -> tuple[bool, str]:
    """Test that the engine responds to a basic chat request."""
    try:
        data = await _chat_completion(session, url, model, {
            "messages": [
                {"role": "user", "content": "Say hello in one word."}
            ],
        })
        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        if not content.strip():
            return False, "Empty content in response"
        return True, f"PASS: got response ({len(content)} chars)"
    except Exception as e:
        return False, f"ERROR: {e}"


async def test_tool_calling(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
) -> tuple[bool, str]:
    """Test that the model can produce a tool call (tool_choice=auto)."""
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


async def test_tool_required(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
) -> tuple[bool, str]:
    """Test forced tool_choice — model MUST call the specified function."""
    try:
        data = await _chat_completion(session, url, model, {
            "messages": [
                {"role": "user", "content": "Tell me the weather in Tokyo."}
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
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"},
            },
        })

        message = data["choices"][0]["message"]
        tool_calls = message.get("tool_calls", [])
        if not tool_calls:
            return False, "No tool_calls with forced tool_choice"

        tc = tool_calls[0]
        func = tc.get("function", {})
        if func.get("name") != "get_weather":
            return False, f"Expected get_weather, got {func.get('name')}"

        args = json.loads(func.get("arguments", "{}"))
        return True, f"PASS: forced tool_choice get_weather(city={args.get('city', '?')})"

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

        names = [lang["name"] for lang in langs]
        return True, f"PASS: {len(langs)} languages: {names}"

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"ERROR: {e}"


async def run_smoke_tests(api_base_url: str, model: str) -> SmokeTestResult:
    """Run all smoke tests against the running engine.

    Returns SmokeTestResult with gate_passed property indicating
    whether the correctness gate is met (basic_chat + tool_calling + json_schema).
    """
    url = f"{api_base_url}/chat/completions"
    result = SmokeTestResult()

    async with aiohttp.ClientSession() as session:
        # Basic chat (most fundamental — if this fails, engine is broken)
        logger.info("Smoke test: basic chat...")
        result.basic_chat, result.basic_chat_detail = await test_basic_chat(
            session, url, model
        )
        logger.info("  %s", result.basic_chat_detail)

        # Tool calling (auto)
        logger.info("Smoke test: tool calling (auto)...")
        result.tool_calling, result.tool_calling_detail = await test_tool_calling(
            session, url, model
        )
        logger.info("  %s", result.tool_calling_detail)

        # Tool calling (required/forced)
        logger.info("Smoke test: tool calling (required)...")
        result.tool_required, result.tool_required_detail = await test_tool_required(
            session, url, model
        )
        logger.info("  %s", result.tool_required_detail)

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

    passed = sum([
        result.basic_chat, result.tool_calling, result.tool_required,
        result.json_mode, result.json_schema,
    ])
    logger.info(
        "Smoke tests: %d/5 passed, gate=%s",
        passed, "PASS" if result.gate_passed else "FAIL",
    )

    return result
