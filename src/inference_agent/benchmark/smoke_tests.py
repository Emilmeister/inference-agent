"""Smoke tests for correctness gate — basic chat, tool-calling, structured output."""

from __future__ import annotations

import json
import logging

import aiohttp

from inference_agent.models import SmokeTestResult

logger = logging.getLogger(__name__)

# Max chars of any single field (content / reasoning / tool_calls) echoed into
# the failure log. Enough to see what the model actually emitted without dumping
# multi-KB reasoning traces into the agent log.
_DEBUG_FIELD_CHARS = 2000


async def _chat_completion(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    payload_extra: dict,
) -> dict:
    """Send a chat completion request and return the parsed response."""
    payload = {"model": model, "max_tokens": 8128, **payload_extra}
    async with session.post(
        url, json=payload, timeout=aiohttp.ClientTimeout(total=60)
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {body[:300]}")
        return await resp.json()


def _response_debug(data: dict | None) -> str:
    """Compact, log-friendly view of what the model actually returned.

    Surfaces the fields that explain a smoke-test failure: finish_reason (e.g.
    'length' = truncated, 'tool_calls'), visible content, reasoning_content
    (reasoning models often dump everything here and leave content empty), and
    any tool_calls. Truncated per field so reasoning traces don't flood the log.
    """
    if not data:
        return "<no response received>"
    try:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {}) or {}
        parts = [f"finish_reason={choice.get('finish_reason')!r}"]

        content = message.get("content")
        parts.append(f"content={(content or '')[:_DEBUG_FIELD_CHARS]!r}")

        reasoning = message.get("reasoning_content")
        if reasoning:
            parts.append(
                f"reasoning_content={reasoning[:_DEBUG_FIELD_CHARS]!r}"
            )

        tool_calls = message.get("tool_calls")
        if tool_calls:
            dumped = json.dumps(tool_calls, ensure_ascii=False)
            parts.append(f"tool_calls={dumped[:_DEBUG_FIELD_CHARS]}")

        return " | ".join(parts)
    except Exception:  # never let debug formatting mask the real failure
        return json.dumps(data, ensure_ascii=False)[:_DEBUG_FIELD_CHARS]


async def test_basic_chat(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
) -> tuple[bool, str, dict | None]:
    """Test that the engine responds to a basic chat request."""
    data = None
    try:
        data = await _chat_completion(session, url, model, {
            "messages": [
                {"role": "user", "content": "Say hello in one word."}
            ],
        })
        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        if not content.strip():
            return False, "Empty content in response", data
        return True, f"PASS: got response ({len(content)} chars)", data
    except Exception as e:
        return False, f"ERROR: {e}", data


async def test_tool_calling(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
) -> tuple[bool, str, dict | None]:
    """Test that the model can produce a tool call (tool_choice=auto)."""
    data = None
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
            return False, "No tool_calls in response", data

        tc = tool_calls[0]
        func = tc.get("function", {})
        if func.get("name") != "get_weather":
            return False, f"Expected get_weather, got {func.get('name')}", data

        args = json.loads(func.get("arguments", "{}"))
        if "city" not in args:
            return False, f"Missing 'city' in arguments: {args}", data

        return True, f"PASS: tool_calls with get_weather(city={args['city']})", data

    except Exception as e:
        return False, f"ERROR: {e}", data


async def test_tool_required(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
) -> tuple[bool, str, dict | None]:
    """Test forced tool_choice — model MUST call the specified function."""
    data = None
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
            return False, "No tool_calls with forced tool_choice", data

        tc = tool_calls[0]
        func = tc.get("function", {})
        if func.get("name") != "get_weather":
            return False, f"Expected get_weather, got {func.get('name')}", data

        args = json.loads(func.get("arguments", "{}"))
        return True, f"PASS: forced tool_choice get_weather(city={args.get('city', '?')})", data

    except Exception as e:
        return False, f"ERROR: {e}", data


async def test_json_schema(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    *,
    enable_thinking: bool = True,
) -> tuple[bool, str, dict | None]:
    """Test structured output with a JSON schema.

    Reasoning models frequently break guided decoding: the engine constrains the
    visible channel to the schema while the model wants to emit a reasoning
    preamble, so it either returns empty ``content`` (all text went to
    ``reasoning_content``) or violates the grammar. Passing
    ``chat_template_kwargs={"enable_thinking": false}`` (a vLLM/SGLang chat-
    template knob) suppresses the thinking channel so structured output has a
    clean path. We run BOTH variants and the gate passes if either works — see
    run_smoke_tests.
    """
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

    payload = {
        "messages": [
            {
                "role": "user",
                "content": "List 3 programming languages with their year of creation. Do not think, just respond directly.",
            }
        ],
        "max_tokens": 8128,
        "response_format": schema,
    }
    if not enable_thinking:
        # vLLM/SGLang chat-template knob — suppress the reasoning channel so
        # guided decoding has a clean path to the schema-constrained output.
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    data = None
    try:
        data = await _chat_completion(session, url, model, payload)

        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        if not content:
            return False, "No content in response (reasoning model may not support JSON schema)", data
        parsed = json.loads(content)

        # Validate structure
        if "languages" not in parsed:
            return False, "Missing 'languages' key", data
        langs = parsed["languages"]
        if not isinstance(langs, list) or len(langs) == 0:
            return False, f"Expected non-empty list, got {type(langs).__name__}", data

        for lang in langs:
            if "name" not in lang or "year" not in lang:
                return False, f"Missing name/year in: {lang}", data
            if not isinstance(lang["year"], int):
                return False, f"Year is not int: {lang['year']}", data

        names = [lang["name"] for lang in langs]
        return True, f"PASS: {len(langs)} languages: {names}", data

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}", data
    except Exception as e:
        return False, f"ERROR: {e}", data


def _log_outcome(name: str, ok: bool, detail: str, raw: dict | None) -> None:
    """Log a single smoke test result; on failure echo the model's raw output."""
    if ok:
        logger.info("  %s", detail)
    else:
        logger.warning(
            "  %s FAILED: %s | model returned: %s",
            name, detail, _response_debug(raw),
        )


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
        result.basic_chat, result.basic_chat_detail, raw = await test_basic_chat(
            session, url, model
        )
        _log_outcome("basic chat", result.basic_chat, result.basic_chat_detail, raw)

        # Tool calling (auto)
        logger.info("Smoke test: tool calling (auto)...")
        result.tool_calling, result.tool_calling_detail, raw = await test_tool_calling(
            session, url, model
        )
        _log_outcome("tool calling", result.tool_calling, result.tool_calling_detail, raw)

        # Tool calling (required/forced)
        logger.info("Smoke test: tool calling (required)...")
        result.tool_required, result.tool_required_detail, raw = await test_tool_required(
            session, url, model
        )
        _log_outcome("tool required", result.tool_required, result.tool_required_detail, raw)

        # JSON schema — two variants. Reasoning models often break guided
        # decoding with thinking ON, so we also try with the reasoning channel
        # suppressed. Gate passes if EITHER variant produces valid output.
        logger.info("Smoke test: JSON schema (thinking on)...")
        ok_think, detail_think, raw_think = await test_json_schema(
            session, url, model, enable_thinking=True
        )
        _log_outcome("json schema [thinking on]", ok_think, detail_think, raw_think)

        logger.info("Smoke test: JSON schema (thinking off)...")
        ok_nothink, detail_nothink, raw_nothink = await test_json_schema(
            session, url, model, enable_thinking=False
        )
        _log_outcome("json schema [thinking off]", ok_nothink, detail_nothink, raw_nothink)

        result.json_schema = ok_think or ok_nothink
        result.json_schema_detail = (
            f"thinking_on={'PASS' if ok_think else 'FAIL'} ({detail_think}); "
            f"thinking_off={'PASS' if ok_nothink else 'FAIL'} ({detail_nothink})"
        )

    passed = sum([
        result.basic_chat, result.tool_calling, result.tool_required,
        result.json_schema,
    ])
    logger.info(
        "Smoke tests: %d/4 passed, gate=%s",
        passed, "PASS" if result.gate_passed else "FAIL",
    )

    return result
