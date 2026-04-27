"""Structured-output LLM client for any OpenAI-compatible Chat Completions API.

Used by the planner and analyzer nodes to produce Pydantic-validated decisions.
The client is configured via `AgentLLMConfig` (base_url, model, api_key).

Two structured-output modes are supported:
  - "json_schema" (default): strict structured outputs via
    response_format={"type": "json_schema", ...}. Recommended for OpenAI and
    providers that implement the strict schema spec.
  - "json_object": response_format={"type": "json_object"}. The schema is
    embedded in the prompt and the result is validated client-side. Use this
    when the provider does not support strict json_schema.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel

from inference_agent.models_pkg.config import AgentLLMConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_SYSTEM_PROMPT = (
    "You are a deterministic structured-output assistant. "
    "Follow the user's task and return valid JSON matching the provided schema. "
    "Do not mention the schema."
)


def _strictify_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a Pydantic JSON schema to OpenAI strict json_schema form.

    Strict mode requires:
      - every object has additionalProperties: false
      - every property is listed in required (Optional types use null in their
        anyOf, which Pydantic already emits for Optional[X] fields)
    """

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                props = node.get("properties")
                if isinstance(props, dict):
                    node["required"] = list(props.keys())
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(schema)
    return schema


def _extract_json(text: str) -> dict:
    """Extract JSON from text that may contain markdown fences or other wrapping."""
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for marker in ("```json", "```"):
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.index("```", start)
            return json.loads(text[start:end].strip())

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        return json.loads(text[brace_start : brace_end + 1])

    raise ValueError(f"Could not extract JSON from output: {text[:500]}")


def _build_client(llm_config: AgentLLMConfig) -> AsyncOpenAI:
    if not llm_config.api_key:
        raise RuntimeError(
            f"LLM api_key not set. Provide agent_llm.api_key in config or set "
            f"environment variable ${llm_config.api_key_env}."
        )
    return AsyncOpenAI(
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
        timeout=llm_config.timeout_sec,
    )


async def structured_output(
    prompt: str,
    output_model: type[T],
    llm_config: AgentLLMConfig,
) -> T:
    """Call an OpenAI-compatible Chat Completions API and return a validated model.

    Args:
        prompt: User-side prompt (system prompt is added internally).
        output_model: Pydantic model class describing the expected JSON.
        llm_config: Endpoint, model, credentials, and structured-output mode.
    """
    client = _build_client(llm_config)
    schema = output_model.model_json_schema()

    if llm_config.structured_output_mode == "json_schema":
        strict_schema = _strictify_schema(json.loads(json.dumps(schema)))
        response_format: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": output_model.__name__,
                "schema": strict_schema,
                "strict": True,
            },
        }
        user_content = prompt
    else:
        response_format = {"type": "json_object"}
        user_content = (
            f"{prompt}\n\nRespond with a JSON object that matches this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```"
        )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    logger.info(
        "Calling LLM %s @ %s for %s (mode=%s)",
        llm_config.model,
        llm_config.base_url,
        output_model.__name__,
        llm_config.structured_output_mode,
    )

    kwargs: dict[str, Any] = {
        "model": llm_config.model,
        "messages": messages,
        "temperature": llm_config.temperature,
        "response_format": response_format,
    }
    if llm_config.max_tokens is not None:
        kwargs["max_tokens"] = llm_config.max_tokens

    try:
        completion = await client.chat.completions.create(**kwargs)
    except OpenAIError as e:
        raise RuntimeError(f"LLM API call failed: {e}") from e

    if not completion.choices:
        raise RuntimeError("LLM response had no choices")

    content = completion.choices[0].message.content or ""
    if not content.strip():
        raise RuntimeError("LLM returned empty content")

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = _extract_json(content)

    usage = completion.usage
    if usage is not None:
        logger.info(
            "LLM result parsed for %s (prompt_tokens=%d, completion_tokens=%d)",
            output_model.__name__,
            usage.prompt_tokens,
            usage.completion_tokens,
        )

    return output_model.model_validate(data)
