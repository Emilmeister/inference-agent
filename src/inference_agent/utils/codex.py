"""Utility for calling Claude Code CLI with structured output."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


async def claude_structured_output(
    prompt: str,
    output_model: type[T],
) -> T:
    """Run `claude` CLI with --json-schema and parse the result.

    Uses Claude Code CLI in bare mode (no hooks, no MCP, no memory)
    with structured output via --json-schema flag.

    Args:
        prompt: The full prompt text (system + user combined).
        output_model: Pydantic model class for structured output.

    Returns:
        Parsed instance of output_model.
    """
    schema = output_model.model_json_schema()
    schema_json = json.dumps(schema)

    cmd = [
        "claude", "--bare",
        "-p", prompt,
        "--output-format", "json",
        "--json-schema", schema_json,
    ]

    prompt_bytes = prompt.encode("utf-8")
    logger.info(
        "Running claude for %s (prompt=%d bytes)...",
        output_model.__name__,
        len(prompt_bytes),
    )
    logger.debug("Claude command: claude --bare -p <prompt> --json-schema <schema> ...")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=600,
        )
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError("claude timed out after 600s")

    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")

    if proc.returncode != 0:
        logger.error(
            "claude failed (rc=%d) for %s",
            proc.returncode,
            output_model.__name__,
        )
        if stderr_text.strip():
            logger.error("claude stderr: %s", stderr_text[:2000])
        if stdout_text.strip():
            logger.error("claude stdout: %s", stdout_text[:500])

        # Check for common failure patterns
        check_text = (stderr_text + stdout_text).lower()
        if "api key" in check_text or "authentication" in check_text or "unauthorized" in check_text:
            logger.error("HINT: API key issue — check ANTHROPIC_API_KEY")
        if "rate limit" in check_text or "429" in check_text:
            logger.error("HINT: Rate limited — consider retry/backoff")

        error_summary = stderr_text.strip()[:300] or stdout_text.strip()[:300]
        raise RuntimeError(
            f"claude failed (rc={proc.returncode}): {error_summary}"
        )

    # Parse JSON response from claude
    try:
        response = json.loads(stdout_text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse claude JSON output: %s", e)
        logger.error("claude stdout: %s", stdout_text[:1000])
        raise RuntimeError(f"claude returned invalid JSON: {e}")

    # Check response type
    resp_type = response.get("type")
    if resp_type != "success":
        error_msg = response.get("error", response.get("result", str(response)))
        raise RuntimeError(f"claude returned type={resp_type}: {error_msg}")

    # Extract structured output
    data = response.get("structured_output")
    if data is None:
        # Fallback: try to extract JSON from the text result
        result_text = response.get("result", "")
        if result_text:
            logger.warning(
                "claude returned no structured_output, trying to parse result text (%d bytes)",
                len(result_text),
            )
            data = _extract_json(result_text)
        else:
            raise RuntimeError("claude returned no structured_output and no result text")

    # Log cost info
    cost = response.get("cost", {})
    logger.info(
        "Claude result parsed for %s (cost=$%.4f, tokens=%d)",
        output_model.__name__,
        cost.get("cost_usd", 0),
        cost.get("total_tokens", 0),
    )

    return output_model.model_validate(data)


# Backward-compatible alias — nodes import this name
codex_structured_output = claude_structured_output


def _extract_json(text: str) -> dict:
    """Extract JSON from text that may contain markdown fences or other wrapping.

    Fallback parser used when structured_output is not available.
    """
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences
    for marker in ("```json", "```"):
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.index("```", start)
            return json.loads(text[start:end].strip())

    # Try finding first { ... } block
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        return json.loads(text[brace_start : brace_end + 1])

    raise ValueError(f"Could not extract JSON from output: {text[:500]}")
