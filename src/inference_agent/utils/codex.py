"""Utility for calling Claude Code CLI with structured output.

We run `claude -p` (NOT `--bare`) so that subscription/OAuth auth is used
instead of API-key billing. To make the call behave like a stateless API
endpoint, we:
  - drop ANTHROPIC_API_KEY (forces subscription auth path)
  - cd into an empty workdir (avoids loading project CLAUDE.md / .claude/*)
  - disable tools, slash-commands, session persistence
  - replace the default Claude Code system prompt with a minimal one
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Empty workdir for claude subprocess — keeps it from picking up
# project-level CLAUDE.md, .claude/settings.json, .mcp.json, etc.
_CLAUDE_WORKDIR = Path.home() / ".cache" / "inference-agent-claude-runner"

# Minimal system prompt that replaces the default Claude Code prompt.
# Kept short and intentionally non-agentic.
_SYSTEM_PROMPT = (
    "You are a deterministic structured-output assistant. "
    "Follow the user's task and return valid JSON matching the provided schema. "
    "Do not mention the schema. Do not use tools."
)


def _ensure_workdir() -> Path:
    """Ensure the empty workdir exists for claude subprocess calls."""
    _CLAUDE_WORKDIR.mkdir(parents=True, exist_ok=True)
    return _CLAUDE_WORKDIR


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
        "claude", "-p", prompt,
        "--output-format", "json",
        "--json-schema", schema_json,
        "--tools", "",                  # no Bash/Read/Edit/etc.
        "--disable-slash-commands",     # no /skills, /plugins
        "--no-session-persistence",     # stateless
        "--system-prompt", _SYSTEM_PROMPT,
        "--max-turns", "2",             # allow 1 retry for schema mismatch
    ]

    prompt_bytes = prompt.encode("utf-8")
    logger.info(
        "Running claude for %s (prompt=%d bytes)...",
        output_model.__name__,
        len(prompt_bytes),
    )
    logger.debug("Claude command: claude -p <prompt> --json-schema <schema> --tools '' ...")

    # Drop ANTHROPIC_API_KEY so claude uses subscription (OAuth) auth instead
    # of API-key billing. Run from an empty workdir so claude doesn't load
    # project CLAUDE.md / .claude/settings / .mcp.json.
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    workdir = _ensure_workdir()

    # Use DEVNULL for stdin — otherwise claude waits 3s for piped input.
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(workdir),
        env=env,
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

    # Try to parse JSON response even if rc != 0 — claude returns
    # error info inside the result payload (e.g. "Not logged in").
    response: dict | None = None
    if stdout_text.strip():
        try:
            response = json.loads(stdout_text)
        except json.JSONDecodeError:
            response = None

    if proc.returncode != 0 or (response and response.get("is_error")):
        logger.error(
            "claude failed (rc=%d) for %s",
            proc.returncode,
            output_model.__name__,
        )
        if stderr_text.strip():
            logger.error("claude stderr: %s", stderr_text[:2000])

        # Extract error message from response payload if available
        if response:
            payload_error = response.get("result") or response.get("error", "")
            logger.error("claude error: %s", str(payload_error)[:500])
        elif stdout_text.strip():
            logger.error("claude stdout: %s", stdout_text[:500])

        # Check for common failure patterns
        check_text = (stderr_text + stdout_text).lower()
        if "not logged in" in check_text or "/login" in check_text:
            logger.error(
                "HINT: claude is not authenticated. Run `claude` interactively "
                "(once) and execute `/login` to OAuth into your subscription. "
                "The token will be saved in the keychain and reused by subprocess calls."
            )
        elif "api key" in check_text or "authentication" in check_text or "unauthorized" in check_text:
            logger.error("HINT: authentication issue — re-run `claude /login`")
        if "rate limit" in check_text or "429" in check_text:
            logger.error("HINT: Rate limited — consider retry/backoff")

        if response:
            error_summary = str(response.get("result") or response.get("error", response))[:300]
        else:
            error_summary = stderr_text.strip()[:300] or stdout_text.strip()[:300]
        raise RuntimeError(
            f"claude failed (rc={proc.returncode}): {error_summary}"
        )

    if response is None:
        raise RuntimeError(f"claude returned invalid JSON: {stdout_text[:500]}")

    # Successful response — extract structured output
    # Schema: {"type": "result", "subtype": "success", "is_error": false,
    #          "result": "...", "structured_output": {...}|null,
    #          "total_cost_usd": 0.01, "usage": {...}}
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

    # Log cost info (claude schema: total_cost_usd, usage.{input,output}_tokens)
    cost_usd = response.get("total_cost_usd", 0.0)
    usage = response.get("usage", {})
    total_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    logger.info(
        "Claude result parsed for %s (cost=$%.4f, tokens=%d)",
        output_model.__name__,
        cost_usd,
        total_tokens,
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
