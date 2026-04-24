"""Utility for calling codex exec with structured output."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


async def codex_structured_output(
    prompt: str,
    output_model: type[T],
) -> T:
    """Run `codex exec` with --output-schema and parse the result.

    Args:
        prompt: The full prompt text (system + user combined).
        output_model: Pydantic model class for structured output.

    Returns:
        Parsed instance of output_model.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        schema_path = Path(tmpdir) / "schema.json"
        result_path = Path(tmpdir) / "result.json"

        # Write JSON Schema from Pydantic model
        schema = output_model.model_json_schema()
        schema_path.write_text(json.dumps(schema, indent=2))

        # Build command — prompt via stdin for large prompts
        cmd = [
            "codex", "exec",
            "-",  # read prompt from stdin
            "--output-schema", str(schema_path),
            "-o", str(result_path),
            "--skip-git-repo-check",
            "--full-auto",
        ]

        prompt_bytes = prompt.encode("utf-8")
        logger.info(
            "Running codex exec for %s (prompt=%d bytes, schema=%s)...",
            output_model.__name__,
            len(prompt_bytes),
            schema_path,
        )
        logger.debug("Codex command: %s", " ".join(cmd))

        # Log environment hints
        codex_path = os.popen("which codex 2>/dev/null").read().strip()
        logger.debug("Codex binary: %s", codex_path or "not found in PATH")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(prompt_bytes),
                timeout=600,
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError("codex exec timed out after 600s")

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            # Split stderr into banner and actual error content.
            # Codex prints a banner (workdir, model, sandbox, session id, etc.)
            # followed by "--------" then the prompt echo, then the real error.
            # We want to find the error AFTER the banner.
            banner_end = _find_after_banner(stderr_text)
            error_body = stderr_text[banner_end:].strip()

            logger.error(
                "codex exec failed (rc=%d) for %s",
                proc.returncode,
                output_model.__name__,
            )
            if error_body:
                logger.error("codex error: %s", error_body[:2000])
            else:
                logger.error("codex stderr (no error body found): %s", stderr_text[-1000:])
            if stdout_text.strip():
                logger.error("codex stdout: %s", stdout_text[:500])

            # Check for common failure patterns in error body (not banner)
            check_text = (error_body or stderr_text).lower()
            if "api key" in check_text or "authentication" in check_text or "unauthorized" in check_text:
                logger.error("HINT: API key issue — check OPENAI_API_KEY or `codex auth`")
            if "rate limit" in check_text or "429" in check_text:
                logger.error("HINT: Rate limited — consider retry/backoff")
            if "could not write" in check_text or "permission denied" in check_text:
                logger.error("HINT: File write permission issue — check sandbox/tmpdir")
            if "output-schema" in check_text or "schema" in check_text:
                logger.error("HINT: Schema-related error — check output schema compatibility")

            # Raise with the actual error, not the banner
            error_summary = error_body[:300] if error_body else stderr_text.strip().split("\n")[-1][:300]
            raise RuntimeError(
                f"codex exec failed (rc={proc.returncode}): {error_summary}"
            )

        # Read result
        if not result_path.exists():
            # Fallback: try to parse stdout
            logger.warning(
                "codex did not write result file, trying stdout (%d bytes)",
                len(stdout_text),
            )
            logger.debug("codex stdout: %s", stdout_text[:1000])
            result_data = _extract_json(stdout_text)
        else:
            result_text = result_path.read_text()
            logger.debug("codex result file (%d bytes): %s", len(result_text), result_text[:500])
            result_data = _extract_json(result_text)

        logger.info("Codex result parsed successfully for %s", output_model.__name__)
        return output_model.model_validate(result_data)


def _find_after_banner(stderr: str) -> int:
    """Find the position in stderr after the codex banner.

    Codex stderr starts with a banner like:
        OpenAI Codex v0.124.0 (research preview)
        --------
        workdir: ...
        model: ...
        ...
        session id: ...
        --------
        user
        <prompt echo>

    We want to skip past all of this to find the actual error message.
    The banner has two "--------" separators. After the second one comes
    the role ("user"/"assistant") and the prompt echo, then the error.
    """
    # Find the second "--------" separator
    first = stderr.find("--------")
    if first == -1:
        return 0
    second = stderr.find("--------", first + 8)
    if second == -1:
        return 0

    # After the second separator, skip the prompt echo.
    # Look for the next line that looks like an error, not prompt content.
    # The prompt is usually very long; find end of stderr content after it.
    rest = stderr[second + 8:]

    # The prompt echo is the largest chunk. If there's content after it,
    # it's typically on the last few lines. Return position of last 20%
    # of remaining stderr as a heuristic, or look for error patterns.
    for marker in ("error:", "Error:", "ERROR", "panic:", "Traceback", "failed", "denied"):
        idx = rest.find(marker)
        if idx != -1:
            # Go back to start of line
            line_start = rest.rfind("\n", 0, idx)
            return second + 8 + (line_start + 1 if line_start != -1 else idx)

    # No error marker found — return everything after the banner
    return second + 8


def _extract_json(text: str) -> dict:
    """Extract JSON from text that may contain markdown fences or other wrapping."""
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

    raise ValueError(f"Could not extract JSON from codex output: {text[:500]}")
