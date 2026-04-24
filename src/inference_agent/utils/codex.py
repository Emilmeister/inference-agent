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
            # Log detailed diagnostics
            logger.error(
                "codex exec failed (rc=%d) for %s",
                proc.returncode,
                output_model.__name__,
            )
            logger.error("codex stderr (first 500 chars): %s", stderr_text[:500])
            logger.error("codex stdout (first 500 chars): %s", stdout_text[:500])

            # Check for common failure patterns
            if "api key" in stderr_text.lower() or "authentication" in stderr_text.lower():
                logger.error("Possible API key issue — check OPENAI_API_KEY or codex auth")
            if "rate limit" in stderr_text.lower():
                logger.error("Rate limited — consider adding retry/backoff")
            if "sandbox" in stderr_text.lower():
                logger.error("Sandbox restriction — check codex permissions")

            # Raise with concise error (first line of stderr)
            first_line = stderr_text.strip().split("\n")[0][:300] if stderr_text.strip() else "no stderr"
            raise RuntimeError(
                f"codex exec failed (rc={proc.returncode}): {first_line}"
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
