"""Crash diagnostician — a dedicated LLM call that distills a container crash.

When an engine container fails to start / never becomes healthy, the executor
captures the full container logs but the planner/analyzer would otherwise only
see a generic "did not become healthy" message (root cause buried in raw
multiproc stderr). This node runs after the executor and, for container-crash
results only, makes ONE structured LLM call over the full logs to extract the
essence — why it fell and how to fix it — and replaces the planner/analyzer-
facing ``result.error`` with that distilled diagnosis. Raw logs stay on disk
(``storage.logs_dir``) and in ``errors[].details`` for the operator.

Every non-crash result passes through untouched, and any LLM failure leaves the
original error intact (fail-safe — diagnosis is a nice-to-have, never a gate).
"""

from __future__ import annotations

import logging
import os

from inference_agent.models import CrashDiagnosis, ExperimentError, ExperimentStatus
from inference_agent.state import AgentState
from inference_agent.utils.llm import structured_output

logger = logging.getLogger(__name__)

# Stages whose errors carry real container logs (the container ran and died).
# image_pull_failed never produces container logs, so it's excluded.
_CRASH_STAGES = {"startup", "healthcheck"}

# Head+tail budget of the log shipped to the diagnosis LLM. The root cause of a
# boot failure can be early (config parse) OR late (CUDA OOM at capture), so we
# keep both ends rather than a single tail.
_MAX_LOG_CHARS = 40_000


def _crash_error(result) -> ExperimentError | None:
    """First startup/healthcheck error that captured container logs, else None."""
    for err in result.errors:
        if err.stage in _CRASH_STAGES and (err.details or {}).get("logs") is not None:
            return err
    return None


def _load_logs(err: ExperimentError) -> str:
    """Full container log for the crash — prefer the on-disk file, else the excerpt."""
    details = err.details or {}
    path = details.get("log_path")
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                full = f.read()
            if len(full) > _MAX_LOG_CHARS:
                head = full[: _MAX_LOG_CHARS // 4]
                tail = full[-(_MAX_LOG_CHARS * 3 // 4):]
                dropped = len(full) - len(head) - len(tail)
                return f"{head}\n\n...[{dropped} chars truncated]...\n\n{tail}"
            return full
        except OSError as e:
            logger.warning("Could not read crash log %s: %s", path, e)
    # Fallback: the focused excerpt the executor already extracted.
    return str(details.get("logs", "") or "")


def _build_prompt(result, logs: str) -> str:
    hw = result.hardware
    gpu = hw.gpus[0].name if hw.gpus else "unknown"
    vram = f"{hw.gpus[0].vram_total_mb} MB/GPU" if hw.gpus else "unknown VRAM"
    return (
        "An inference engine container failed to start (or died during startup) "
        "and never became healthy. Read its FULL logs and extract the essence: "
        "the single root cause and a concrete fix.\n\n"
        f"Engine: {result.engine.value}\n"
        f"Model: {result.model}\n"
        f"Hardware: {hw.gpu_count}x {gpu} ({vram})\n"
        f"Heuristic classification: {result.failure_classification}\n\n"
        f"Launch command:\n{result.container_command}\n\n"
        f"Container logs:\n```\n{logs}\n```\n\n"
        "Extract:\n"
        "- summary: ONE sentence — why the container fell.\n"
        "- root_cause: what actually failed and why; cite the key log line.\n"
        "- fix: the single most likely concrete change (which flag/value to "
        "change, or which resource to fix). Be specific and actionable.\n"
        "- config_fixable: true if changing the launch config/flags can fix it; "
        "false if it's infra/external (missing image, driver/CUDA mismatch, "
        "corrupt weights) that a different config won't solve."
    )


def _format_diagnosis(diag: CrashDiagnosis) -> str:
    """Render the diagnosis as the planner/analyzer-facing error string."""
    fixable = "yes" if diag.config_fixable else "no (infra/external)"
    return (
        f"CONTAINER CRASH — {diag.summary}\n"
        f"Root cause: {diag.root_cause}\n"
        f"Suggested fix: {diag.fix}\n"
        f"Config-fixable: {fixable}"
    )


async def crash_diagnostician_node(state: AgentState) -> dict:
    """Diagnose a crashed engine container via a dedicated LLM call.

    No-op unless ``current_result`` is a FAILED container crash with captured
    logs. On success, sets ``result.crash_diagnosis`` and rewrites
    ``result.error`` to the distilled root cause + fix.
    """
    result = state.get("current_result")
    if result is None or result.status != ExperimentStatus.FAILED:
        return {}
    err = _crash_error(result)
    if err is None:
        return {}
    logs = _load_logs(err)
    if not logs.strip():
        return {}

    config = state["config"]
    try:
        diag = await structured_output(
            _build_prompt(result, logs), CrashDiagnosis, config.agent_llm
        )
    except Exception as e:  # noqa: BLE001 — diagnosis must never gate the run
        logger.warning(
            "Crash-diagnosis LLM call failed (%s) — keeping the raw error.", e
        )
        return {}

    logger.info(
        "Container crash diagnosed: %s (config_fixable=%s)",
        diag.summary, diag.config_fixable,
    )
    result.crash_diagnosis = diag
    result.error = _format_diagnosis(diag)
    return {"current_result": result}
