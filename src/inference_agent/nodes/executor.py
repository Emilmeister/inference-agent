"""Executor node — launches engine container via nerdctl and runs benchmark."""

from __future__ import annotations

import asyncio
import logging
import os
import statistics
import time

from inference_agent.benchmark.gpu_monitor import GPUMonitor
from inference_agent.benchmark.runner import (
    AgenticSLO,
    _is_connection_failure,
    get_benchmark_phases,
    run_agentic_long_context_phase,
    run_benchmark_phase,
)
from inference_agent.benchmark.smoke_tests import run_smoke_tests
from inference_agent.engines.base import BaseEngine
from inference_agent.engines.sglang import SGLangEngine
from inference_agent.engines.vllm import VLLMEngine
from inference_agent.nodes.discovery import prefetch_and_normalize_model
from inference_agent.models import (
    BenchmarkResult,
    CeilingProbeInfo,
    ConcurrencyResult,
    EngineType,
    ExperimentError,
    ExperimentResult,
    ExperimentStatus,
    PercentileStats,
    SmokeTestResult,
)
from inference_agent.state import AgentState
from inference_agent.utils.container import (
    extract_error_excerpt,
    get_container_exit_code,
    get_container_logs,
    get_engine_version,
    get_image_digest,
    image_exists_locally,
    pull_image,
    run_container,
    scan_engine_logs,
    stop_container,
    wait_for_healthy,
)
from inference_agent.utils.logging import clear_experiment_context, set_experiment_context
from inference_agent.utils.metrics import extract_kv_cache_metrics, fetch_prometheus_metrics

logger = logging.getLogger(__name__)


async def _capture_failure_logs(
    engine: BaseEngine,
    container_name: str,
    experiment_id: str,
) -> tuple[str, str | None, str | None]:
    """Pull the full container log, persist it, scan it, return (excerpt, path, classification).

    Engine boot failures often die inside multiproc worker processes whose
    tracebacks land far above the wrapper's "Engine core initialization
    failed" tail. A 100-line tail throws the root cause away. We instead
    fetch the entire log, save it to `<storage.logs_dir>/<exp>-<engine>-
    container.log` for post-mortem, and return an excerpt that focuses on
    the first stack trace — small enough to attach to an ExperimentError
    and ship into the analyzer prompt without blowing the context budget.

    We also rescan the FULL log (not just the tail) for fatal patterns and
    return that classification, so even a startup-stage error that died
    inside a worker stderr still gets a precise root_cause instead of the
    generic "startup_crash".
    """
    full_log = await get_container_logs(container_name, tail=0)
    excerpt = extract_error_excerpt(full_log)
    scan = scan_engine_logs(full_log)
    scanned_class = scan.get("classification") if scan.get("state") == "fatal" else None
    log_path: str | None = None
    try:
        logs_dir = engine.config.storage.logs_dir
        os.makedirs(logs_dir, exist_ok=True)
        # container_name is `bench-<engine>-<id>`; reuse its engine token so
        # we never depend on isinstance plumbing.
        parts = container_name.split("-")
        engine_tag = parts[1] if len(parts) >= 3 else "engine"
        log_path = os.path.join(
            logs_dir, f"{experiment_id}-{engine_tag}-container.log"
        )
        with open(log_path, "w") as f:
            f.write(full_log)
    except Exception as e:
        # Persistence is best-effort — never let a write failure mask the
        # actual engine error we're trying to report.
        logger.warning("Could not persist full container log: %s", e)
        log_path = None
    return excerpt, log_path, scanned_class


def _get_engine(state: AgentState) -> BaseEngine:
    config = state["config"]
    experiment = state["current_config"]
    if experiment.engine == EngineType.VLLM:
        return VLLMEngine(config)
    return SGLangEngine(config)


async def _start_engine(
    engine: BaseEngine,
    container_args: list[str],
    container_name: str,
    experiment_id: str,
) -> tuple[str | None, list[ExperimentError], float]:
    """Start engine container and wait for health check.

    Returns (container_id, errors, time_to_healthy_sec).
    If container_id is None, startup failed.
    """
    errors: list[ExperimentError] = []

    # Stop any previous container
    await stop_container(container_name)

    # Ensure the engine image is present locally before `nerdctl run -d`.
    # `nerdctl run` on a missing image triggers an implicit pull that can take
    # 5-15 min for multi-GB engine images, blowing past any reasonable run
    # timeout. Doing pull as an explicit step keeps timeouts honest and lets
    # us classify pull failures distinctly from container start failures.
    image = engine.image()
    startup_start = time.time()
    if not await image_exists_locally(image):
        pull_timeout = engine.config.startup.image_pull_timeout_sec
        logger.info(
            "Image %s not present locally, pulling (timeout %ds)...",
            image, pull_timeout,
        )
        try:
            await pull_image(image, timeout=pull_timeout)
            logger.info("Image %s pulled successfully", image)
        except RuntimeError as e:
            errors.append(ExperimentError(
                stage="startup",
                message=str(e),
                details={"classification": "image_pull_failed", "image": image},
            ))
            return None, errors, time.time() - startup_start
    else:
        logger.info("Image %s already present locally, skipping pull", image)

    # Run container. Image is local but `nerdctl run -d` itself can be slow
    # under NVIDIA runtime with multi-GPU + new CUDA images — timeout is
    # configurable via startup.container_run_timeout_sec.
    run_timeout = engine.config.startup.container_run_timeout_sec
    try:
        container_id = await run_container(container_args, timeout=run_timeout)
        logger.info("Container started: %s", container_id[:12])
    except (RuntimeError, asyncio.TimeoutError) as e:
        excerpt, log_path, scanned_class = await _capture_failure_logs(
            engine, container_name, experiment_id
        )
        if isinstance(e, asyncio.TimeoutError):
            message = (
                f"Container start timed out after {run_timeout}s "
                f"(nerdctl run -d did not return a container ID — likely slow "
                f"NVIDIA runtime or partial image. Bump "
                f"startup.container_run_timeout_sec)"
            )
            fallback_class = "startup_timeout"
        else:
            message = f"Container start failed: {e}"
            fallback_class = "startup_error"
        # Prefer the scanned root cause over the generic wrapper-level class.
        classification = scanned_class or fallback_class
        details: dict = {"logs": excerpt, "classification": classification}
        if log_path:
            details["log_path"] = log_path
        errors.append(ExperimentError(
            stage="startup",
            message=message,
            details=details,
        ))
        return None, errors, time.time() - startup_start

    # Wait for health check
    logger.info("Waiting for engine health check...")
    health_result = await wait_for_healthy(
        engine.health_url(),
        timeout_sec=engine.config.startup.hard_timeout_sec,
        idle_timeout_sec=engine.config.startup.idle_timeout_sec,
        log_scan_interval_sec=engine.config.startup.log_scan_interval_sec,
        container_name=container_name,
    )
    time_to_healthy = time.time() - startup_start

    if not health_result["healthy"]:
        excerpt, log_path, scanned_class = await _capture_failure_logs(
            engine, container_name, experiment_id
        )
        exit_code = await get_container_exit_code(container_name)
        # Order of preference for the failure classification:
        #   1. scan of the FULL container log (catches root-cause patterns
        #      that the in-flight 200-line tail scan missed),
        #   2. classification picked up by wait_for_healthy's incremental
        #      scanner during the wait,
        #   3. generic "healthcheck_timeout" as a last resort.
        classification = (
            scanned_class
            or health_result.get("classification")
            or "healthcheck_timeout"
        )
        marker = health_result.get("marker")
        message = (
            f"Engine did not become healthy "
            f"(reason={health_result['reason']}, classification={classification}"
            + (f", marker='{marker}'" if marker else "")
            + f", elapsed={time_to_healthy:.0f}s)"
        )
        details: dict = {
            "logs": excerpt,
            "exit_code": exit_code,
            "time_elapsed_sec": time_to_healthy,
            "reason": health_result["reason"],
            "classification": classification,
            "marker": marker,
        }
        if log_path:
            details["log_path"] = log_path
        errors.append(ExperimentError(
            stage="healthcheck",
            message=message,
            details=details,
        ))
        # Show the focused excerpt in the agent log rather than the last 500
        # bytes — for multiproc tracebacks the tail is the wrapper unwind, not
        # the root cause. log_path lets the operator open the full log if the
        # excerpt is still ambiguous.
        location_hint = f" (full log: {log_path})" if log_path else ""
        logger.error("FAILED: %s%s\n%s", message, location_hint, excerpt)
        await stop_container(container_name)
        return None, errors, time_to_healthy

    logger.info("Engine is healthy! (%.1fs)", time_to_healthy)
    return container_id, errors, time_to_healthy


_TIMEOUT_ERROR_PATTERNS: tuple[str, ...] = (
    "timeout",
    "timed out",
    "asyncio.timeouterror",
    "session timeout",
    "per-turn timeout",
)


def _classify_phase_outcome(
    workload_id: str,
    error_details: list[str],
) -> str:
    """Decide whether an over-threshold phase is a real malfunction or a
    ceiling-probe SLO miss.

    Returns:
      "ceiling_probe" — agentic_long_context phase whose errors are dominantly
        timeout-shaped. This is an expected sweep outcome ("this concurrency
        doesn't fit SLO"), NOT a bug. Surfaced via ceiling_probe_phases.
      "malfunction"   — everything else (any non-agentic failure, or agentic
        with non-timeout errors like HTTP 5xx, connection refused, parse
        errors). Surfaced via failed_phases / errors.

    Rationale: agentic_long_context is a deliberate ceiling probe; pushing
    concurrency past SLO is the whole point. Classifying those as failures
    pollutes status=partial, error strings, and the LLM's view of the run.
    """
    if workload_id != "agentic_long_context":
        return "malfunction"
    if not error_details:
        # No sample to inspect — be conservative and call it malfunction so
        # the user/LLM at least sees it.
        return "malfunction"
    samples = [s.lower() for s in error_details if s]
    if not samples:
        return "malfunction"
    timeout_hits = sum(
        1 for s in samples
        if any(p in s for p in _TIMEOUT_ERROR_PATTERNS)
    )
    # Need a clear majority of timeout-shaped errors to call it ceiling.
    return "ceiling_probe" if timeout_hits * 2 >= len(samples) else "malfunction"


def _summarize_ceiling_reason(error_details: list[str]) -> str:
    """One short human-readable phrase for the ceiling-probe reason."""
    for s in error_details:
        low = s.lower()
        if "session timeout" in low:
            return "session timeout"
        if "per-turn timeout" in low or "per_turn" in low:
            return "per-turn timeout"
        if "timeout" in low or "timed out" in low:
            return "request timeout"
    return "slo violation"


def _classify_agentic_miss(result: ConcurrencyResult, slo: AgenticSLO) -> str:
    """Classify a non-viable agentic phase as 'hard' or 'marginal'.

    'hard'     — a real ceiling: any request-level failures over the error SLO,
                 OR a latency metric (ttft/tpot/e2e) overshooting its SLO by
                 more than _AGENTIC_MARGINAL_TOLERANCE. Lock the ceiling at once.
    'marginal' — no excess errors and every violated latency metric sits within
                 the tolerance band just above its SLO. Could be transient
                 timing noise → worth one same-level retry before locking.
    """
    # Real request failures are never "noise" — straight to hard.
    if result.error_rate > slo.session_error_rate:
        return "hard"
    worst_ratio = 1.0
    for actual, limit in (
        (result.ttft_ms.p95, slo.ttft_p95_ms),
        (result.tpot_ms.p95, slo.tpot_p95_ms),
        (result.e2e_latency_ms.p95, slo.e2e_p95_ms),
    ):
        if limit > 0 and actual > limit:
            worst_ratio = max(worst_ratio, actual / limit)
    return "hard" if worst_ratio > 1.0 + _AGENTIC_MARGINAL_TOLERANCE else "marginal"


_AGENTIC_SATURATION_RATE = 0.95  # error_rate at/above which we lock the agentic sweep

# Adaptive agentic ceiling detection. A non-viable agentic phase is either a
# HARD miss (a real ceiling — stop now) or a MARGINAL miss (a latency metric
# only just over SLO, no request failures — possibly transient noise, so re-run
# the SAME concurrency once to confirm before locking). This is better than
# probing the NEXT, higher concurrency to "confirm": more load is near-certain
# to fail too, so it wasted ~10 min on a foregone conclusion AND conflated "is
# this noise?" with "can it take more?". A same-level retry isolates the noise
# question with identical inputs (same seed).
#
# MARGINAL = every violated latency SLO (ttft/tpot/e2e) is within this fraction
# above its limit, and error_rate is within SLO. Anything worse is HARD.
_AGENTIC_MARGINAL_TOLERANCE = 0.15  # +15% over an SLO still counts as marginal

# Experiment-level watchdog: after this many consecutive phases die with
# socket-level errors, we stop the sweep entirely. The engine is gone and
# every remaining phase will just produce another batch of unhelpful
# ECONNREFUSED noise. We allow ONE such phase as the trigger (so a single
# flaky phase doesn't kill an experiment) and bail on the second.
_MAX_CONSECUTIVE_DEAD_PHASES = 2


def _looks_like_dead_engine_phase(result: ConcurrencyResult) -> bool:
    """Decide whether a finished phase reads like the engine has crashed.

    Two signals must agree:
      1. error_rate is at/near 100% — anything lower could still be a real
         load result with a few failures.
      2. The majority of captured error samples are socket-level failures
         ("Cannot connect to host", connection refused, …). HTTP 5xx or
         timeouts don't trigger this — those can happen on a healthy but
         overloaded engine.
    """
    if result.error_rate < _AGENTIC_SATURATION_RATE:
        return False
    samples = [s for s in result.error_details if s]
    if not samples:
        return False
    conn = sum(1 for s in samples if _is_connection_failure(s))
    return conn * 2 >= len(samples)


async def _run_all_phases(
    engine: BaseEngine,
    config: object,
    hardware: object,
    experiment: object,
    seed: int | None,
) -> tuple[list[ConcurrencyResult], list[ExperimentError], list[CeilingProbeInfo]]:
    """Run all benchmark phases and return (passing results, malfunction errors, ceiling probes)."""
    concurrency_results: list[ConcurrencyResult] = []
    phase_errors: list[ExperimentError] = []
    ceiling_probes: list[CeilingProbeInfo] = []

    phases = get_benchmark_phases(
        model_max_context=hardware.model_max_context,
        max_model_len=experiment.max_model_len,
        benchmark_config=config.benchmark,
    )

    error_rate_threshold = config.benchmark.phase_error_rate_threshold
    agentic_slo = AgenticSLO.from_config(config.benchmark)

    async def _run_agentic_phase(c: int, max_o: int, pid: str) -> ConcurrencyResult:
        """Run one agentic_long_context phase at concurrency `c`.

        Factored out so the adaptive ceiling logic can re-run the SAME level
        (identical params + seed → identical workload) to confirm a marginal
        SLO miss is a real ceiling and not transient timing noise.
        """
        return await run_agentic_long_context_phase(
            api_base_url=engine.api_base_url(),
            model_name=config.model_name,
            concurrency=c,
            shared_prefix_tokens=config.benchmark.agentic_shared_prefix_tokens,
            unique_prompt_tokens=config.benchmark.agentic_unique_prompt_tokens,
            max_output_tokens=max_o,
            turns=config.benchmark.agentic_turns_per_session,
            tool_result_min=config.benchmark.agentic_tool_result_min_tokens,
            tool_result_max=config.benchmark.agentic_tool_result_max_tokens,
            session_timeout_sec=config.benchmark.agentic_session_timeout_sec,
            per_turn_timeout_sec=config.benchmark.agentic_per_turn_timeout_sec,
            seed=seed,
            workload_id="agentic_long_context",
            phase_id=pid,
            slo=agentic_slo,
        )

    # Agentic ceiling detection (adaptive). A non-viable agentic phase locks the
    # ceiling and terminates the experiment — but a MARGINAL miss (latency just
    # over SLO, no errors) first gets ONE same-level retry to rule out transient
    # noise; a HARD miss locks immediately. See _classify_agentic_miss. Since the
    # agentic objective is primary, a config that lost the agentic race
    # contributes only informational data via remaining phases — not worth the
    # executor time.
    agentic_ceiling_lock: int | None = None

    # Experiment-level watchdog: count consecutive phases that came back with
    # socket-shaped 100% failure (engine crashed mid-sweep). After the second,
    # we break out of the loop instead of running another dozen phases worth
    # of no-op ECONNREFUSED noise.
    consecutive_dead_phases = 0

    for phase_id, workload_id, concurrency, prompt_len, max_out in phases:
        is_warmup = workload_id == "warmup"
        is_agentic = workload_id == "agentic_long_context"
        duration = 10 if is_warmup else config.benchmark.duration_per_level_sec

        if is_agentic:
            logger.info(
                "  Phase: %s [%s] (c=%d, prompt=%d, max_out=%d, turns=%d) "
                "SLO ttft_p95≤%.0fms tpot_p95≤%.0fms errors≤%.0f%%",
                phase_id, workload_id, concurrency, prompt_len, max_out,
                config.benchmark.agentic_turns_per_session,
                agentic_slo.ttft_p95_ms, agentic_slo.tpot_p95_ms,
                agentic_slo.session_error_rate * 100,
            )
        else:
            logger.info(
                "  Phase: %s [%s] (c=%d, prompt=%d, max_out=%d, dur=%ds)",
                phase_id, workload_id, concurrency, prompt_len, max_out, duration,
            )

        try:
            if is_agentic:
                result = await _run_agentic_phase(concurrency, max_out, phase_id)
            else:
                result = await run_benchmark_phase(
                    api_base_url=engine.api_base_url(),
                    model_name=config.model_name,
                    concurrency=concurrency,
                    prompt_length=prompt_len,
                    max_output_tokens=max_out,
                    duration_sec=duration,
                    warmup=is_warmup,
                    seed=seed,
                    workload_id=workload_id,
                    phase_id=phase_id,
                )
            if not is_warmup:
                # Adaptive agentic ceiling. A non-viable agentic phase that is
                # only MARGINALLY over SLO (latency just past the limit, no
                # request failures) gets ONE same-level retry first — identical
                # workload (same seed), so a pass means the first miss was
                # transient noise and we keep the level. A HARD miss, or a retry
                # that still fails, is a real ceiling.
                if is_agentic and not result.viable:
                    if _classify_agentic_miss(result, agentic_slo) == "marginal":
                        logger.info(
                            "  Phase %s MARGINAL agentic miss at c=%d (%s) — "
                            "re-running the SAME level once to rule out "
                            "transient noise.",
                            phase_id, concurrency, "; ".join(result.slo_violations),
                        )
                        result = await _run_agentic_phase(concurrency, max_out, phase_id)
                        if result.viable:
                            logger.info(
                                "  Retry at c=%d PASSED (throughput=%.1f tok/s, "
                                "ttft_p95=%.0fms, tpot_p95=%.1fms) — first miss "
                                "was transient, keeping this level.",
                                concurrency, result.output_tokens_per_sec,
                                result.ttft_ms.p95, result.tpot_ms.p95,
                            )
                        else:
                            logger.info(
                                "  Retry at c=%d still NOT VIABLE (%s) — "
                                "confirmed ceiling.",
                                concurrency, "; ".join(result.slo_violations),
                            )

                # Agentic phases are gated by full SLO (TTFT + tpot + error rate
                # + e2e). A still-non-viable phase (hard miss, or a marginal miss
                # whose retry also failed) is a real ceiling: record the probe,
                # lock, and terminate — the agentic objective is primary, so the
                # remaining phases would only add informational data.
                if is_agentic and not result.viable:
                    reason = (
                        "; ".join(result.slo_violations)
                        if result.slo_violations
                        else _summarize_ceiling_reason(list(result.error_details))
                    )
                    ceiling_probes.append(CeilingProbeInfo(
                        phase_id=phase_id,
                        workload_id=workload_id,
                        concurrency=concurrency,
                        prompt_length=prompt_len,
                        error_rate=result.error_rate,
                        errors=result.errors,
                        reason=reason,
                    ))
                    agentic_ceiling_lock = concurrency
                    logger.info(
                        "  Agentic ceiling locked at c=%d (%s) — terminating "
                        "experiment: this config lost the agentic SLO race, "
                        "remaining phases would only add informational data.",
                        concurrency, reason,
                    )
                    break
                # Error-rate gate for non-agentic workloads (and agentic
                # phases that passed SLO but somehow tripped the error gate
                # — currently impossible since session_error_rate ≤ phase
                # threshold, but guard kept for safety). Real malfunctions
                # land in phase_errors so the LLM sees them.
                elif result.error_rate > error_rate_threshold:
                    outcome = _classify_phase_outcome(
                        workload_id, list(result.error_details)
                    )
                    if outcome == "ceiling_probe":
                        reason = _summarize_ceiling_reason(list(result.error_details))
                        ceiling_probes.append(CeilingProbeInfo(
                            phase_id=phase_id,
                            workload_id=workload_id,
                            concurrency=concurrency,
                            prompt_length=prompt_len,
                            error_rate=result.error_rate,
                            errors=result.errors,
                            reason=reason,
                        ))
                    else:
                        logger.warning(
                            "  Phase %s error_rate=%.1f%% exceeds threshold %.1f%%, marking invalid",
                            phase_id, result.error_rate * 100, error_rate_threshold * 100,
                        )
                        phase_errors.append(ExperimentError(
                            stage="benchmark_phase",
                            message=(
                                f"Phase {phase_id} error_rate={result.error_rate:.2f} "
                                f"exceeds threshold"
                                + (
                                    f" (sample error: {result.error_details[0][:200]})"
                                    if result.error_details
                                    else ""
                                )
                            ),
                            details={
                                "phase_id": phase_id,
                                "workload_id": workload_id,
                                "concurrency": concurrency,
                                "prompt_length": prompt_len,
                                "error_rate": result.error_rate,
                                "errors": result.errors,
                                "threshold": error_rate_threshold,
                                "error_samples": list(result.error_details[:3]),
                            },
                        ))
                else:
                    concurrency_results.append(result)

            # Update the dead-engine watchdog regardless of which branch above
            # took the phase — warmup is excluded because we don't aggregate
            # it anyway, and a one-off warmup blip shouldn't tip the counter.
            if not is_warmup:
                if _looks_like_dead_engine_phase(result):
                    consecutive_dead_phases += 1
                    if consecutive_dead_phases >= _MAX_CONSECUTIVE_DEAD_PHASES:
                        logger.warning(
                            "  Aborting remaining phases: %d consecutive "
                            "phases died with socket-level errors — engine "
                            "is gone, no point loading the rest of the sweep.",
                            consecutive_dead_phases,
                        )
                        break
                else:
                    consecutive_dead_phases = 0
        except Exception as e:
            logger.error("  Phase %s failed: %s", phase_id, e)
            phase_errors.append(ExperimentError(
                stage="benchmark_phase",
                message=str(e),
                details={
                    "phase_id": phase_id,
                    "workload_id": workload_id,
                    "concurrency": concurrency,
                    "prompt_length": prompt_len,
                },
            ))

    return concurrency_results, phase_errors, ceiling_probes


def _classify_failure(
    startup_errors: list[ExperimentError],
    phase_errors: list[ExperimentError],
    correctness_gate_passed: bool,
    post_correctness_degraded: bool,
    container_crashed: bool,
) -> str | None:
    """Classify the experiment failure reason."""
    if (
        not startup_errors and not phase_errors
        and correctness_gate_passed and not post_correctness_degraded
        and not container_crashed
    ):
        return None

    for err in startup_errors:
        if err.stage == "startup":
            logs = err.details.get("logs", "").lower()
            if "oom" in logs or "out of memory" in logs or "cuda" in logs:
                return "oom"
            return "startup_crash"
        if err.stage == "healthcheck":
            # Prefer classification from log scanner over generic timeout —
            # tells the planner whether it was argparse, OOM, hard cap, idle
            # stall, etc.
            scanned = err.details.get("classification")
            if scanned:
                return scanned
            exit_code = err.details.get("exit_code")
            if exit_code == 137:
                return "oom"
            return "healthcheck_timeout"

    if not correctness_gate_passed:
        return "correctness_failure"

    if post_correctness_degraded or container_crashed:
        return "runtime_crash"

    if phase_errors:
        return "benchmark_error"

    return None


async def executor_node(state: AgentState) -> dict:
    """Launch the inference engine container via nerdctl and run benchmarks.

    Flow:
    1. Start engine container, wait for health
    2. Capture engine version
    3. Run correctness gate (smoke tests) — if fails, skip performance
    4. Run performance phases with error-rate gating
    5. Run post-performance correctness check
    6. Aggregate results with workload-aware metrics
    """
    experiment = state["current_config"]
    hardware = state["hardware"]
    config = state["config"]
    engine = _get_engine(state)

    # Set structured logging context for this experiment
    set_experiment_context(
        experiment_id=experiment.experiment_id,
        engine=experiment.engine.value,
    )

    start_time = time.time()
    container_name = engine.container_name(experiment)

    exp_num = state.get("experiments_count", 0) + 1
    max_exp = config.experiments.max_experiments
    logger.info("")
    logger.info("=" * 60)
    logger.info(
        "EXPERIMENT %d/%d: %s | %s | TP=%d | quant=%s",
        exp_num, max_exp,
        experiment.experiment_id,
        experiment.engine.value.upper(),
        experiment.tensor_parallel_size,
        experiment.quantization or "none",
    )
    logger.info("Rationale: %s", experiment.rationale)
    logger.info("=" * 60)

    # Build nerdctl command
    container_args = engine.build_container_args(experiment)
    container_command = " ".join(container_args)
    logger.info("Container command: %s", container_command)

    # Resolve image digest for reproducibility
    image_digest = await get_image_digest(engine.image())

    # Determine benchmark seed
    seed = config.benchmark.seed

    # ── Step 0: Prefetch external speculative draft model ─────────────
    # Engines run with HF_HUB_OFFLINE=1 (so the patched config.json is used
    # instead of being overwritten by HF re-validation), so any draft model
    # the planner picked must be in the host cache before container start.
    # snapshot_download is idempotent — if the draft is already cached this
    # is a no-op. A failed download (bad repo ID, auth, network) surfaces
    # here as a clean `prefetch_failed` failure instead of a 20-minute
    # container hang during engine startup.
    draft_model = experiment.speculative_draft_model
    if draft_model and config.startup.prefetch_model:
        logger.info("Prefetching speculative_draft_model: %s", draft_model)
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                prefetch_and_normalize_model,
                draft_model,
                config.container.host_cache_dir,
                None,  # no separate revision pin for draft model
                config.hf_token,
                config.startup.prefetch_allow_patterns,
                True,  # raise_on_failure
            )
        except Exception as e:
            error_msg = f"Failed to prefetch speculative_draft_model={draft_model}: {e}"
            logger.error(error_msg)
            clear_experiment_context()
            return {
                "current_result": ExperimentResult(
                    experiment_id=experiment.experiment_id,
                    engine=experiment.engine,
                    model=config.model_name,
                    hardware=hardware,
                    config=experiment,
                    is_baseline=experiment.is_baseline,
                    status=ExperimentStatus.FAILED,
                    error=error_msg,
                    errors=[ExperimentError(
                        stage="prefetch",
                        message=error_msg,
                        details={
                            "classification": "prefetch_failed",
                            "draft_model": draft_model,
                        },
                    )],
                    container_command=container_command,
                    container_args=container_args,
                    container_image_digest=image_digest,
                    benchmark_seed=seed,
                    duration_seconds=time.time() - start_time,
                    time_to_healthy_sec=0.0,
                    failure_classification="prefetch_failed",
                )
            }

    # ── Step 1: Start engine ──────────────────────────────────────────
    container_id, startup_errors, time_to_healthy = await _start_engine(
        engine, container_args, container_name, experiment.experiment_id,
    )

    if container_id is None:
        failure_class = _classify_failure(startup_errors, [], False, False, True)
        clear_experiment_context()
        error_msg = "; ".join(e.message for e in startup_errors)
        return {
            "current_result": ExperimentResult(
                experiment_id=experiment.experiment_id,
                engine=experiment.engine,
                model=config.model_name,
                hardware=hardware,
                config=experiment,
                is_baseline=experiment.is_baseline,
                status=ExperimentStatus.FAILED,
                error=error_msg,
                errors=startup_errors,
                container_command=container_command,
                container_args=container_args,
                container_image_digest=image_digest,
                benchmark_seed=seed,
                duration_seconds=time.time() - start_time,
                time_to_healthy_sec=time_to_healthy,
                failure_classification=failure_class,
            )
        }

    # ── Step 2: Capture engine version ────────────────────────────────
    engine_version = await get_engine_version(engine.api_base_url())
    if engine_version:
        logger.info("Engine version: %s", engine_version)

    # ── Step 3: Correctness gate (smoke tests BEFORE performance) ─────
    logger.info("Running correctness gate...")
    try:
        smoke_results = await run_smoke_tests(engine.api_base_url(), config.model_name)
    except Exception as e:
        logger.error("Correctness gate crashed: %s", e)
        smoke_results = SmokeTestResult()

    correctness_gate_passed = smoke_results.gate_passed

    if not correctness_gate_passed:
        logger.warning(
            "CORRECTNESS GATE FAILED — skipping performance phases. "
            "basic_chat=%s, tool_calling=%s, json_schema=%s",
            smoke_results.basic_chat,
            smoke_results.tool_calling,
            smoke_results.json_schema,
        )

        # Collect Prometheus metrics even on correctness failure
        prom_metrics = await fetch_prometheus_metrics(engine.metrics_url())
        kv_metrics = extract_kv_cache_metrics(prom_metrics, experiment.engine.value)

        await stop_container(container_name)
        clear_experiment_context()

        failure_class = _classify_failure(startup_errors, [], False, False, False)
        return {
            "current_result": ExperimentResult(
                experiment_id=experiment.experiment_id,
                engine=experiment.engine,
                model=config.model_name,
                hardware=hardware,
                config=experiment,
                is_baseline=experiment.is_baseline,
                status=ExperimentStatus.FAILED_CORRECTNESS,
                error="Correctness gate failed: " + "; ".join(
                    f"{name}={getattr(smoke_results, name)}"
                    for name in ("basic_chat", "tool_calling", "json_schema")
                    if not getattr(smoke_results, name)
                ),
                errors=startup_errors + [ExperimentError(
                    stage="correctness_gate",
                    message="Correctness gate failed",
                    details={
                        "basic_chat": smoke_results.basic_chat_detail,
                        "tool_calling": smoke_results.tool_calling_detail,
                        "tool_required": smoke_results.tool_required_detail,
                        "json_mode": smoke_results.json_mode_detail,
                        "json_schema": smoke_results.json_schema_detail,
                    },
                )],
                smoke_tests=smoke_results,
                container_command=container_command,
                container_args=container_args,
                container_image_digest=image_digest,
                engine_version=engine_version,
                benchmark_seed=seed,
                duration_seconds=time.time() - start_time,
                time_to_healthy_sec=time_to_healthy,
                failure_classification=failure_class,
                correctness_gate_passed=False,
            )
        }

    logger.info("Correctness gate PASSED")

    # ── Step 4: Start GPU monitoring and run performance phases ───────
    gpu_monitor = GPUMonitor(interval_ms=1000)
    await gpu_monitor.start()

    logger.info("Starting benchmark...")
    concurrency_results, phase_errors, ceiling_probes = await _run_all_phases(
        engine, config, hardware, experiment, seed,
    )

    # Collect Prometheus metrics
    prom_metrics = await fetch_prometheus_metrics(engine.metrics_url())
    kv_metrics = extract_kv_cache_metrics(prom_metrics, experiment.engine.value)

    # ── Step 5: Post-performance correctness regression check ─────────
    logger.info("Running post-benchmark correctness check...")
    post_correctness_degraded = False
    try:
        post_smoke = await run_smoke_tests(engine.api_base_url(), config.model_name)
    except Exception as e:
        logger.error("Post-benchmark correctness crashed: %s", e)
        post_smoke = SmokeTestResult()
        post_correctness_degraded = True

    if not post_correctness_degraded and not post_smoke.basic_chat:
        post_correctness_degraded = True
        logger.warning(
            "POST-BENCHMARK CORRECTNESS DEGRADED: basic_chat failed after load"
        )

    # ── Step 6: Stop monitoring and container ─────────────────────────
    gpu_snapshots = await gpu_monitor.stop()
    gpu_agg = GPUMonitor.aggregate_snapshots(gpu_snapshots)

    # Check if container is still alive
    from inference_agent.utils.container import _is_container_running
    container_alive = await _is_container_running(container_name)
    container_crashed = not container_alive

    if container_crashed:
        exit_code = await get_container_exit_code(container_name)
        logs = await get_container_logs(container_name)
        logger.warning(
            "Container crashed during benchmark! exit_code=%s, last logs:\n%s",
            exit_code, logs[-500:],
        )
        phase_errors.append(ExperimentError(
            stage="runtime",
            message=f"Container crashed during benchmark (exit_code={exit_code})",
            details={"exit_code": exit_code, "logs": logs[:5000]},
        ))

    await stop_container(container_name)

    # ── Step 7: Aggregate results ─────────────────────────────────────
    benchmark = _aggregate_benchmark(
        concurrency_results, gpu_agg, kv_metrics, config.benchmark,
        ceiling_probes=ceiling_probes,
    )

    duration = time.time() - start_time
    all_errors = startup_errors + phase_errors

    # Determine status
    if container_crashed or post_correctness_degraded:
        status = ExperimentStatus.PARTIAL
    elif not concurrency_results:
        status = ExperimentStatus.PARTIAL
    elif phase_errors:
        status = ExperimentStatus.PARTIAL
    else:
        status = ExperimentStatus.SUCCESS

    failure_class = _classify_failure(
        startup_errors, phase_errors,
        correctness_gate_passed, post_correctness_degraded, container_crashed,
    )

    logger.info(
        "Experiment %s complete: status=%s, peak_throughput=%.1f tok/s, "
        "low_ttft_p95=%.1f ms, duration=%.0fs, phase_errors=%d, "
        "ceiling_probes=%d, correctness_gate=%s, post_correctness_degraded=%s",
        experiment.experiment_id,
        status.value,
        benchmark.peak_output_tokens_per_sec,
        benchmark.low_concurrency_ttft_p95_ms,
        duration,
        len(phase_errors),
        len(ceiling_probes),
        correctness_gate_passed,
        post_correctness_degraded,
    )

    clear_experiment_context()

    return {
        "current_result": ExperimentResult(
            experiment_id=experiment.experiment_id,
            engine=experiment.engine,
            model=config.model_name,
            hardware=hardware,
            config=experiment,
            is_baseline=experiment.is_baseline,
            status=status,
            error="; ".join(e.message for e in all_errors) if all_errors else None,
            errors=all_errors,
            benchmark=benchmark,
            smoke_tests=smoke_results,
            container_command=container_command,
            container_args=container_args,
            container_image_digest=image_digest,
            engine_version=engine_version,
            benchmark_seed=seed,
            duration_seconds=duration,
            time_to_healthy_sec=time_to_healthy,
            failure_classification=failure_class,
            correctness_gate_passed=correctness_gate_passed,
            post_benchmark_correctness=post_smoke if post_smoke else None,
            ceiling_probe_phases=ceiling_probes,
        )
    }


def _aggregate_benchmark(
    results: list[ConcurrencyResult],
    gpu_agg: dict[int, dict],
    kv_metrics: dict,
    benchmark_config=None,
    ceiling_probes: list[CeilingProbeInfo] | None = None,
) -> BenchmarkResult:
    """Aggregate per-phase results into a single BenchmarkResult.

    Workload-aware aggregation:
    - peak_throughput: max from agent_short + throughput workloads (not stress/long_context/agentic_long_context)
    - low_concurrency_ttft_p95: median of p95 TTFTs from c=1 agent_short phases
    - max_viable_agentic_concurrency: highest agentic concurrency that passes
      error_rate gate + TTFT p95 SLO + E2E p95 SLO (см. BenchmarkConfig agentic_*).
    """
    if not results:
        return BenchmarkResult()

    # Workload-aware peak throughput: exclude stress and long_context
    perf_results = [
        r for r in results
        if r.workload_id in ("agent_short", "throughput", "")
    ]
    if perf_results:
        peak_throughput_result = max(perf_results, key=lambda r: r.output_tokens_per_sec)
    else:
        peak_throughput_result = max(results, key=lambda r: r.output_tokens_per_sec)

    # Low-concurrency latency: median of c=1 agent_short phases (not long_context)
    low_conc = [
        r for r in results
        if r.concurrency == 1 and r.workload_id in ("agent_short", "")
    ]
    if low_conc:
        ttft_p95_values = [r.ttft_ms.p95 for r in low_conc if r.ttft_ms.p95 > 0]
        low_ttft_p95 = statistics.median(ttft_p95_values) if ttft_p95_values else 0.0
        tpot_p95_values = [r.tpot_ms.p95 for r in low_conc if r.tpot_ms.p95 > 0]
        low_tpot_p95 = statistics.median(tpot_p95_values) if tpot_p95_values else 0.0
        ttft_cv_values = [r.ttft_ms.cv for r in low_conc if r.ttft_ms.cv > 0]
        low_ttft_cv = statistics.median(ttft_cv_values) if ttft_cv_values else 0.0
    else:
        # Fallback: use any c=1 results
        any_low = [r for r in results if r.concurrency == 1]
        low_ttft_p95 = statistics.median([r.ttft_ms.p95 for r in any_low]) if any_low else 0.0
        low_tpot_p95 = statistics.median([r.tpot_ms.p95 for r in any_low]) if any_low else 0.0
        cv_values = [r.ttft_ms.cv for r in any_low if r.ttft_ms.cv > 0]
        low_ttft_cv = statistics.median(cv_values) if cv_values else 0.0

    # Noise indicator for peak throughput: cv of e2e_latency at the phase that
    # won peak. e2e is the right proxy because output_tokens_per_sec is itself
    # a phase-level scalar without dispersion — its instability lives in the
    # spread of per-request latencies.
    peak_e2e_cv = peak_throughput_result.e2e_latency_ms.cv

    # GPU metrics
    gpu_util = [gpu_agg[i]["util_avg"] for i in sorted(gpu_agg)]
    gpu_mem = [gpu_agg[i]["mem_peak"] for i in sorted(gpu_agg)]
    gpu_power = [gpu_agg[i]["power_avg"] for i in sorted(gpu_agg)]
    gpu_temp = [gpu_agg[i]["temp_max"] for i in sorted(gpu_agg)]

    # ── Agentic derived metrics (max viable parallel agents, saturation) ──
    agentic_metrics = _compute_agentic_metrics(
        results, benchmark_config, ceiling_probes=ceiling_probes or [],
    )

    return BenchmarkResult(
        ttft_ms=peak_throughput_result.ttft_ms,
        tpot_ms=peak_throughput_result.tpot_ms,
        itl_ms=peak_throughput_result.itl_ms,
        e2e_latency_ms=peak_throughput_result.e2e_latency_ms,
        peak_requests_per_sec=peak_throughput_result.requests_per_sec,
        peak_output_tokens_per_sec=peak_throughput_result.output_tokens_per_sec,
        peak_total_tokens_per_sec=peak_throughput_result.total_tokens_per_sec,
        low_concurrency_ttft_p95_ms=low_ttft_p95,
        low_concurrency_tpot_p95_ms=low_tpot_p95,
        peak_throughput_e2e_cv=peak_e2e_cv,
        low_concurrency_ttft_cv=low_ttft_cv,
        kv_cache_usage_percent=kv_metrics.get("kv_cache_usage_percent", 0.0),
        prefix_cache_hit_rate=kv_metrics.get("prefix_cache_hit_rate", 0.0),
        gpu_utilization_percent=gpu_util,
        gpu_memory_used_mb=gpu_mem,
        gpu_power_draw_watts=gpu_power,
        gpu_temperature_celsius=gpu_temp,
        concurrency_results=results,
        max_viable_agentic_concurrency=agentic_metrics["max_viable"],
        agentic_concurrency_ceiling_hit=agentic_metrics["ceiling_hit"],
        agentic_saturation_concurrency=agentic_metrics["saturation"],
        agentic_peak_output_tokens_per_sec=agentic_metrics["peak_throughput"],
        agentic_tpot_p95_ms=agentic_metrics["tpot_p95"],
        agentic_ttft_p95_ms=agentic_metrics["ttft_p95"],
    )


def _compute_agentic_metrics(
    results: list[ConcurrencyResult],
    benchmark_config,
    ceiling_probes: list[CeilingProbeInfo] | None = None,
) -> dict:
    """Derive headline agentic metrics from per-phase results.

    Returns dict keys: max_viable, ceiling_hit, saturation, peak_throughput,
    tpot_p95, ttft_p95. Empty/zero when no agentic phases ran.

    `viable` is now decided by the runner (AgenticSLO) — `result.viable` is
    the canonical signal. We only re-derive the geometry of the sweep here
    (max viable concurrency, where peak throughput sat, and the tpot/ttft
    at the max-viable phase so the analyzer can tie-break two configs with
    the same max_viable_c).
    """
    agentic_results = [r for r in results if r.workload_id == "agentic_long_context"]
    ceiling_probes = ceiling_probes or []
    if not agentic_results and not ceiling_probes:
        return {
            "max_viable": 0,
            "ceiling_hit": False,
            "saturation": 0,
            "peak_throughput": 0.0,
            "tpot_p95": 0.0,
            "ttft_p95": 0.0,
        }

    viable_results = [r for r in agentic_results if r.viable]
    max_viable = max((r.concurrency for r in viable_results), default=0)

    # sweep_max должен включать и ceiling-probe'ы, иначе при «c=8 прошло,
    # c=16/32/64 ceiling-probe» мы бы получили sweep_max=8, max_viable=8,
    # ceiling_hit=True — ложно («потолок не найден»). Правильно — False.
    sweep_candidates = (
        [r.concurrency for r in agentic_results]
        + [p.concurrency for p in ceiling_probes]
    )
    sweep_max = max(sweep_candidates) if sweep_candidates else 0
    ceiling_hit = max_viable > 0 and max_viable == sweep_max

    # Peak throughput across VIABLE agentic phases (a non-viable phase
    # technically generated tokens but at terrible latency — we don't want
    # to credit it).
    if viable_results:
        peak_result = max(viable_results, key=lambda r: r.output_tokens_per_sec)
        saturation = peak_result.concurrency
        peak_throughput = peak_result.output_tokens_per_sec
    else:
        saturation = 0
        peak_throughput = 0.0

    # tpot/ttft at the max-viable phase — tells the analyzer how snappy
    # responses were at the headline concurrency. If max_viable == 0,
    # leave both at 0.
    tpot_p95 = 0.0
    ttft_p95 = 0.0
    if max_viable > 0:
        anchor = next((r for r in viable_results if r.concurrency == max_viable), None)
        if anchor is not None:
            tpot_p95 = anchor.tpot_ms.p95
            ttft_p95 = anchor.ttft_ms.p95

    return {
        "max_viable": max_viable,
        "ceiling_hit": ceiling_hit,
        "saturation": saturation,
        "peak_throughput": peak_throughput,
        "tpot_p95": tpot_p95,
        "ttft_p95": ttft_p95,
    }
