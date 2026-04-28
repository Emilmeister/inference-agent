"""Executor node — launches Docker container and runs benchmark."""

from __future__ import annotations

import asyncio
import logging
import statistics
import time

from inference_agent.benchmark.gpu_monitor import GPUMonitor
from inference_agent.benchmark.runner import get_benchmark_phases, run_benchmark_phase
from inference_agent.benchmark.smoke_tests import run_smoke_tests
from inference_agent.engines.base import BaseEngine
from inference_agent.engines.sglang import SGLangEngine
from inference_agent.engines.vllm import VLLMEngine
from inference_agent.models import (
    BenchmarkResult,
    ConcurrencyResult,
    EngineType,
    ExperimentError,
    ExperimentResult,
    ExperimentStatus,
    PercentileStats,
    SmokeTestResult,
)
from inference_agent.state import AgentState
from inference_agent.utils.docker import (
    get_container_exit_code,
    get_container_logs,
    get_engine_version,
    get_image_digest,
    image_exists_locally,
    pull_image,
    run_container,
    stop_container,
    wait_for_healthy,
)
from inference_agent.utils.logging import clear_experiment_context, set_experiment_context
from inference_agent.utils.metrics import extract_kv_cache_metrics, fetch_prometheus_metrics

logger = logging.getLogger(__name__)


def _get_engine(state: AgentState) -> BaseEngine:
    config = state["config"]
    experiment = state["current_config"]
    if experiment.engine == EngineType.VLLM:
        return VLLMEngine(config)
    return SGLangEngine(config)


async def _start_engine(
    engine: BaseEngine,
    docker_args: list[str],
    container_name: str,
    experiment_id: str,
) -> tuple[str | None, list[ExperimentError], float]:
    """Start Docker container and wait for health check.

    Returns (container_id, errors, time_to_healthy_sec).
    If container_id is None, startup failed.
    """
    errors: list[ExperimentError] = []

    # Stop any previous container
    await stop_container(container_name)

    # Ensure the engine image is present locally before `docker run -d`.
    # `docker run` on a missing image triggers an implicit pull that can take
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

    # Run container — image is now guaranteed local, so 60s is plenty for
    # `docker run -d` to return the container ID.
    try:
        container_id = await run_container(docker_args, timeout=60)
        logger.info("Container started: %s", container_id[:12])
    except (RuntimeError, asyncio.TimeoutError) as e:
        logs = await get_container_logs(container_name)
        if isinstance(e, asyncio.TimeoutError):
            message = "Container start timed out after 60s (docker run -d hung)"
            classification = "startup_timeout"
        else:
            message = f"Container start failed: {e}"
            classification = "startup_error"
        errors.append(ExperimentError(
            stage="startup",
            message=message,
            details={"logs": logs[:5000], "classification": classification},
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
        logs = await get_container_logs(container_name)
        exit_code = await get_container_exit_code(container_name)
        classification = health_result.get("classification") or "healthcheck_timeout"
        marker = health_result.get("marker")
        message = (
            f"Engine did not become healthy "
            f"(reason={health_result['reason']}, classification={classification}"
            + (f", marker='{marker}'" if marker else "")
            + f", elapsed={time_to_healthy:.0f}s)"
        )
        errors.append(ExperimentError(
            stage="healthcheck",
            message=message,
            details={
                "logs": logs[:5000],
                "exit_code": exit_code,
                "time_elapsed_sec": time_to_healthy,
                "reason": health_result["reason"],
                "classification": classification,
                "marker": marker,
            },
        ))
        logger.error("FAILED: %s\nLast logs:\n%s", message, logs[-500:])
        await stop_container(container_name)
        return None, errors, time_to_healthy

    logger.info("Engine is healthy! (%.1fs)", time_to_healthy)
    return container_id, errors, time_to_healthy


async def _run_all_phases(
    engine: BaseEngine,
    config: object,
    hardware: object,
    experiment: object,
    seed: int | None,
) -> tuple[list[ConcurrencyResult], list[ExperimentError]]:
    """Run all benchmark phases and return results with any phase errors."""
    concurrency_results: list[ConcurrencyResult] = []
    phase_errors: list[ExperimentError] = []

    phases = get_benchmark_phases(
        model_max_context=hardware.model_max_context,
        max_model_len=experiment.max_model_len,
        benchmark_config=config.benchmark,
    )

    error_rate_threshold = config.benchmark.phase_error_rate_threshold

    for phase_id, workload_id, concurrency, prompt_len, max_out in phases:
        is_warmup = workload_id == "warmup"
        duration = 10 if is_warmup else config.benchmark.duration_per_level_sec

        logger.info(
            "  Phase: %s [%s] (c=%d, prompt=%d, max_out=%d, dur=%ds)",
            phase_id, workload_id, concurrency, prompt_len, max_out, duration,
        )

        try:
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
                # Error-rate gate: discard phases with too many errors
                if result.error_rate > error_rate_threshold:
                    logger.warning(
                        "  Phase %s error_rate=%.1f%% exceeds threshold %.1f%%, marking invalid",
                        phase_id, result.error_rate * 100, error_rate_threshold * 100,
                    )
                    phase_errors.append(ExperimentError(
                        stage="benchmark_phase",
                        message=f"Phase {phase_id} error_rate={result.error_rate:.2f} exceeds threshold",
                        details={
                            "phase_id": phase_id,
                            "workload_id": workload_id,
                            "error_rate": result.error_rate,
                            "errors": result.errors,
                            "threshold": error_rate_threshold,
                        },
                    ))
                else:
                    concurrency_results.append(result)
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

    return concurrency_results, phase_errors


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
    """Launch the inference engine in Docker and run benchmarks.

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

    # Build Docker command
    docker_args = engine.build_docker_args(experiment)
    docker_command = " ".join(docker_args)
    logger.info("Docker command: %s", docker_command)

    # Resolve image digest for reproducibility
    image_digest = await get_image_digest(engine.image())

    # Determine benchmark seed
    seed = config.benchmark.seed

    # ── Step 1: Start engine ──────────────────────────────────────────
    container_id, startup_errors, time_to_healthy = await _start_engine(
        engine, docker_args, container_name, experiment.experiment_id,
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
                status=ExperimentStatus.FAILED,
                error=error_msg,
                errors=startup_errors,
                docker_command=docker_command,
                docker_args=docker_args,
                docker_image_digest=image_digest,
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
                docker_command=docker_command,
                docker_args=docker_args,
                docker_image_digest=image_digest,
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
    concurrency_results, phase_errors = await _run_all_phases(
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
    from inference_agent.utils.docker import _is_container_running
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
    benchmark = _aggregate_benchmark(concurrency_results, gpu_agg, kv_metrics)

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
        "correctness_gate=%s, post_correctness_degraded=%s",
        experiment.experiment_id,
        status.value,
        benchmark.peak_output_tokens_per_sec,
        benchmark.low_concurrency_ttft_p95_ms,
        duration,
        len(phase_errors),
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
            status=status,
            error="; ".join(e.message for e in all_errors) if all_errors else None,
            errors=all_errors,
            benchmark=benchmark,
            smoke_tests=smoke_results,
            docker_command=docker_command,
            docker_args=docker_args,
            docker_image_digest=image_digest,
            engine_version=engine_version,
            benchmark_seed=seed,
            duration_seconds=duration,
            time_to_healthy_sec=time_to_healthy,
            failure_classification=failure_class,
            correctness_gate_passed=correctness_gate_passed,
            post_benchmark_correctness=post_smoke if post_smoke else None,
        )
    }


def _aggregate_benchmark(
    results: list[ConcurrencyResult],
    gpu_agg: dict[int, dict],
    kv_metrics: dict,
) -> BenchmarkResult:
    """Aggregate per-phase results into a single BenchmarkResult.

    Workload-aware aggregation:
    - peak_throughput: max from agent_short + throughput workloads (not stress/long_context)
    - low_concurrency_ttft_p95: median of p95 TTFTs from c=1 agent_short phases
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
    else:
        # Fallback: use any c=1 results
        any_low = [r for r in results if r.concurrency == 1]
        low_ttft_p95 = statistics.median([r.ttft_ms.p95 for r in any_low]) if any_low else 0.0
        low_tpot_p95 = statistics.median([r.tpot_ms.p95 for r in any_low]) if any_low else 0.0

    # GPU metrics
    gpu_util = [gpu_agg[i]["util_avg"] for i in sorted(gpu_agg)]
    gpu_mem = [gpu_agg[i]["mem_peak"] for i in sorted(gpu_agg)]
    gpu_power = [gpu_agg[i]["power_avg"] for i in sorted(gpu_agg)]
    gpu_temp = [gpu_agg[i]["temp_max"] for i in sorted(gpu_agg)]

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
        kv_cache_usage_percent=kv_metrics.get("kv_cache_usage_percent", 0.0),
        prefix_cache_hit_rate=kv_metrics.get("prefix_cache_hit_rate", 0.0),
        gpu_utilization_percent=gpu_util,
        gpu_memory_used_mb=gpu_mem,
        gpu_power_draw_watts=gpu_power,
        gpu_temperature_celsius=gpu_temp,
        concurrency_results=results,
    )
