"""Executor node — launches Docker container and runs benchmark."""

from __future__ import annotations

import logging
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
    get_container_logs,
    get_image_digest,
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
) -> tuple[str | None, list[ExperimentError]]:
    """Start Docker container and wait for health check.

    Returns (container_id, errors). If container_id is None, startup failed.
    """
    errors: list[ExperimentError] = []

    # Stop any previous container
    await stop_container(container_name)

    # Run container
    try:
        container_id = await run_container(docker_args, timeout=60)
        logger.info("Container started: %s", container_id[:12])
    except RuntimeError as e:
        logs = await get_container_logs(container_name)
        errors.append(ExperimentError(
            stage="startup",
            message=f"Container start failed: {e}",
            details={"logs": logs[:5000]},
        ))
        return None, errors

    # Wait for health check
    logger.info("Waiting for engine health check...")
    healthy = await wait_for_healthy(
        engine.health_url(), timeout_sec=900, container_name=container_name,
    )
    if not healthy:
        logs = await get_container_logs(container_name)
        errors.append(ExperimentError(
            stage="healthcheck",
            message="Engine did not become healthy within 900s",
            details={"logs": logs[:5000]},
        ))
        logger.error("FAILED: Engine did not become healthy. Last logs:\n%s", logs[-500:])
        await stop_container(container_name)
        return None, errors

    logger.info("Engine is healthy!")
    return container_id, errors


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
    )

    for phase_name, concurrency, prompt_len, max_out in phases:
        is_warmup = phase_name == "warmup"
        duration = 10 if is_warmup else config.benchmark.duration_per_level_sec

        logger.info(
            "  Phase: %s (c=%d, prompt=%d, max_out=%d, dur=%ds)",
            phase_name, concurrency, prompt_len, max_out, duration,
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
            )
            if not is_warmup:
                concurrency_results.append(result)
        except Exception as e:
            logger.error("  Phase %s failed: %s", phase_name, e)
            phase_errors.append(ExperimentError(
                stage="benchmark_phase",
                message=str(e),
                details={
                    "phase": phase_name,
                    "concurrency": concurrency,
                    "prompt_length": prompt_len,
                },
            ))

    return concurrency_results, phase_errors


async def executor_node(state: AgentState) -> dict:
    """Launch the inference engine in Docker and run benchmarks."""
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

    # Start engine container
    container_id, startup_errors = await _start_engine(
        engine, docker_args, container_name, experiment.experiment_id,
    )

    if container_id is None:
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
                docker_image_digest=image_digest,
                benchmark_seed=seed,
                duration_seconds=time.time() - start_time,
            )
        }

    # Start GPU monitoring
    gpu_monitor = GPUMonitor(interval_ms=1000)
    await gpu_monitor.start()

    # Run benchmark phases
    logger.info("Starting benchmark...")
    concurrency_results, phase_errors = await _run_all_phases(
        engine, config, hardware, experiment, seed,
    )

    # Collect Prometheus metrics
    prom_metrics = await fetch_prometheus_metrics(engine.metrics_url())
    kv_metrics = extract_kv_cache_metrics(prom_metrics, experiment.engine.value)

    # Run smoke tests
    logger.info("Running smoke tests...")
    try:
        smoke_results = await run_smoke_tests(engine.api_base_url(), config.model_name)
    except Exception as e:
        logger.error("Smoke tests failed: %s", e)
        smoke_results = SmokeTestResult()
        phase_errors.append(ExperimentError(
            stage="smoke",
            message=str(e),
        ))

    # Stop GPU monitoring and container
    gpu_snapshots = await gpu_monitor.stop()
    gpu_agg = GPUMonitor.aggregate_snapshots(gpu_snapshots)
    await stop_container(container_name)

    # Aggregate results
    benchmark = _aggregate_benchmark(concurrency_results, gpu_agg, kv_metrics)

    duration = time.time() - start_time
    all_errors = startup_errors + phase_errors

    # Determine status based on results and errors
    if not concurrency_results:
        status = ExperimentStatus.PARTIAL
    elif phase_errors:
        status = ExperimentStatus.PARTIAL
    else:
        status = ExperimentStatus.SUCCESS

    logger.info(
        "Experiment %s complete: peak_throughput=%.1f tok/s, "
        "low_ttft_p95=%.1f ms, duration=%.0fs, phase_errors=%d",
        experiment.experiment_id,
        benchmark.peak_output_tokens_per_sec,
        benchmark.low_concurrency_ttft_p95_ms,
        duration,
        len(phase_errors),
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
            docker_image_digest=image_digest,
            benchmark_seed=seed,
            duration_seconds=duration,
        )
    }


def _aggregate_benchmark(
    results: list[ConcurrencyResult],
    gpu_agg: dict[int, dict],
    kv_metrics: dict,
) -> BenchmarkResult:
    """Aggregate per-phase results into a single BenchmarkResult."""
    if not results:
        return BenchmarkResult()

    # Find peak throughput
    peak_throughput_result = max(results, key=lambda r: r.output_tokens_per_sec)

    # Find low-concurrency results (concurrency=1)
    low_conc = [r for r in results if r.concurrency == 1]
    low_ttft_p95 = min((r.ttft_ms.p95 for r in low_conc), default=0.0)
    low_tpot_p95 = min((r.tpot_ms.p95 for r in low_conc), default=0.0)

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
