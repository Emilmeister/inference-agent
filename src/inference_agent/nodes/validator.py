"""Validator node — checks experiment config before expensive Docker run."""

from __future__ import annotations

import logging

from inference_agent.models import (
    EngineType,
    ExperimentConfig,
    ExperimentError,
    ExperimentResult,
    ExperimentStatus,
    HardwareProfile,
)
from inference_agent.state import AgentState

logger = logging.getLogger(__name__)


def validate_experiment(
    experiment: ExperimentConfig,
    hardware: HardwareProfile,
) -> list[str]:
    """Validate experiment config against hardware and engine capabilities.

    Returns a list of error messages. Empty list means valid.
    """
    errors: list[str] = []

    # ── Parallelism checks ────────────────────────────────────────────
    tp = experiment.tensor_parallel_size
    if hardware.gpu_count > 0 and tp > hardware.gpu_count:
        errors.append(
            f"tensor_parallel_size={tp} exceeds gpu_count={hardware.gpu_count}"
        )
    if hardware.gpu_count > 0 and hardware.gpu_count % tp != 0:
        errors.append(
            f"tensor_parallel_size={tp} does not divide evenly into "
            f"gpu_count={hardware.gpu_count}"
        )

    pp = experiment.pipeline_parallel_size
    dp = experiment.data_parallel_size
    total_required = tp * pp * dp
    if hardware.gpu_count > 0 and total_required > hardware.gpu_count:
        errors.append(
            f"TP*PP*DP={total_required} exceeds gpu_count={hardware.gpu_count}"
        )

    # ── Context length checks ─────────────────────────────────────────
    if experiment.max_model_len is not None:
        if experiment.max_model_len > hardware.model_max_context:
            errors.append(
                f"max_model_len={experiment.max_model_len} exceeds "
                f"model_max_context={hardware.model_max_context}"
            )
        if experiment.max_model_len < 512:
            errors.append(
                f"max_model_len={experiment.max_model_len} is too small (min 512)"
            )

    # ── Engine-specific scheduling policy ─────────────────────────────
    sp = experiment.scheduling_policy
    if experiment.engine == EngineType.VLLM and sp not in ("fcfs", "priority"):
        errors.append(
            f"vLLM does not support scheduling_policy='{sp}'. "
            f"Use 'fcfs' or 'priority'."
        )
    if experiment.engine == EngineType.SGLANG and sp not in ("fcfs", "lpm"):
        errors.append(
            f"SGLang does not support scheduling_policy='{sp}'. "
            f"Use 'fcfs' or 'lpm'."
        )

    # ── Cross-engine parameter checks ─────────────────────────────────
    if experiment.engine == EngineType.VLLM:
        if experiment.mem_fraction_static is not None:
            errors.append("mem_fraction_static is SGLang-only, not applicable to vLLM")
        if experiment.max_running_requests is not None:
            errors.append("max_running_requests is SGLang-only, not applicable to vLLM")
        if experiment.dp_size is not None and experiment.dp_size > 1:
            errors.append("dp_size is SGLang-only, use data_parallel_size for vLLM")
    elif experiment.engine == EngineType.SGLANG:
        if experiment.max_num_seqs is not None:
            errors.append("max_num_seqs is vLLM-only, not applicable to SGLang")
        if experiment.max_num_batched_tokens is not None:
            errors.append("max_num_batched_tokens is vLLM-only, not applicable to SGLang")

    # ── Speculative decoding checks ───────────────────────────────────
    if experiment.speculative_algorithm:
        algo = experiment.speculative_algorithm.upper()
        if experiment.engine == EngineType.VLLM:
            if not experiment.speculative_draft_model:
                errors.append(
                    "vLLM speculative decoding requires speculative_draft_model"
                )
        elif experiment.engine == EngineType.SGLANG:
            if algo == "NEXTN" and not hardware.has_mtp:
                errors.append(
                    "NEXTN speculative decoding requires a model with MTP layers "
                    "(has_mtp=false for this model)"
                )

    # ── Memory utilization bounds ─────────────────────────────────────
    if experiment.gpu_memory_utilization <= 0 or experiment.gpu_memory_utilization > 1.0:
        errors.append(
            f"gpu_memory_utilization={experiment.gpu_memory_utilization} must be in (0, 1.0]"
        )
    if experiment.mem_fraction_static is not None:
        if experiment.mem_fraction_static <= 0 or experiment.mem_fraction_static > 1.0:
            errors.append(
                f"mem_fraction_static={experiment.mem_fraction_static} must be in (0, 1.0]"
            )

    # ── Engine availability check ─────────────────────────────────────
    if experiment.engine not in hardware.available_engines:
        errors.append(
            f"Engine '{experiment.engine.value}' not in available_engines: "
            f"{[e.value for e in hardware.available_engines]}"
        )

    return errors


async def validator_node(state: AgentState) -> dict:
    """Validate experiment config before running. Fails fast on bad configs."""
    experiment = state["current_config"]
    hardware = state["hardware"]
    config = state["config"]

    errors = validate_experiment(experiment, hardware)

    if errors:
        error_msg = "Validation failed: " + "; ".join(errors)
        logger.error(
            "Experiment %s failed validation: %s",
            experiment.experiment_id,
            error_msg,
        )
        return {
            "current_result": ExperimentResult(
                experiment_id=experiment.experiment_id,
                engine=experiment.engine,
                model=config.model_name,
                hardware=hardware,
                config=experiment,
                status=ExperimentStatus.FAILED,
                error=error_msg,
                errors=[
                    ExperimentError(
                        stage="validation",
                        message=err,
                    )
                    for err in errors
                ],
            ),
            # Skip executor, go straight to analyzer
            "skip_executor": True,
        }

    logger.info("Experiment %s passed validation", experiment.experiment_id)
    return {"skip_executor": False}
