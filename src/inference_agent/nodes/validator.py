"""Validator node — checks experiment config before expensive container run."""

from __future__ import annotations

import logging

from inference_agent.models import (
    AgentConfig,
    EngineType,
    ExperimentConfig,
    ExperimentError,
    ExperimentResult,
    ExperimentStatus,
    HardwareProfile,
    OptimizationGoal,
)
from inference_agent.state import AgentState

logger = logging.getLogger(__name__)


def validate_experiment(
    experiment: ExperimentConfig,
    hardware: HardwareProfile,
    agent_config: AgentConfig | None = None,
    goal: OptimizationGoal | None = None,
) -> list[str]:
    """Validate experiment config against hardware and engine capabilities.

    When `goal` is OptimizationGoal.AGENTIC, additional gates fire:
      - prefix_caching MUST be enabled (shared-prefix workload reuses KV)
      - oversized max_model_len is rejected when it dwarfs the agentic
        workload budget — agent reserved VRAM that could host more agents.

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
    # We only block configurations that are physically impossible at the
    # engine level (e.g. NEXTN without native MTP heads). Algorithms that the
    # engine supports without a draft model — ngram for vLLM, or self-
    # speculation methods (mtp / eagle3) on models with native MTP heads —
    # are allowed through so the planner can actually probe them.
    if experiment.speculative_algorithm:
        algo = experiment.speculative_algorithm.upper()
        has_native_mtp = hardware.mtp_num_layers > 0
        # Algorithms that don't need a draft model on the right hardware.
        # ngram works on any model (no draft, no MTP). mtp/eagle3 can run
        # without a draft when the model itself ships MTP heads.
        SELF_SPEC_ALGOS = {"MTP", "EAGLE3"}
        is_ngram = algo == "NGRAM"
        is_self_speculation = algo in SELF_SPEC_ALGOS and has_native_mtp

        if experiment.engine == EngineType.VLLM:
            if (
                not experiment.speculative_draft_model
                and not is_ngram
                and not is_self_speculation
            ):
                errors.append(
                    f"vLLM speculative_algorithm={algo} requires speculative_draft_model "
                    "(only ngram, or mtp/eagle3 on a model with native MTP heads, "
                    "can run without one)"
                )
        elif experiment.engine == EngineType.SGLANG:
            if algo == "NEXTN" and not has_native_mtp:
                errors.append(
                    "NEXTN speculative decoding requires a model with native MTP heads "
                    "(mtp_num_layers=0 for this model)"
                )

        # num_steps cap only applies to self-speculation. With an external
        # draft model the draft can predict an arbitrary number of tokens —
        # the engine itself caps it, not us. NEXTN is always self-speculation
        # in SGLang, so it gets the cap regardless of draft model field.
        is_self_spec_run = (
            algo == "NEXTN"
            or (algo in SELF_SPEC_ALGOS and not experiment.speculative_draft_model)
        )
        if (
            experiment.speculative_num_steps is not None
            and is_self_spec_run
            and has_native_mtp
            and experiment.speculative_num_steps > hardware.mtp_num_layers
        ):
            errors.append(
                f"speculative_num_steps={experiment.speculative_num_steps} exceeds "
                f"this model's native MTP head count ({hardware.mtp_num_layers}); "
                f"self-speculation cannot predict more tokens than available heads"
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

    # ── Agentic-goal gates ────────────────────────────────────────────
    # These fire only when the planner is optimizing for max parallel
    # agents. Other goals get the synthetic-throughput knobs back.
    if goal == OptimizationGoal.AGENTIC and agent_config is not None:
        if not experiment.enable_prefix_caching:
            errors.append(
                "enable_prefix_caching must be true under optimize_agentic: "
                "the benchmark shares a prefix across all sessions; without "
                "prefix caching each session pays its own ~16K-token prefill "
                "and concurrency collapses. Set enable_prefix_caching=true."
            )

        # Oversized max_model_len under the agentic goal wastes KV budget on
        # context windows the workload never uses. We reject when the
        # requested ctx is > 2× the realistic agentic budget AND there is a
        # smaller power-of-2 bucket that still fits — the planner can pick a
        # tighter value next round.
        #
        # EXCEPTION: when the operator has fixed max_model_len at the
        # AgentConfig level, this is an explicit invariant — don't override
        # their choice with a heuristic about agentic budget. The whole point
        # of the fixed value is that they want this context window across
        # every experiment regardless of workload shape.
        bench = agent_config.benchmark
        if (
            bench.enable_agentic_long_context
            and experiment.max_model_len is not None
            and agent_config.max_model_len is None
        ):
            agentic_budget = (
                bench.agentic_shared_prefix_tokens
                + bench.agentic_unique_prompt_tokens
                + bench.agentic_turns_per_session
                * (
                    bench.agentic_max_output_tokens
                    + bench.agentic_tool_result_max_tokens
                )
            )
            # Headroom: 1.5× the live budget so the engine isn't packed at the
            # limit; anything above 2× is the LLM picking a 131K context when
            # the workload needs ~40K.
            if experiment.max_model_len > 2 * agentic_budget:
                errors.append(
                    f"max_model_len={experiment.max_model_len} is "
                    f">2× the agentic workload budget ({agentic_budget}); "
                    f"under optimize_agentic this wastes KV that could host "
                    f"more parallel sessions. Pick the smallest power-of-2 "
                    f"≥ {agentic_budget} (typically 32768 or 65536)."
                )

    return errors


async def validator_node(state: AgentState) -> dict:
    """Validate experiment config before running. Fails fast on bad configs."""
    experiment = state["current_config"]
    hardware = state["hardware"]
    config = state["config"]
    goal = state.get("next_optimization_goal")

    errors = validate_experiment(experiment, hardware, config, goal)

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
                failure_classification="validation",
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
