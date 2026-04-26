"""Planner node — LLM-driven experiment configuration selection."""

from __future__ import annotations

import json
import logging

from inference_agent.models import (
    EngineType,
    ExperimentConfig,
    ExperimentSummary,
    HardwareProfile,
    OptimizationGoal,
    PlannerOutput,
)
from inference_agent.state import AgentState
from inference_agent.utils.codex import claude_structured_output

logger = logging.getLogger(__name__)

_PROMPT_HEADER = """\
You are an expert LLM inference optimization engineer. Your job is to choose \
the next configuration to benchmark for an LLM inference engine running on a GPU server.

## Hardware Profile
{hardware_json}

## Engine for This Experiment
{engine_instruction}

## Optimization Goal
{optimization_goal}

## Previous Experiments (most recent last)
{history_json}

## Best Results So Far
- Best throughput: {best_throughput:.1f} tok/s (config: {best_throughput_id})
- Best latency (TTFT p95): {best_latency:.1f} ms (config: {best_latency_id})
- Best balanced: throughput={best_balanced_tp:.1f}, latency={best_balanced_lat:.1f} \
(config: {best_balanced_id})

## Experiment Count
{exp_count} / {max_experiments}

## Rules
1. For BASELINES: use default params — no quantization, no speculative decoding, \
kv_cache_dtype=auto, scheduling_policy=fcfs.
2. After baselines, analyze trends and improve based on the optimization goal.
3. If goal is "optimize_throughput": focus on batching, DP, quantization, high concurrency.
4. If goal is "optimize_latency": focus on TP, enforce_eager, lower batch sizes.
5. If goal is "optimize_balanced": find configs where TTFT p95 < {latency_threshold} ms \
AND throughput is maximized.
6. If goal is "explore": try something new — quantization, speculative decoding, different max_model_len.
7. Never repeat an exact configuration that was already tested.
8. max_model_len must not exceed {model_max_context}. NEVER set it to the full model max unless you \
have enough VRAM. Use binary search: start safe (32768 for 40GB), double if success, halve if OOM.
9. tensor_parallel_size must divide evenly into {gpu_count} GPUs.
10. CRITICAL: If a feature (NEXTN, quantization, etc.) consistently gives WORSE results than baseline, \
STOP using it and move on to other optimizations.
11. If a previous experiment FAILED, read the error, fix the root cause, do NOT repeat the same config.
"""

_VLLM_SECTION = """
## vLLM-Specific Parameters
You are configuring a **vLLM** experiment. Only use vLLM-relevant parameters:
- gpu_memory_utilization (float, default 0.9): GPU memory fraction. Higher = more KV cache but OOM risk.
- max_num_seqs (int): Max concurrent sequences (default 256).
- max_num_batched_tokens (int): Max batched tokens per iteration.
- enable_chunked_prefill (bool): Enable chunked prefill for better throughput.
- enable_prefix_caching (bool): Enable prefix caching.
- enforce_eager (bool): Disable CUDA graphs — reduces latency for small batches.
- scheduling_policy: ONLY "fcfs" (default) or "priority". Do NOT use "lpm".
- speculative_draft_model: Real HuggingFace model ID for speculative decoding. \
  Do NOT set speculative_algorithm without a valid draft model path.

## vLLM Best Practices
- Chunked prefill + prefix caching work well together for throughput.
- enforce_eager=true can reduce latency for small batches but hurts throughput.
- Higher gpu_memory_utilization (0.95) allows more KV cache but risks OOM.
- fp8 quantization usually gives ~1.5-2x throughput with minimal quality loss.

Leave SGLang-specific fields at defaults: mem_fraction_static=null, \
max_running_requests=null, max_prefill_tokens=null, chunked_prefill_size=null, \
num_continuous_decode_steps=1, dp_size=null.
"""

_SGLANG_SECTION = """
## SGLang-Specific Parameters
You are configuring an **SGLang** experiment. Only use SGLang-relevant parameters:
- mem_fraction_static (float): Static memory fraction (default ~0.8). \
  For NEXTN, engine auto-bumps to 0.9.
- max_running_requests (int): Max running requests.
- max_prefill_tokens (int): Max prefill tokens per batch.
- scheduling_policy: "fcfs" (default) or "lpm". lpm benefits from prefix caching.
- chunked_prefill_size (int): Chunked prefill token size. Set to enable chunked prefill.
- enable_prefix_caching (bool): Maps to radix cache (ON by default in SGLang). \
  Set to false only if needed.
- num_continuous_decode_steps (int): Reduce scheduling overhead (default 1, try 2-4).
- dp_size (int): Data parallelism.
- speculative_algorithm: Set to "NEXTN" for MTP speculative decoding (if has_mtp=true). \
  Engine auto-handles mamba-scheduler-strategy, speculative-eagle-topk, env vars — \
  do NOT set them manually. Try SGLang WITHOUT NEXTN first to get a baseline, \
  then compare WITH NEXTN. If NEXTN is slower, STOP using it.
- speculative_num_steps (int): Steps for speculative decoding (3 for NEXTN).

## SGLang Best Practices
- Radix cache (prefix caching) ON by default — lpm scheduling benefits from it.
- num_continuous_decode_steps > 1 reduces scheduling overhead.
- NEXTN is OPTIONAL: compare with and without. If slower, abandon it.
- fp8 quantization usually gives ~1.5-2x throughput with minimal quality loss.

Leave vLLM-specific fields at defaults: gpu_memory_utilization=0.9, \
max_num_seqs=null, max_num_batched_tokens=null, enforce_eager=false.
"""

_BOTH_ENGINES_SECTION = """
## Available Engines: vLLM and SGLang
Choose ONE engine and set only its relevant parameters.

### vLLM parameters
- gpu_memory_utilization, max_num_seqs, max_num_batched_tokens, \
enable_chunked_prefill, enable_prefix_caching, enforce_eager
- scheduling_policy: ONLY "fcfs" or "priority"
- speculative_draft_model: needs a real HuggingFace model path

### SGLang parameters
- mem_fraction_static, max_running_requests, max_prefill_tokens, chunked_prefill_size, \
num_continuous_decode_steps, dp_size, enable_prefix_caching (radix cache)
- scheduling_policy: "fcfs" or "lpm"
- speculative_algorithm: "NEXTN" for MTP (optional, try without first)

Set ONLY the parameters for your chosen engine. Leave the other engine's fields at defaults.
"""

_PROMPT_FOOTER = """
## extra_engine_args
Use ONLY for engine CLI flags that have no dedicated field in this schema. \
Do NOT duplicate flags the engine manages automatically.
{user_instructions}
Generate the next experiment configuration.
"""


def _aggregate_failure_patterns(history: list[ExperimentSummary]) -> str:
    """Aggregate failure patterns from history into a concise summary for the LLM.

    Groups failures by classification and engine, showing the planner
    what config classes to avoid.
    """
    if not history:
        return ""

    failed = [h for h in history if h.status.value in ("failed", "failed_correctness", "partial")]
    if not failed:
        return ""

    # Group by (failure_classification, engine)
    pattern_counts: dict[tuple[str, str], int] = {}
    pattern_examples: dict[tuple[str, str], str] = {}
    for h in failed:
        fc = h.failure_classification or "unknown"
        key = (fc, h.engine.value)
        pattern_counts[key] = pattern_counts.get(key, 0) + 1
        if key not in pattern_examples and h.error:
            pattern_examples[key] = h.error[:150]

    # Also count correctness failures
    correctness_failed = [h for h in history if h.status.value == "failed_correctness"]
    if correctness_failed:
        engines_failed_correctness = set(h.engine.value for h in correctness_failed)
        for eng in engines_failed_correctness:
            count = sum(1 for h in correctness_failed if h.engine.value == eng)
            key = ("correctness_failure", eng)
            if key not in pattern_counts:
                pattern_counts[key] = count

    lines = []
    for (fc, eng), count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        example = pattern_examples.get((fc, eng), "")
        line = f"- {eng}: {fc} ({count}x)"
        if example:
            line += f" — e.g. {example}"
        lines.append(line)

    return "\n".join(lines)


def _estimate_safe_context(hardware: HardwareProfile) -> int:
    """Estimate a safe max_model_len for fallback when LLM planner is unavailable.

    Uses a rough heuristic: total VRAM minus estimated model weight, then
    convert remaining VRAM to context length. Conservative but much better
    than vram_total // 10.
    """
    if not hardware.gpus:
        return min(32768, hardware.model_max_context)

    total_vram_mb = sum(g.vram_total_mb for g in hardware.gpus)

    # Estimate model weight in MB (params * 2 bytes for fp16)
    if hardware.model_size_params:
        model_weight_mb = hardware.model_size_params * 2 / (1024 * 1024)
    else:
        model_weight_mb = total_vram_mb * 0.5  # assume half VRAM for model

    # Available VRAM for KV cache (leave 10% headroom)
    available_mb = (total_vram_mb - model_weight_mb) * 0.9

    # Rough estimate: ~2 MB per 1K context for a typical model
    # This is very approximate but much better than vram // 10
    estimated_ctx = int(available_mb / 2 * 1024)

    # Clamp: cap fallback at 65536 (conservative — LLM planner can go higher)
    safe_ctx = max(4096, min(estimated_ctx, 65536, hardware.model_max_context))

    # Round down to nearest 4096 for cleanliness
    safe_ctx = (safe_ctx // 4096) * 4096

    return max(safe_ctx, 4096)


def _get_forced_engine(
    history: list[ExperimentSummary],
    available: list[EngineType],
) -> EngineType | None:
    """Force engine switch if last 2 experiments used the same engine."""
    if len(available) < 2 or len(history) < 2:
        return None
    last_engines = [h.engine for h in history[-2:]]
    if len(set(last_engines)) == 1:
        current = last_engines[0]
        other = [e for e in available if e != current]
        if other:
            return other[0]
    return None


def _should_disable_speculative(
    history: list[ExperimentSummary],
    engine: EngineType,
    threshold: int = 3,
) -> bool:
    """Disable speculative decoding if last N experiments with this engine all failed."""
    engine_history = [h for h in history if h.engine == engine]
    if len(engine_history) < threshold:
        return False
    recent = engine_history[-threshold:]
    return all(h.status.value == "failed" for h in recent)


async def planner_node(state: AgentState) -> dict:
    """Use LLM to select the next experiment configuration."""
    config = state["config"]
    hardware = state["hardware"]
    history = state.get("experiment_history", [])
    goal = state.get("next_optimization_goal", OptimizationGoal.EXPLORE)

    # Format history for LLM (last 15 experiments)
    history_for_llm = []
    for h in history[-15:]:
        entry: dict = {
            "id": h.experiment_id,
            "engine": h.engine.value,
            "status": h.status.value,
            "config": h.config_digest,
            "docker_command": h.docker_command,
            "rationale": h.rationale,
            "peak_throughput": h.peak_throughput,
            "ttft_p95": h.low_concurrency_ttft_p95,
            "tpot_p95": h.low_concurrency_tpot_p95,
            "smoke_pass": f"{h.smoke_tests_passed}/{h.smoke_tests_total}",
            "correctness_gate": h.correctness_gate_passed,
            "classification": h.optimization_classification.value,
        }
        # Include error details for failed experiments so LLM can learn from failures
        if h.error:
            entry["error"] = h.error
        if h.failure_classification:
            entry["failure_type"] = h.failure_classification
        # Include LLM commentary from analyzer for context
        if h.llm_commentary:
            entry["analysis"] = h.llm_commentary
        history_for_llm.append(entry)

    # Determine if we need to force a specific engine (alternation rule)
    forced_engine = _get_forced_engine(history, hardware.available_engines)
    if forced_engine:
        logger.info("Forcing engine=%s (alternation rule)", forced_engine.value)

    # Check if speculative decoding should be disabled for each engine
    spec_disabled: dict[EngineType, bool] = {}
    for eng in hardware.available_engines:
        spec_disabled[eng] = _should_disable_speculative(history, eng)
        if spec_disabled[eng]:
            logger.info(
                "Auto-disabling speculative decoding for %s (last 3 experiments failed)",
                eng.value,
            )

    # Build engine-specific prompt section
    if forced_engine == EngineType.VLLM:
        engine_section = _VLLM_SECTION
        engine_instruction = "You MUST use **vLLM** for this experiment."
    elif forced_engine == EngineType.SGLANG:
        engine_section = _SGLANG_SECTION
        engine_instruction = "You MUST use **SGLang** for this experiment."
    else:
        engine_section = _BOTH_ENGINES_SECTION
        engine_instruction = f"Choose from: {', '.join(e.value for e in hardware.available_engines)}"

    # Aggregate failure patterns for the LLM
    failure_patterns = _aggregate_failure_patterns(history)

    user_instructions = ""
    if config.planner_instructions:
        user_instructions = f"\n## User Instructions\n{config.planner_instructions}\n"

    prompt = _PROMPT_HEADER.format(
        hardware_json=hardware.model_dump_json(indent=2),
        engine_instruction=engine_instruction,
        optimization_goal=goal.value,
        history_json=json.dumps(history_for_llm, indent=2) if history_for_llm else "No experiments yet.",
        best_throughput=state.get("best_throughput", 0),
        best_throughput_id=state.get("best_throughput_config_id", "none"),
        best_latency=state.get("best_latency_ttft_p95", float("inf")),
        best_latency_id=state.get("best_latency_config_id", "none"),
        best_balanced_tp=state.get("best_balanced_throughput", 0),
        best_balanced_lat=state.get("best_balanced_latency", float("inf")),
        best_balanced_id=state.get("best_balanced_config_id", "none"),
        exp_count=state.get("experiments_count", 0),
        max_experiments=config.experiments.max_experiments,
        latency_threshold=config.benchmark.latency_threshold_ms,
        model_max_context=hardware.model_max_context,
        gpu_count=hardware.gpu_count,
    ) + engine_section

    if failure_patterns:
        prompt += f"\n## Known Failure Patterns\n{failure_patterns}\n"

    prompt += _PROMPT_FOOTER.format(user_instructions=user_instructions)

    logger.info("Asking LLM to plan experiment #%d...", state.get("experiments_count", 0) + 1)

    full_prompt = prompt + "\n\nGenerate the next experiment configuration."

    try:
        result: PlannerOutput = await claude_structured_output(full_prompt, PlannerOutput)
    except Exception as e:
        error_summary = str(e).split("\n")[0][:200]
        logger.error("Claude structured output failed, using fallback: %s", error_summary)
        safe_ctx = _estimate_safe_context(hardware)
        result = PlannerOutput(
            engine=hardware.available_engines[0].value if hardware.available_engines else "vllm",
            tensor_parallel_size=hardware.gpu_count,
            max_model_len=safe_ctx,
            rationale=f"Fallback: claude failed — {error_summary}",
        )

    # Override engine if alternation rule requires it
    if forced_engine:
        result.engine = forced_engine.value

    # Convert PlannerOutput to ExperimentConfig with validation
    resolved_engine = EngineType.VLLM if "vllm" in result.engine.lower() else EngineType.SGLANG
    if spec_disabled.get(resolved_engine, False):
        if result.speculative_algorithm:
            logger.info(
                "Stripping speculative_algorithm=%s (auto-disabled for %s)",
                result.speculative_algorithm, resolved_engine.value,
            )
            result.speculative_algorithm = None
            result.speculative_draft_model = None
            result.speculative_num_steps = None
    experiment = _build_experiment_config(result, hardware, config)

    logger.info(
        "Planned: %s, TP=%d, ctx=%s, quant=%s, rationale: %s",
        experiment.engine.value,
        experiment.tensor_parallel_size,
        experiment.max_model_len,
        experiment.quantization or "none",
        experiment.rationale,
    )

    return {"current_config": experiment}


def _build_experiment_config(
    output: PlannerOutput,
    hardware: HardwareProfile,
    config: object,
) -> ExperimentConfig:
    """Convert validated PlannerOutput to ExperimentConfig with safety checks."""
    # Normalize engine
    engine_str = output.engine.lower()
    engine = EngineType.VLLM if "vllm" in engine_str else EngineType.SGLANG

    # Validate TP size
    tp = output.tensor_parallel_size
    if tp > hardware.gpu_count:
        tp = hardware.gpu_count
    if hardware.gpu_count > 0 and hardware.gpu_count % tp != 0:
        valid_tps = [i for i in range(1, hardware.gpu_count + 1) if hardware.gpu_count % i == 0]
        tp = min(valid_tps, key=lambda x: abs(x - tp))

    # Validate max_model_len — must be set, capped to model max
    max_model_len = output.max_model_len
    if max_model_len > hardware.model_max_context:
        max_model_len = hardware.model_max_context
    # Sanity: at least 512
    max_model_len = max(max_model_len, 512)

    # Validate scheduling_policy per engine
    scheduling_policy = output.scheduling_policy
    if engine == EngineType.VLLM and scheduling_policy not in ("fcfs", "priority"):
        scheduling_policy = "fcfs"
    elif engine == EngineType.SGLANG and scheduling_policy not in ("fcfs", "lpm"):
        scheduling_policy = "fcfs"

    # Strip cross-engine env vars
    extra_env = dict(output.extra_env)
    if engine == EngineType.VLLM:
        # Remove SGLang-specific env vars
        for k in list(extra_env.keys()):
            if k.startswith("SGLANG"):
                extra_env.pop(k)

    return ExperimentConfig(
        engine=engine,
        tensor_parallel_size=tp,
        pipeline_parallel_size=output.pipeline_parallel_size,
        data_parallel_size=output.data_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=output.gpu_memory_utilization,
        mem_fraction_static=output.mem_fraction_static,
        max_num_seqs=output.max_num_seqs,
        max_running_requests=output.max_running_requests,
        max_num_batched_tokens=output.max_num_batched_tokens,
        max_prefill_tokens=output.max_prefill_tokens,
        scheduling_policy=scheduling_policy,
        quantization=output.quantization,
        dtype=output.dtype,
        kv_cache_dtype=output.kv_cache_dtype,
        enable_chunked_prefill=output.enable_chunked_prefill,
        chunked_prefill_size=output.chunked_prefill_size,
        enable_prefix_caching=output.enable_prefix_caching,
        enforce_eager=output.enforce_eager,
        speculative_algorithm=output.speculative_algorithm,
        speculative_draft_model=output.speculative_draft_model,
        speculative_num_steps=output.speculative_num_steps,
        num_continuous_decode_steps=output.num_continuous_decode_steps,
        dp_size=output.dp_size,
        extra_engine_args=output.extra_engine_args,
        extra_env=extra_env,
        rationale=output.rationale,
    )
