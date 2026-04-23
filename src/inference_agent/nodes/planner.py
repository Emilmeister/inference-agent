"""Planner node — LLM-driven experiment configuration selection."""

from __future__ import annotations

import json
import logging

from langchain_openai import ChatOpenAI

from inference_agent.models import (
    EngineType,
    ExperimentConfig,
    ExperimentSummary,
    HardwareProfile,
    OptimizationGoal,
    PlannerOutput,
)
from inference_agent.state import AgentState

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """\
You are an expert LLM inference optimization engineer. Your job is to choose \
the next configuration to benchmark for an LLM inference engine (vLLM or SGLang) \
running on a GPU server.

## Hardware Profile
{hardware_json}

## Available Engines
{engines}

## Optimization Goal for This Experiment
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
1. CRITICAL: For the FIRST 2 experiments, run BASELINES for BOTH engines: \
experiment #1 = vLLM baseline (TP=1, default params, no speculative), \
experiment #2 = SGLang baseline (TP=1, default params, no speculative). \
Experiments #3-5: try small variations (chunked prefill, prefix caching, different max_model_len). \
Only AFTER baselines, start exploring speculative decoding, quantization, etc.
2. After baselines, analyze trends and try improvements based on the optimization goal.
3. If goal is "optimize_throughput": focus on batching, DP, quantization, high concurrency.
4. If goal is "optimize_latency": focus on TP, enforce_eager, lower batch sizes.
5. If goal is "optimize_balanced": find configs where TTFT p95 < {latency_threshold} ms \
AND throughput is maximized.
6. If goal is "explore": try something new — different engine, quantization, speculative decoding.
7. Never repeat an exact configuration that was already tested.
8. max_model_len must not exceed {model_max_context}.
9. tensor_parallel_size must divide evenly into {gpu_count} GPUs.
10. ALTERNATE between engines: don't run more than 3 experiments in a row with the same engine.

## Best Practices
- vLLM: chunked prefill + prefix caching works well together for throughput.
- SGLang: radix cache (prefix caching) is ON by default. lpm schedule policy \
benefits from prefix caching.
- fp8 quantization usually gives ~1.5-2x throughput with minimal quality loss.
- Higher gpu_memory_utilization (0.95) allows more KV cache but risks OOM.
- data_parallel_size > 1 is great for throughput when model fits in fewer GPUs.
- enforce_eager=true in vLLM can reduce latency for small batches.
- num_continuous_decode_steps > 1 in SGLang reduces scheduling overhead.
- If is_vlm=true: for vLLM text-only benchmarks, the model may need special handling. \
  Do NOT set speculative_algorithm for vLLM with VLM models unless you have a valid draft model path.
- If has_mtp=true: SGLang can use NEXTN speculative decoding with the model's built-in \
  MTP layers. Set speculative_algorithm="NEXTN" and speculative_num_steps=3. No draft model needed. \
  IMPORTANT FOR NEXTN: The engine auto-handles the following — do NOT set them manually or in extra_engine_args: \
    * --mamba-scheduler-strategy (auto-set based on radix cache setting) \
    * --speculative-eagle-topk (auto-set to 1) \
    * SGLANG_ENABLE_SPEC_V2 env var (auto-set) \
  Just set speculative_algorithm="NEXTN" and speculative_num_steps=3, and enable_prefix_caching=true. \
  If NEXTN OOMs: reduce max_model_len or increase mem_fraction_static to 0.85-0.9.
- For speculative decoding on vLLM: you MUST provide a valid speculative_draft_model path \
  (a real HuggingFace model ID). If you don't have a draft model, do NOT set speculative_algorithm.
- For the FIRST few experiments, do NOT use speculative decoding — get baselines first.
- IMPORTANT — max_model_len strategy: \
  max_model_len is a KEY parameter to benchmark. It controls how much KV cache is allocated \
  and directly affects throughput, latency, and max concurrent requests. \
  Use binary-search logic: start with a moderate value that fits in VRAM, then explore up/down. \
  Rough heuristic: each GPU can hold ~2K-4K context per GB of VRAM for a typical 7-10B model \
  (bf16). So 40GB ≈ 80K-160K max, but with model weights loaded it's less. \
  For the FIRST experiment, start with a safe value (e.g. 32768 for 40GB, 8192 for 24GB). \
  If it succeeds, try doubling. If it OOMs, halve. \
  NEVER set max_model_len to the model's full max_position_embeddings (e.g. 262144) unless \
  you have enough VRAM — it WILL OOM. NEVER set it to null — always set an explicit value.
- If a previous experiment FAILED (status="failed"), read the error carefully. Common fixes: \
  halve max_model_len, reduce gpu_memory_utilization/mem_fraction_static, remove kv_cache_dtype=fp8 \
  (not all models support it), try the OTHER engine. Do NOT repeat the same config that failed.
- extra_engine_args: Use ONLY for engine flags that have no dedicated field in this schema. \
  Do NOT use it for flags the engine manages automatically (speculative params for NEXTN, etc.).
{user_instructions}
Generate the next experiment configuration.
"""


async def planner_node(state: AgentState) -> dict:
    """Use LLM to select the next experiment configuration."""
    config = state["config"]
    hardware = state["hardware"]
    history = state.get("experiment_history", [])
    goal = state.get("next_optimization_goal", OptimizationGoal.EXPLORE)

    llm = ChatOpenAI(
        base_url=config.agent_llm.base_url,
        api_key=config.agent_llm.api_key,
        model=config.agent_llm.model,
        temperature=0.3,
    )

    # Use structured output
    structured_llm = llm.with_structured_output(PlannerOutput)

    # Format history for LLM (last 15 experiments)
    history_for_llm = []
    for h in history[-15:]:
        entry: dict = {
            "id": h.experiment_id,
            "engine": h.engine.value,
            "status": h.status.value,
            "config": h.config_digest,
            "peak_throughput": h.peak_throughput,
            "ttft_p95": h.low_concurrency_ttft_p95,
            "tpot_p95": h.low_concurrency_tpot_p95,
            "smoke_pass": f"{h.smoke_tests_passed}/{h.smoke_tests_total}",
            "classification": h.optimization_classification.value,
        }
        # Include error details for failed experiments so LLM can learn from failures
        if h.error:
            entry["error"] = h.error
        history_for_llm.append(entry)

    # Add user instructions section if provided
    user_instructions = ""
    if config.planner_instructions:
        user_instructions = f"\n## User Instructions\n{config.planner_instructions}\n"

    prompt = PLANNER_SYSTEM_PROMPT.format(
        hardware_json=hardware.model_dump_json(indent=2),
        engines=", ".join(e.value for e in hardware.available_engines),
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
        user_instructions=user_instructions,
    )

    logger.info("Asking LLM to plan experiment #%d...", state.get("experiments_count", 0) + 1)

    try:
        result: PlannerOutput = await structured_llm.ainvoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Generate the next experiment configuration."},
        ])
    except Exception as e:
        logger.error("Structured output failed, using fallback: %s", e)
        # Safe fallback: ~4K context per GB of VRAM per GPU
        safe_ctx = min(hardware.gpus[0].vram_total_mb // 10 if hardware.gpus else 32768, hardware.model_max_context)
        result = PlannerOutput(
            engine=hardware.available_engines[0].value if hardware.available_engines else "vllm",
            tensor_parallel_size=hardware.gpu_count,
            max_model_len=safe_ctx,
            rationale=f"Fallback: structured output failed ({e})",
        )

    # Convert PlannerOutput to ExperimentConfig with validation
    experiment = _build_experiment_config(result, hardware, config)

    logger.info(
        "Planned: %s, TP=%d, ctx=%s, quant=%s, rationale: %s",
        experiment.engine.value,
        experiment.tensor_parallel_size,
        experiment.max_model_len,
        experiment.quantization or "none",
        experiment.rationale[:120],
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
        scheduling_policy=output.scheduling_policy,
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
        extra_env=output.extra_env,
        rationale=output.rationale,
    )
