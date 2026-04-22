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
1. For the FIRST 3-5 experiments, try baseline configs: default params with TP=1, \
TP=max_gpus for both vllm and sglang.
2. After baselines, analyze trends and try improvements based on the optimization goal.
3. If goal is "optimize_throughput": focus on batching, DP, quantization, high concurrency.
4. If goal is "optimize_latency": focus on TP, enforce_eager, lower batch sizes.
5. If goal is "optimize_balanced": find configs where TTFT p95 < {latency_threshold} ms \
AND throughput is maximized.
6. If goal is "explore": try something new — different engine, quantization, speculative decoding.
7. Never repeat an exact configuration that was already tested.
8. max_model_len must not exceed {model_max_context}.
9. tensor_parallel_size must divide evenly into {gpu_count} GPUs.

## Best Practices
- vLLM: chunked prefill + prefix caching works well together for throughput.
- SGLang: radix cache (prefix caching) is ON by default. lpm schedule policy \
benefits from prefix caching.
- fp8 quantization usually gives ~1.5-2x throughput with minimal quality loss.
- Higher gpu_memory_utilization (0.95) allows more KV cache but risks OOM.
- data_parallel_size > 1 is great for throughput when model fits in fewer GPUs.
- enforce_eager=true in vLLM can reduce latency for small batches.
- num_continuous_decode_steps > 1 in SGLang reduces scheduling overhead.

Respond with a JSON object matching the ExperimentConfig schema. Include a "rationale" \
field explaining your choice. No markdown, just JSON.
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

    # Format history for LLM (last 15 experiments)
    history_for_llm = []
    for h in history[-15:]:
        history_for_llm.append({
            "id": h.experiment_id,
            "engine": h.engine.value,
            "status": h.status.value,
            "config": h.config_digest,
            "peak_throughput": h.peak_throughput,
            "ttft_p95": h.low_concurrency_ttft_p95,
            "tpot_p95": h.low_concurrency_tpot_p95,
            "smoke_pass": f"{h.smoke_tests_passed}/{h.smoke_tests_total}",
            "classification": h.optimization_classification.value,
        })

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
    )

    logger.info("Asking LLM to plan experiment #%d...", state.get("experiments_count", 0) + 1)

    response = await llm.ainvoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Generate the next experiment configuration as JSON."},
    ])

    # Parse LLM response
    content = response.content.strip()
    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
        content = content.strip()

    try:
        raw = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error("LLM returned invalid JSON: %s\nContent: %s", e, content[:500])
        # Fallback: baseline config
        raw = _fallback_config(hardware, state)

    # Build ExperimentConfig from LLM output
    experiment = _parse_experiment_config(raw, hardware)

    logger.info(
        "Planned: %s, TP=%d, quant=%s, rationale: %s",
        experiment.engine.value,
        experiment.tensor_parallel_size,
        experiment.quantization or "none",
        experiment.rationale[:100],
    )

    return {"current_config": experiment}


def _fallback_config(hardware: HardwareProfile, state: AgentState) -> dict:
    """Generate a safe fallback configuration."""
    engine = hardware.available_engines[0] if hardware.available_engines else EngineType.VLLM
    return {
        "engine": engine.value,
        "tensor_parallel_size": hardware.gpu_count,
        "dtype": "auto",
        "rationale": "Fallback: default config with all GPUs for tensor parallelism",
    }


def _parse_experiment_config(raw: dict, hardware: HardwareProfile) -> ExperimentConfig:
    """Parse LLM output into a validated ExperimentConfig."""
    # Normalize engine
    engine_str = raw.get("engine", "vllm")
    if isinstance(engine_str, str):
        engine_str = engine_str.lower()
    engine = EngineType.VLLM if "vllm" in str(engine_str) else EngineType.SGLANG

    # Validate TP size
    tp = raw.get("tensor_parallel_size", 1)
    if tp > hardware.gpu_count:
        tp = hardware.gpu_count
    if hardware.gpu_count % tp != 0:
        # Find nearest valid TP
        valid_tps = [i for i in range(1, hardware.gpu_count + 1) if hardware.gpu_count % i == 0]
        tp = min(valid_tps, key=lambda x: abs(x - tp))

    # Validate max_model_len
    max_model_len = raw.get("max_model_len")
    if max_model_len is not None and max_model_len > hardware.model_max_context:
        max_model_len = hardware.model_max_context

    return ExperimentConfig(
        engine=engine,
        tensor_parallel_size=tp,
        pipeline_parallel_size=raw.get("pipeline_parallel_size", 1),
        data_parallel_size=raw.get("data_parallel_size", 1),
        max_model_len=max_model_len,
        gpu_memory_utilization=raw.get("gpu_memory_utilization", 0.90),
        mem_fraction_static=raw.get("mem_fraction_static"),
        max_num_seqs=raw.get("max_num_seqs"),
        max_running_requests=raw.get("max_running_requests"),
        max_num_batched_tokens=raw.get("max_num_batched_tokens"),
        max_prefill_tokens=raw.get("max_prefill_tokens"),
        scheduling_policy=raw.get("scheduling_policy", "fcfs"),
        quantization=raw.get("quantization"),
        dtype=raw.get("dtype", "auto"),
        kv_cache_dtype=raw.get("kv_cache_dtype", "auto"),
        enable_chunked_prefill=raw.get("enable_chunked_prefill", False),
        chunked_prefill_size=raw.get("chunked_prefill_size"),
        enable_prefix_caching=raw.get("enable_prefix_caching", False),
        enforce_eager=raw.get("enforce_eager", False),
        speculative_algorithm=raw.get("speculative_algorithm"),
        speculative_draft_model=raw.get("speculative_draft_model"),
        speculative_num_steps=raw.get("speculative_num_steps"),
        num_continuous_decode_steps=raw.get("num_continuous_decode_steps", 1),
        dp_size=raw.get("dp_size"),
        rationale=raw.get("rationale", ""),
    )
