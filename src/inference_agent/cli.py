"""CLI entrypoint for the inference benchmark agent."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

import yaml

from inference_agent.agent import compile_agent
from inference_agent.models import AgentConfig, OptimizationGoal
from inference_agent.utils.docker import stop_all_bench_containers
from inference_agent.utils.logging import setup_logging


def _load_config(path: str) -> AgentConfig:
    """Load and validate config from YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    # Resolve environment variables in api_key
    if "agent_llm" in raw:
        api_key = raw["agent_llm"].get("api_key", "")
        if api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            raw["agent_llm"]["api_key"] = os.environ.get(env_var, "")

    return AgentConfig(**raw)


async def _run(config: AgentConfig) -> None:
    """Run the agent."""
    agent = compile_agent()

    initial_state = {
        "config": config,
        "experiment_history": [],
        "experiments_count": 0,
        "best_throughput": 0.0,
        "best_throughput_config_id": "",
        "best_latency_ttft_p95": float("inf"),
        "best_latency_config_id": "",
        "best_balanced_config_id": "",
        "best_balanced_throughput": 0.0,
        "best_balanced_latency": float("inf"),
        "pareto_front": [],
        "next_optimization_goal": OptimizationGoal.EXPLORE,
        "status": "running",
        "stop_reason": None,
        "current_config": None,
        "current_result": None,
        "skip_executor": False,
    }

    logger = logging.getLogger("inference_agent")
    logger.info("Starting inference benchmark agent")
    logger.info("Model: %s", config.model_name)
    logger.info("Max experiments: %d", config.experiments.max_experiments)
    logger.info("Engines: %s", [e.value for e in config.experiments.engines])

    try:
        final_state = await agent.ainvoke(initial_state)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        stop_all_bench_containers()
        return
    except Exception:
        logger.exception("Agent failed")
        stop_all_bench_containers()
        raise

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Experiments run: {final_state.get('experiments_count', 0)}")
    print(f"Stop reason: {final_state.get('stop_reason', 'unknown')}")
    print()
    print("=== BEST CONFIGURATIONS ===")
    print()

    tp = final_state.get("best_throughput", 0)
    tp_id = final_state.get("best_throughput_config_id", "")
    print(f"Best Throughput: {tp:.1f} tok/s (experiment: {tp_id})")

    lat = final_state.get("best_latency_ttft_p95", 0)
    lat_id = final_state.get("best_latency_config_id", "")
    print(f"Best Latency (TTFT p95): {lat:.1f} ms (experiment: {lat_id})")

    bal_id = final_state.get("best_balanced_config_id", "")
    bal_tp = final_state.get("best_balanced_throughput", 0)
    bal_lat = final_state.get("best_balanced_latency", 0)
    print(f"Best Balanced: {bal_tp:.1f} tok/s @ {bal_lat:.1f} ms (experiment: {bal_id})")

    pareto = final_state.get("pareto_front", [])
    if pareto:
        print(f"\nPareto front: {len(pareto)} configurations")
        for p in pareto:
            print(f"  {p.config_id}: {p.throughput:.1f} tok/s, TTFT p95={p.ttft_p95:.1f} ms")

    print(f"\nResults saved to: {config.storage.experiments_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Inference Benchmark Agent"
    )
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Stop all benchmark containers and exit",
    )
    args = parser.parse_args()

    # Setup structured logging with experiment context
    setup_logging(verbose=args.verbose)

    if args.cleanup:
        stop_all_bench_containers()
        print("Cleaned up all benchmark containers")
        return

    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = _load_config(args.config)
    asyncio.run(_run(config))


if __name__ == "__main__":
    main()
