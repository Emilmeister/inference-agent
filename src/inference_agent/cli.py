"""CLI entrypoint for the inference benchmark agent."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

import yaml

from inference_agent.agent import compile_agent
from inference_agent.api_client import ExperimentApiClient
from inference_agent.models import AgentConfig, OptimizationGoal
from inference_agent.utils.container import stop_all_bench_containers
from inference_agent.utils.logging import setup_logging


# agent_llm fields that can be overridden via env vars (AGENT_LLM_<UPPER>).
# Values are coerced by Pydantic when AgentConfig is constructed.
_AGENT_LLM_ENV_FIELDS = (
    "base_url",
    "model",
    "api_key",
    "api_key_env",
    "temperature",
    "max_tokens",
    "timeout_sec",
    "structured_output_mode",
    "max_budget_usd",
)


# api.<field> overridable via AGENT_API_<UPPER>.
_API_ENV_FIELDS = (
    "base_url",
    "token",
    "token_env",
    "timeout_sec",
)


def _apply_agent_llm_env_overrides(raw: dict) -> None:
    """Override agent_llm.<field> with env var AGENT_LLM_<FIELD> if set."""
    section = raw.setdefault("agent_llm", {})
    for field in _AGENT_LLM_ENV_FIELDS:
        env_name = f"AGENT_LLM_{field.upper()}"
        if env_name in os.environ:
            section[field] = os.environ[env_name]


def _apply_api_env_overrides(raw: dict) -> None:
    """Override api.<field> with env var AGENT_API_<FIELD> if set."""
    section = raw.setdefault("api", {})
    for field in _API_ENV_FIELDS:
        env_name = f"AGENT_API_{field.upper()}"
        if env_name in os.environ:
            section[field] = os.environ[env_name]


def _load_config(path: str, baseline_path: str | None = None) -> AgentConfig:
    """Load and validate config from YAML file.

    Env vars override agent_llm and api fields (AGENT_LLM_* / AGENT_API_*).
    If `baseline_path` is given and the file exists, it is parsed as an
    ExperimentConfig and attached as `config.baseline` (the operator anchor).
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    # Resolve ${VAR} placeholder in api_key
    if "agent_llm" in raw:
        api_key = raw["agent_llm"].get("api_key", "")
        if isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            raw["agent_llm"]["api_key"] = os.environ.get(env_var, "")

    _apply_agent_llm_env_overrides(raw)
    _apply_api_env_overrides(raw)

    # Baseline anchor — separate file (default baseline.yaml). Shaped like an
    # ExperimentConfig. Inline `baseline:` in config.yaml takes precedence if
    # someone set it there; otherwise the side file wins.
    if baseline_path and os.path.exists(baseline_path) and "baseline" not in raw:
        with open(baseline_path) as bf:
            raw["baseline"] = yaml.safe_load(bf)

    return AgentConfig(**raw)


async def _run(config: AgentConfig) -> None:
    """Run the agent."""
    logger = logging.getLogger("inference_agent")

    if not config.api.token:
        raise RuntimeError(
            "API token not configured — set INFERENCE_API_TOKEN (or api.token in YAML). "
            "The agent talks to the inference-api REST service via Bearer auth."
        )

    client = ExperimentApiClient(
        base_url=config.api.base_url,
        token=config.api.token,
        timeout_sec=config.api.timeout_sec,
    )

    async with client:
        agent = compile_agent(client)

        initial_state = {
            "config": config,
            "experiment_history": [],
            "loaded_top_history": [],
            "experiments_count": 0,
            # Agentic-first bests (primary leaderboard).
            "best_agentic_max_viable_c": 0,
            "best_agentic_config_id": "",
            "best_agentic_tpot_p95": float("inf"),
            "best_agentic_throughput": 0.0,
            # Throughput/latency/balanced tracked for backward compatibility.
            "best_throughput": 0.0,
            "best_throughput_config_id": "",
            "best_latency_ttft_p95": float("inf"),
            "best_latency_config_id": "",
            "best_balanced_config_id": "",
            "best_balanced_throughput": 0.0,
            "best_balanced_latency": float("inf"),
            "pareto_front": [],
            "agentic_pareto_front": [],
            # Start with the agentic goal so the FIRST experiment goes through
            # the validator's agentic gates (prefix-cache mandatory, sane
            # max_model_len). Without this, exp #1 lands under explore and
            # the planner is free to pick a throughput-shaped config.
            "next_optimization_goal": OptimizationGoal.AGENTIC,
            "status": "running",
            "stop_reason": None,
            "current_config": None,
            "current_result": None,
            "skip_executor": False,
            # Baseline anchor: pending iff an operator baseline is configured.
            # history_loader downgrades this to False when a baseline already
            # exists in the DB for this hardware/model (don't re-run it).
            "baseline_pending": config.baseline is not None,
            "baseline_summary": None,
        }

        logger.info("Starting inference benchmark agent")
        logger.info("Model: %s", config.model_name)
        logger.info("Max experiments: %d", config.experiments.max_experiments)
        logger.info("Engines: %s", [e.value for e in config.experiments.engines])
        logger.info("Inference API: %s", config.api.base_url)

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

    print(f"\nResults posted to {config.api.base_url}")


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
        "--baseline",
        default="baseline.yaml",
        help=(
            "Path to baseline launch config (default: baseline.yaml). When the "
            "file exists it is run as the anchor experiment #1 and highlighted "
            "as the baseline in the dashboard. Missing file → no baseline."
        ),
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

    config = _load_config(args.config, baseline_path=args.baseline)
    asyncio.run(_run(config))


if __name__ == "__main__":
    main()
