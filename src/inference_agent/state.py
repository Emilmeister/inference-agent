"""LangGraph agent state definition."""

from __future__ import annotations

from typing import Annotated, TypedDict

from inference_agent.models import (
    AgentConfig,
    ExperimentConfig,
    ExperimentResult,
    ExperimentSummary,
    HardwareProfile,
    OptimizationGoal,
    ParetoPoint,
)


def _replace(old: object, new: object) -> object:
    """Reducer that always takes the new value."""
    return new


def _append_list(old: list, new: list) -> list:
    """Reducer that appends new items to the list."""
    return old + new


class AgentState(TypedDict, total=False):
    # Config (set once at start)
    config: Annotated[AgentConfig, _replace]

    # Hardware profile (set by discovery node)
    hardware: Annotated[HardwareProfile, _replace]

    # Current experiment
    current_config: Annotated[ExperimentConfig | None, _replace]
    current_result: Annotated[ExperimentResult | None, _replace]

    # History — current session only (used for plateau detection and best_* updates)
    experiment_history: Annotated[list[ExperimentSummary], _append_list]
    experiments_count: Annotated[int, _replace]

    # Top experiments loaded from DB on startup (max 6 — top-2 in 3 categories,
    # deduplicated). Read-only after history_loader. Combined with
    # `experiment_history` for leaderboards and Pareto, but NOT used for
    # plateau detection (otherwise plateau triggers immediately on prior tops).
    loaded_top_history: Annotated[list[ExperimentSummary], _replace]

    # Leaderboards — throughput
    best_throughput: Annotated[float, _replace]
    best_throughput_config_id: Annotated[str, _replace]

    # Leaderboards — latency
    best_latency_ttft_p95: Annotated[float, _replace]
    best_latency_config_id: Annotated[str, _replace]

    # Leaderboards — balanced (Pareto)
    best_balanced_config_id: Annotated[str, _replace]
    best_balanced_throughput: Annotated[float, _replace]
    best_balanced_latency: Annotated[float, _replace]

    # Pareto front
    pareto_front: Annotated[list[ParetoPoint], _replace]

    # Next optimization direction
    next_optimization_goal: Annotated[OptimizationGoal, _replace]

    # Validation
    skip_executor: Annotated[bool, _replace]

    # Status
    status: Annotated[str, _replace]  # "running" | "completed" | "failed"
    stop_reason: Annotated[str | None, _replace]
