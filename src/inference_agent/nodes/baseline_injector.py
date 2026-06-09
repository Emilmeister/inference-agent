"""Baseline injector node — seeds the anchor experiment deterministically.

Unlike the planner, this node does NOT call the LLM. It takes the operator's
`baseline` config (from baseline.yaml, attached to AgentConfig) verbatim, stamps
`is_baseline=True`, and emits it as `current_config`. The rest of the graph
(validator → executor → analyzer → reporter) runs it exactly like any planned
experiment, so the benchmark harness measures a real, reproducible baseline that
the agent then iterates upon.

Routing: the graph sends control here (instead of the planner) only while
`state["baseline_pending"]` is True. The node clears that flag so subsequent
loop iterations go through the planner as usual.
"""

from __future__ import annotations

import logging

from inference_agent.models import AgentConfig, ExperimentConfig
from inference_agent.state import AgentState

logger = logging.getLogger(__name__)


async def baseline_injector_node(state: AgentState) -> dict:
    """Emit the operator-defined baseline as the next experiment config."""
    config: AgentConfig = state["config"]
    baseline = config.baseline

    if baseline is None:
        # Should never happen — the router only routes here when a baseline is
        # pending, which implies one exists. Guard defensively and fall through
        # to the planner by clearing the flag.
        logger.warning(
            "baseline_injector reached with no config.baseline — clearing "
            "baseline_pending and deferring to planner."
        )
        return {"baseline_pending": False}

    # Copy so we never mutate the shared AgentConfig.baseline across iterations
    # (the model is reused if the graph somehow re-enters). Force the metadata
    # flag regardless of what the YAML said.
    experiment = baseline.model_copy(deep=True)
    experiment.is_baseline = True
    if not experiment.rationale:
        experiment.rationale = (
            "Operator-defined baseline (baseline.yaml) — anchor experiment run "
            "deterministically without the planner. The agent iterates on these "
            "measured numbers."
        )

    logger.info(
        "Injecting BASELINE experiment %s: engine=%s, TP=%d, ctx=%s "
        "(deterministic, no LLM)",
        experiment.experiment_id,
        experiment.engine.value,
        experiment.tensor_parallel_size,
        experiment.max_model_len,
    )

    return {
        "current_config": experiment,
        "baseline_pending": False,
    }
