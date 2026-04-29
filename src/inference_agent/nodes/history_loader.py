"""History loader node — pulls top-N experiments from Postgres on startup.

Runs once per session, between `discovery` (which fixes hardware) and `planner`.
Loads top-2 experiments per category (throughput, latency, balanced) for the
current hardware + model, deduplicates by experiment_id, and writes them to
`state["loaded_top_history"]`. Subsequent nodes read these tops alongside the
in-session `experiment_history`.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from inference_agent.db.repository import ExperimentRepository
from inference_agent.state import AgentState

logger = logging.getLogger(__name__)


HistoryLoaderNode = Callable[[AgentState], Awaitable[dict]]


def make_history_loader_node(repo: ExperimentRepository) -> HistoryLoaderNode:
    """Build a history_loader node closing over the experiment repository."""

    async def history_loader_node(state: AgentState) -> dict:
        config = state["config"]
        hardware = state.get("hardware")
        if hardware is None:
            logger.warning(
                "history_loader: no hardware in state — discovery must run first; "
                "skipping with empty top history"
            )
            return {"loaded_top_history": []}

        summaries = await repo.find_top_for_hardware(
            hardware=hardware,
            model_name=config.model_name,
            latency_threshold_ms=config.benchmark.latency_threshold_ms,
            limit=2,
        )
        logger.info(
            "history_loader: loaded %d prior experiments for %s x%d (model=%s)",
            len(summaries),
            hardware.gpus[0].name if hardware.gpus else "?",
            hardware.gpu_count,
            config.model_name,
        )
        return {"loaded_top_history": summaries}

    return history_loader_node
