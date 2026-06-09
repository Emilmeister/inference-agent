"""History loader node — pulls top-N experiments via inference-api on startup.

Runs once per session, between `discovery` (which fixes hardware) and `planner`.
Loads top-2 experiments per category (throughput, latency, balanced) for the
current hardware + model, deduplicates by experiment_id, and writes them to
`state["loaded_top_history"]`. Subsequent nodes read these tops alongside the
in-session `experiment_history`.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from inference_agent.api_client import ExperimentApiClient
from inference_agent.state import AgentState

logger = logging.getLogger(__name__)


HistoryLoaderNode = Callable[[AgentState], Awaitable[dict]]


def make_history_loader_node(client: ExperimentApiClient) -> HistoryLoaderNode:
    """Build a history_loader node closing over the API client."""

    async def history_loader_node(state: AgentState) -> dict:
        config = state["config"]
        hardware = state.get("hardware")
        if hardware is None:
            logger.warning(
                "history_loader: no hardware in state — discovery must run first; "
                "skipping with empty top history"
            )
            return {"loaded_top_history": []}

        summaries = await client.find_top_for_hardware(
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

        # Baseline anchor: look up whether a baseline already exists for this
        # hardware+model. If so, surface it (planner anchor + impact tracking)
        # and clear baseline_pending so we don't re-run it. If not and the
        # operator configured one, leave baseline_pending True so the graph
        # injects it as experiment #1.
        out: dict = {"loaded_top_history": summaries}
        existing_baseline = await client.find_baseline(
            hardware=hardware,
            model_name=config.model_name,
        )
        if existing_baseline is not None:
            logger.info(
                "history_loader: found existing baseline %s — anchoring on it, "
                "will NOT re-run (max_viable_agents=%d, ttft_p95=%.1fms)",
                existing_baseline.experiment_id,
                existing_baseline.agentic_max_viable_concurrency,
                existing_baseline.low_concurrency_ttft_p95,
            )
            out["baseline_summary"] = existing_baseline
            out["baseline_pending"] = False
        elif config.baseline is not None:
            logger.info(
                "history_loader: no baseline in DB yet — will run the configured "
                "baseline.yaml as experiment #1."
            )
            out["baseline_pending"] = True
        else:
            out["baseline_pending"] = False
        return out

    return history_loader_node
