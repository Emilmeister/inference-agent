"""Reporter node — persists experiment results to Postgres."""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from inference_agent.db.repository import ExperimentRepository
from inference_agent.state import AgentState

logger = logging.getLogger(__name__)


ReporterNode = Callable[[AgentState], Awaitable[dict]]


def make_reporter_node(repo: ExperimentRepository) -> ReporterNode:
    """Build a reporter node closing over the experiment repository."""

    async def reporter_node(state: AgentState) -> dict:
        result = state.get("current_result")
        if result is None:
            logger.warning("Reporter: no current_result in state, skipping save")
            return {}

        await repo.insert_experiment(result)
        return {}

    return reporter_node
