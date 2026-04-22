"""Reporter node — saves experiment results to JSON files."""

from __future__ import annotations

import json
import logging
import os

from inference_agent.models import ExperimentResult
from inference_agent.state import AgentState

logger = logging.getLogger(__name__)


def save_experiment(result: ExperimentResult, experiments_dir: str) -> str:
    """Save an experiment result to a JSON file. Returns the filepath."""
    os.makedirs(experiments_dir, exist_ok=True)
    filename = f"{result.experiment_id}.json"
    filepath = os.path.join(experiments_dir, filename)
    data = result.model_dump(mode="json")
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Saved experiment result to %s", filepath)
    return filepath


async def reporter_node(state: AgentState) -> dict:
    """Save the current experiment result to a JSON file."""
    config = state["config"]
    result = state["current_result"]
    save_experiment(result, config.storage.experiments_dir)
    return {}
