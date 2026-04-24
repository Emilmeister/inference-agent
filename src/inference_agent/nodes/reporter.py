"""Reporter node — saves experiment results to JSON files."""

from __future__ import annotations

import json
import logging
import os
import tempfile

from inference_agent.models import ExperimentResult
from inference_agent.state import AgentState

logger = logging.getLogger(__name__)


def save_experiment(result: ExperimentResult, experiments_dir: str) -> str:
    """Save an experiment result to a JSON file atomically.

    Uses temp file → fsync → rename to prevent partial writes.
    Returns the filepath.
    """
    os.makedirs(experiments_dir, exist_ok=True)
    filename = f"{result.experiment_id}.json"
    filepath = os.path.join(experiments_dir, filename)
    data = result.model_dump(mode="json")

    # Atomic write: write to temp file, fsync, then rename
    fd, tmp_path = tempfile.mkstemp(
        dir=experiments_dir, prefix=f".{result.experiment_id}_", suffix=".json.tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, filepath)
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    logger.info("Saved experiment result to %s", filepath)
    return filepath


async def reporter_node(state: AgentState) -> dict:
    """Persist the enriched experiment result to storage."""
    config = state["config"]
    result = state.get("current_result")

    if result is None:
        logger.warning("Reporter: no current_result in state, skipping save")
        return {}

    save_experiment(result, config.storage.experiments_dir)
    return {}
