"""Postgres persistence layer.

`ExperimentRepository` is the only entry point used by agent nodes (reporter,
history_loader). `init_schema` is called once on startup from `cli._run`.
"""

from inference_agent.db.engine import init_schema
from inference_agent.db.models import Base, ExperimentRow
from inference_agent.db.repository import ExperimentRepository

__all__ = [
    "Base",
    "ExperimentRow",
    "ExperimentRepository",
    "init_schema",
]
