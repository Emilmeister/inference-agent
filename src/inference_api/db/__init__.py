"""Postgres persistence layer for the REST service.

`ExperimentRepository` is the only entry point used by HTTP route handlers
(experiments router). `init_schema` is called once on service startup from
`inference_api.app.lifespan`.
"""

from inference_api.db.engine import init_schema
from inference_api.db.models import Base, ExperimentRow
from inference_api.db.repository import ExperimentRepository

__all__ = [
    "Base",
    "ExperimentRow",
    "ExperimentRepository",
    "init_schema",
]
