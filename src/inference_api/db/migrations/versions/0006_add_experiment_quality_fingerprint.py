"""add experiments.quality_fingerprint (+ backfill)

Revision ID: 0006
Revises: 0005
Create Date: 2026-07-03 12:00:00

Persist the quality fingerprint on each experiment so quality_runs (keyed by
fingerprint) attribute to EVERY experiment sharing it via a live join, not just
the finalist ids stored on the run. Existing rows are backfilled by recomputing
the fingerprint from their JSONB `data` payload — the same function the mapper
uses on write, so the values match going forward.
"""

from __future__ import annotations

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

logger = logging.getLogger("alembic.runtime.migration")


def _backfill() -> None:
    """Recompute the fingerprint for every existing row from its `data` JSONB."""
    # Imported lazily so the migration module is importable without the agent
    # package on the path during offline `alembic revision` tooling.
    from inference_agent.models_pkg.domain import ExperimentResult
    from inference_agent.quality.fingerprint import quality_fingerprint

    bind = op.get_bind()
    rows = bind.execute(
        sa.text("SELECT experiment_id, data FROM experiments")
    ).fetchall()
    updated = 0
    for experiment_id, data in rows:
        try:
            result = ExperimentResult.model_validate(data)
            fp = quality_fingerprint(result.config, result.hardware, result.model)
        except Exception:  # noqa: BLE001 — a malformed legacy row must not abort the migration
            logger.warning("0006 backfill: skipping unparseable experiment %s", experiment_id)
            continue
        bind.execute(
            sa.text(
                "UPDATE experiments SET quality_fingerprint = :fp "
                "WHERE experiment_id = :eid"
            ),
            {"fp": fp, "eid": experiment_id},
        )
        updated += 1
    logger.info("0006 backfill: set quality_fingerprint on %d experiment(s)", updated)


def upgrade() -> None:
    op.add_column(
        "experiments",
        sa.Column(
            "quality_fingerprint",
            sa.String(length=32),
            nullable=False,
            server_default="",
        ),
    )
    op.create_index(
        "ix_experiments_quality_fingerprint", "experiments", ["quality_fingerprint"]
    )
    _backfill()


def downgrade() -> None:
    op.drop_index("ix_experiments_quality_fingerprint", table_name="experiments")
    op.drop_column("experiments", "quality_fingerprint")
