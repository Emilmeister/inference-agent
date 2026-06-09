"""add is_baseline column to experiments

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-09 10:00:00

Adds an indexed boolean flag marking the operator-defined baseline anchor run
(from baseline.yaml). Existing rows default to False (no baseline recorded yet).
The index supports the agent's `find_baseline` lookup and dashboard filtering
of the reference run per hardware+model.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "experiments",
        sa.Column(
            "is_baseline",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.create_index(
        "ix_experiments_is_baseline",
        "experiments",
        ["is_baseline"],
    )


def downgrade() -> None:
    op.drop_index("ix_experiments_is_baseline", table_name="experiments")
    op.drop_column("experiments", "is_baseline")
