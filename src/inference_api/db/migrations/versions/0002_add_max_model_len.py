"""add max_model_len column to experiments

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-29 13:05:00

Adds an indexable column for the effective context length the engine was
launched with (config.max_model_len, falling back to model intrinsic max).
Existing rows get 0, which mappers and dashboards treat as "unknown".
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "experiments",
        sa.Column(
            "max_model_len",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column("experiments", "max_model_len")
