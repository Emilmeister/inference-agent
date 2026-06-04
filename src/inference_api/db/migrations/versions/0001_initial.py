"""initial schema (pre max_model_len)

Revision ID: 0001
Revises:
Create Date: 2026-04-29 13:00:00

This is the schema as it existed before the alembic switch — i.e. exactly
what `Base.metadata.create_all` produced up to and including the original
indexable columns. Subsequent revisions ALTER from here.

For databases that already have data from the create_all era, run
`alembic stamp 0001` to mark this revision as applied without re-running it.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "experiments",
        sa.Column("experiment_id", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("engine", sa.String(length=32), nullable=False),
        sa.Column("engine_version", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("gpu_name", sa.String(length=128), nullable=False),
        sa.Column("gpu_count", sa.Integer(), nullable=False),
        sa.Column("gpu_vram_mb", sa.Integer(), nullable=False),
        sa.Column("nvlink_available", sa.Boolean(), nullable=False),
        sa.Column("docker_image_digest", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("docker_command", sa.Text(), nullable=False, server_default=""),
        sa.Column("docker_args", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("correctness_gate_passed", sa.Boolean(), nullable=False),
        sa.Column("peak_throughput", sa.Float(), nullable=False, server_default="0"),
        sa.Column("low_concurrency_ttft_p95", sa.Float(), nullable=False, server_default="0"),
        sa.Column("data", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint("experiment_id"),
    )
    op.create_index(
        "ix_experiments_hardware_model",
        "experiments",
        ["gpu_name", "gpu_count", "gpu_vram_mb", "nvlink_available", "model_name"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_experiments_hardware_model", table_name="experiments")
    op.drop_table("experiments")
