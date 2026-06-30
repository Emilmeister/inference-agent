"""add quality_runs table

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-30 12:00:00

Stores prod-readiness validation runs of the quality suites (so-testing,
terminal-bench) for finalist configs. One row per (fingerprint, suite) —
`id` = "<fingerprint>-<suite>" — attributed to every finalist experiment that
shares the fingerprint via `experiment_ids`.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "quality_runs",
        sa.Column("id", sa.String(length=96), primary_key=True),
        sa.Column("fingerprint", sa.String(length=32), nullable=False),
        sa.Column("suite", sa.String(length=32), nullable=False),
        sa.Column("suite_version", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("gpu_name", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("gpu_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("gpu_vram_mb", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("nvlink_available", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("error", sa.Text(), nullable=False, server_default=""),
        sa.Column("experiment_ids", JSONB(), nullable=False, server_default="[]"),
        sa.Column("categories", JSONB(), nullable=False, server_default="[]"),
        sa.Column("data", JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_quality_runs_fingerprint", "quality_runs", ["fingerprint"])
    op.create_index("ix_quality_runs_model_name", "quality_runs", ["model_name"])
    op.create_index(
        "ix_quality_runs_hardware_model",
        "quality_runs",
        ["gpu_name", "gpu_count", "gpu_vram_mb", "nvlink_available", "model_name"],
    )


def downgrade() -> None:
    op.drop_index("ix_quality_runs_hardware_model", table_name="quality_runs")
    op.drop_index("ix_quality_runs_model_name", table_name="quality_runs")
    op.drop_index("ix_quality_runs_fingerprint", table_name="quality_runs")
    op.drop_table("quality_runs")
