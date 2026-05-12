"""rename docker_* columns to container_*

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-12 12:00:00

Project migrated from docker to nerdctl/containerd as the container runtime.
Column names are renamed in lockstep with the Python field renames in
`ExperimentResult` and `ExperimentRow`. JSONB `data` payloads still contain
the new field names because `ExperimentResult.model_dump` is the source of
truth for that blob — old rows written before this migration keep their
`docker_*` keys inside `data`, and the dashboard's JSONB projection now
reads `container_*`. Old rows are not retro-migrated; the JSONB blob is
not consumed by the agent past insert.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "experiments", "docker_image_digest",
        new_column_name="container_image_digest",
    )
    op.alter_column(
        "experiments", "docker_command",
        new_column_name="container_command",
    )
    op.alter_column(
        "experiments", "docker_args",
        new_column_name="container_args",
    )


def downgrade() -> None:
    op.alter_column(
        "experiments", "container_image_digest",
        new_column_name="docker_image_digest",
    )
    op.alter_column(
        "experiments", "container_command",
        new_column_name="docker_command",
    )
    op.alter_column(
        "experiments", "container_args",
        new_column_name="docker_args",
    )
