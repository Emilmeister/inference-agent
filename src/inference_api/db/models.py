"""SQLAlchemy ORM model for the `experiments` table."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base — `Base.metadata.create_all` builds the schema."""


class ExperimentRow(Base):
    __tablename__ = "experiments"

    experiment_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    engine: Mapped[str] = mapped_column(String(32), nullable=False)
    engine_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Hardware (flat, homogeneous cluster)
    gpu_name: Mapped[str] = mapped_column(String(128), nullable=False)
    gpu_count: Mapped[int] = mapped_column(Integer, nullable=False)
    gpu_vram_mb: Mapped[int] = mapped_column(Integer, nullable=False)
    nvlink_available: Mapped[bool] = mapped_column(Boolean, nullable=False)

    # Runtime / container (next to data, for direct SQL queries)
    container_image_digest: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    container_command: Mapped[str] = mapped_column(Text, nullable=False, default="")
    container_args: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)

    # Indexable metrics (for WHERE / ORDER BY in history_loader)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    correctness_gate_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    peak_throughput: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    low_concurrency_ttft_p95: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Operator-defined baseline anchor. Indexed flat column so the dashboard
    # and the agent's `find_baseline` lookup can locate the reference run for a
    # hardware+model without unpacking JSONB.
    is_baseline: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True
    )

    # Effective max context the engine was launched with: either the launch
    # flag (`config.max_model_len`) when the planner pinned it, or the
    # model's intrinsic max from HF config (`hardware.model_max_context`)
    # when no override was set. Surfaced as a flat column so the dashboard
    # and history queries can filter/sort by context size without unpacking
    # JSONB.
    max_model_len: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Full ExperimentResult.model_dump(mode="json")
    data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    __table_args__ = (
        Index(
            "ix_experiments_hardware_model",
            "gpu_name",
            "gpu_count",
            "gpu_vram_mb",
            "nvlink_available",
            "model_name",
        ),
    )


class QualityRunRow(Base):
    """One prod-readiness validation run of a quality suite for a fingerprint.

    Quality suites (so-testing, terminal-bench) validate the FINALIST configs
    after the optimization loop converges. Because a suite's outcome depends
    only on the quality fingerprint (model + quant + dtype + sampling + tool
    parser + …), the expensive run executes once per (fingerprint, suite) and
    is attributed to every finalist experiment that shares the fingerprint —
    listed in `experiment_ids`. The dashboard joins finalist experiments to
    their quality runs via this list. Idempotency: `id` = "<fingerprint>-<suite>"
    so a re-run upserts the same row.
    """

    __tablename__ = "quality_runs"

    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    suite: Mapped[str] = mapped_column(String(32), nullable=False)
    suite_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")

    model_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    gpu_name: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    gpu_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    gpu_vram_mb: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    nvlink_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    status: Mapped[str] = mapped_column(String(16), nullable=False)  # running|done|failed
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Finalist experiments sharing this fingerprint + their leaderboard labels.
    experiment_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    categories: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)

    # Full suite report (so-testing JSON contract, or terminal-bench summary).
    data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False,
    )

    __table_args__ = (
        Index(
            "ix_quality_runs_hardware_model",
            "gpu_name", "gpu_count", "gpu_vram_mb", "nvlink_available", "model_name",
        ),
    )
