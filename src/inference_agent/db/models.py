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

    # Runtime / docker (next to data, for direct SQL queries)
    docker_image_digest: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    docker_command: Mapped[str] = mapped_column(Text, nullable=False, default="")
    docker_args: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)

    # Indexable metrics (for WHERE / ORDER BY in history_loader)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    correctness_gate_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    peak_throughput: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    low_concurrency_ttft_p95: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

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
