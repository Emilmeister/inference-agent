"""Async repository for experiment persistence and top-N retrieval."""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from inference_agent.db.mappers import (
    _assert_homogeneous,
    result_to_row,
    row_to_summary,
)
from inference_agent.db.models import ExperimentRow
from inference_agent.models_pkg.domain import (
    ExperimentResult,
    ExperimentSummary,
    HardwareProfile,
)

logger = logging.getLogger(__name__)


class ExperimentRepository:
    """Persist experiment results and load top-N history for matching hardware."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]):
        self._sessionmaker = sessionmaker

    async def insert_experiment(self, result: ExperimentResult) -> None:
        """Insert a single experiment result. Commits immediately."""
        row = result_to_row(result)
        async with self._sessionmaker() as session:
            session.add(row)
            await session.commit()
        logger.info(
            "Persisted experiment %s to Postgres (engine=%s, status=%s)",
            result.experiment_id,
            result.engine.value,
            result.status.value,
        )

    async def find_top_for_hardware(
        self,
        hardware: HardwareProfile,
        model_name: str,
        latency_threshold_ms: float,
        limit: int = 2,
    ) -> list[ExperimentSummary]:
        """Load top-`limit` experiments in each of 3 categories.

        Categories:
          - top throughput: highest `peak_throughput`
          - top latency:    lowest `low_concurrency_ttft_p95` (must be > 0)
          - top balanced:   highest `peak_throughput` where ttft_p95 < threshold

        All filters require: matching hardware (gpu_name/count/vram/nvlink) and
        model_name, plus eligibility (correctness gate passed, status='success',
        peak_throughput > 0).

        Returns deduplicated list (max `3*limit` summaries, often fewer).
        """
        _assert_homogeneous(hardware)
        primary = hardware.gpus[0]

        eligibility = (
            (ExperimentRow.gpu_name == primary.name)
            & (ExperimentRow.gpu_count == hardware.gpu_count)
            & (ExperimentRow.gpu_vram_mb == primary.vram_total_mb)
            & (ExperimentRow.nvlink_available == hardware.nvlink_available)
            & (ExperimentRow.model_name == model_name)
            & (ExperimentRow.correctness_gate_passed.is_(True))
            & (ExperimentRow.status == "success")
            & (ExperimentRow.peak_throughput > 0)
        )

        async with self._sessionmaker() as session:
            top_tp_q = (
                select(ExperimentRow)
                .where(eligibility)
                .order_by(ExperimentRow.peak_throughput.desc())
                .limit(limit)
            )
            top_lat_q = (
                select(ExperimentRow)
                .where(eligibility, ExperimentRow.low_concurrency_ttft_p95 > 0)
                .order_by(ExperimentRow.low_concurrency_ttft_p95.asc())
                .limit(limit)
            )
            top_balanced_q = (
                select(ExperimentRow)
                .where(
                    eligibility,
                    ExperimentRow.low_concurrency_ttft_p95 > 0,
                    ExperimentRow.low_concurrency_ttft_p95 < latency_threshold_ms,
                )
                .order_by(ExperimentRow.peak_throughput.desc())
                .limit(limit)
            )

            top_tp = (await session.execute(top_tp_q)).scalars().all()
            top_lat = (await session.execute(top_lat_q)).scalars().all()
            top_balanced = (await session.execute(top_balanced_q)).scalars().all()

        seen: set[str] = set()
        summaries: list[ExperimentSummary] = []
        for row in [*top_tp, *top_lat, *top_balanced]:
            if row.experiment_id in seen:
                continue
            seen.add(row.experiment_id)
            summaries.append(row_to_summary(row))
        return summaries
