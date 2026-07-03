"""Async repository for quality_runs (prod-readiness validation of finalists)."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from inference_api.db.models import ExperimentRow, QualityRunRow

logger = logging.getLogger(__name__)

_UPSERT_FIELDS = (
    "fingerprint", "suite", "suite_version", "model_name",
    "gpu_name", "gpu_count", "gpu_vram_mb", "nvlink_available",
    "status", "score", "error", "experiment_ids", "categories", "data",
)


def row_to_dict(row: QualityRunRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "fingerprint": row.fingerprint,
        "suite": row.suite,
        "suite_version": row.suite_version,
        "model_name": row.model_name,
        "gpu_name": row.gpu_name,
        "gpu_count": row.gpu_count,
        "gpu_vram_mb": row.gpu_vram_mb,
        "nvlink_available": row.nvlink_available,
        "status": row.status,
        "score": row.score,
        "error": row.error,
        "experiment_ids": list(row.experiment_ids or []),
        "categories": list(row.categories or []),
        "data": dict(row.data or {}),
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


class QualityRepository:
    """Persist + query quality suite runs. Upsert keyed by `id`."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]):
        self._sessionmaker = sessionmaker

    async def upsert(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Insert or replace a quality run by `id`. Returns the stored row."""
        async with self._sessionmaker() as session:
            existing = await session.get(QualityRunRow, payload["id"])
            if existing is None:
                row = QualityRunRow(id=payload["id"])
                for field in _UPSERT_FIELDS:
                    setattr(row, field, payload.get(field))
                session.add(row)
            else:
                for field in _UPSERT_FIELDS:
                    if field in payload:
                        setattr(existing, field, payload[field])
                row = existing
            await session.commit()
            await session.refresh(row)
            result = row_to_dict(row)
        logger.info(
            "Upserted quality run %s (suite=%s, status=%s, score=%s)",
            payload["id"], payload.get("suite"), payload.get("status"), payload.get("score"),
        )
        return result

    async def _fingerprint_of_experiment(
        self, session: AsyncSession, experiment_id: str
    ) -> str | None:
        """The quality fingerprint of one experiment, or None if unknown/empty."""
        fp = (
            await session.execute(
                select(ExperimentRow.quality_fingerprint).where(
                    ExperimentRow.experiment_id == experiment_id
                )
            )
        ).scalar_one_or_none()
        return fp or None

    async def _matched_ids(
        self, session: AsyncSession, fingerprints: set[str]
    ) -> dict[str, list[str]]:
        """Map each fingerprint → all experiment ids that share it (live join).

        This is the attribution: a quality run belongs to its fingerprint, so
        every experiment carrying that fingerprint inherits the result — not
        just the finalists that happened to trigger the run.
        """
        fps = {f for f in fingerprints if f}
        if not fps:
            return {}
        rows = (
            await session.execute(
                select(
                    ExperimentRow.quality_fingerprint, ExperimentRow.experiment_id
                ).where(ExperimentRow.quality_fingerprint.in_(fps))
            )
        ).all()
        out: dict[str, list[str]] = {}
        for fp, eid in rows:
            out.setdefault(fp, []).append(eid)
        return {fp: sorted(ids) for fp, ids in out.items()}

    async def get(self, run_id: str) -> dict[str, Any] | None:
        async with self._sessionmaker() as session:
            row = await session.get(QualityRunRow, run_id)
            if row is None:
                return None
            record = row_to_dict(row)
            matched = await self._matched_ids(session, {row.fingerprint})
            record["matched_experiment_ids"] = matched.get(row.fingerprint, [])
            return record

    async def list(
        self,
        *,
        fingerprint: str | None = None,
        suite: str | None = None,
        model_name: str | None = None,
        gpu_name: str | None = None,
        gpu_count: int | None = None,
        gpu_vram_mb: int | None = None,
        nvlink_available: bool | None = None,
        experiment_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = []
        if fingerprint is not None:
            clauses.append(QualityRunRow.fingerprint == fingerprint)
        if suite is not None:
            clauses.append(QualityRunRow.suite == suite)
        if model_name is not None:
            clauses.append(QualityRunRow.model_name == model_name)
        if gpu_name is not None:
            clauses.append(QualityRunRow.gpu_name == gpu_name)
        if gpu_count is not None:
            clauses.append(QualityRunRow.gpu_count == gpu_count)
        if gpu_vram_mb is not None:
            clauses.append(QualityRunRow.gpu_vram_mb == gpu_vram_mb)
        if nvlink_available is not None:
            clauses.append(QualityRunRow.nvlink_available == nvlink_available)

        async with self._sessionmaker() as session:
            # Resolve experiment_id → fingerprint so filtering covers EVERY
            # config sharing the fingerprint (finalist or not), not just the
            # static finalist list stored on the run.
            if experiment_id is not None:
                exp_fp = await self._fingerprint_of_experiment(session, experiment_id)
                if exp_fp is None:
                    return []
                clauses.append(QualityRunRow.fingerprint == exp_fp)

            query = select(QualityRunRow)
            if clauses:
                query = query.where(*clauses)
            query = query.order_by(QualityRunRow.updated_at.desc())
            rows = (await session.execute(query)).scalars().all()

            out = [row_to_dict(r) for r in rows]
            matched = await self._matched_ids(session, {r.fingerprint for r in rows})
            for record, r in zip(out, rows):
                record["matched_experiment_ids"] = matched.get(r.fingerprint, [])
        return out
