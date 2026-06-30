"""Async repository for quality_runs (prod-readiness validation of finalists)."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from inference_api.db.models import QualityRunRow

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

    async def get(self, run_id: str) -> dict[str, Any] | None:
        async with self._sessionmaker() as session:
            row = await session.get(QualityRunRow, run_id)
            return row_to_dict(row) if row is not None else None

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
        query = select(QualityRunRow)
        if clauses:
            query = query.where(*clauses)
        query = query.order_by(QualityRunRow.updated_at.desc())
        async with self._sessionmaker() as session:
            rows = (await session.execute(query)).scalars().all()
        out = [row_to_dict(r) for r in rows]
        # JSONB containment for experiment_id is awkward to express portably;
        # filter in Python (the result set here is small — finalists only).
        if experiment_id is not None:
            out = [r for r in out if experiment_id in r["experiment_ids"]]
        return out
