"""Quality-run routes — prod-readiness validation of finalist configs.

The agent's `quality_finalize` node upserts a run per (fingerprint, suite) and
queries existing runs for idempotency (skip already-done suites). The dashboard
lists runs to render the Prod-readiness tab.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from inference_api.auth import require_bearer
from inference_api.schemas import (
    QualityRunListResponse,
    QualityRunRecord,
    QualityRunUpsert,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quality", dependencies=[Depends(require_bearer)])


def _repo(request: Request):
    return request.app.state.quality_repository


@router.post("/runs", response_model=QualityRunRecord, status_code=status.HTTP_200_OK)
async def upsert_run(body: QualityRunUpsert, request: Request) -> QualityRunRecord:
    stored = await _repo(request).upsert(body.model_dump())
    return QualityRunRecord(**stored)


@router.get("/runs", response_model=QualityRunListResponse)
async def list_runs(
    request: Request,
    fingerprint: str | None = Query(None),
    suite: str | None = Query(None),
    model_name: str | None = Query(None),
    gpu_name: str | None = Query(None),
    gpu_count: int | None = Query(None),
    gpu_vram_mb: int | None = Query(None),
    nvlink_available: bool | None = Query(None),
    experiment_id: str | None = Query(None),
) -> QualityRunListResponse:
    rows = await _repo(request).list(
        fingerprint=fingerprint,
        suite=suite,
        model_name=model_name,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        gpu_vram_mb=gpu_vram_mb,
        nvlink_available=nvlink_available,
        experiment_id=experiment_id,
    )
    return QualityRunListResponse(runs=[QualityRunRecord(**r) for r in rows])


@router.get("/runs/{run_id}", response_model=QualityRunRecord)
async def get_run(run_id: str, request: Request) -> QualityRunRecord:
    row = await _repo(request).get(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="quality run not found")
    return QualityRunRecord(**row)
