"""Experiment CRUD and history routes."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from inference_agent.models_pkg.domain import (
    EngineType,
    ExperimentResult,
    ExperimentStatus,
    ExperimentSummary,
    GPUInfo,
    HardwareProfile,
)
from inference_api.auth import require_bearer
from inference_api.db.repository import ExperimentRepository
from inference_api.schemas import (
    AgenticTurnRow,
    BaselineResponse,
    CreateExperimentResponse,
    DeleteResponse,
    ExperimentDetailResponse,
    ExperimentPhaseRow,
    ExperimentSummaryRow,
    HardwareOption,
    IdListBody,
    TopHistoryResponse,
)

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/experiments", dependencies=[Depends(require_bearer)])


def _repo(request: Request) -> ExperimentRepository:
    return request.app.state.repository


# ── Writes ─────────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=CreateExperimentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_experiment(
    body: ExperimentResult, request: Request
) -> CreateExperimentResponse:
    await _repo(request).insert_experiment(body)
    return CreateExperimentResponse(experiment_id=body.experiment_id)


@router.delete("", response_model=DeleteResponse)
async def delete_experiments(
    body: IdListBody, request: Request
) -> DeleteResponse:
    deleted = await _repo(request).delete_experiments(body.experiment_ids)
    return DeleteResponse(deleted=deleted)


# ── History (agent) ────────────────────────────────────────────────────────


@router.get("/top", response_model=TopHistoryResponse)
async def get_top(
    request: Request,
    gpu_name: str = Query(...),
    gpu_count: int = Query(..., ge=1),
    gpu_vram_mb: int = Query(..., ge=1),
    nvlink_available: bool = Query(...),
    model_name: str = Query(...),
    latency_threshold_ms: float = Query(..., gt=0),
    limit: int = Query(2, ge=1, le=50),
) -> TopHistoryResponse:
    """Top-N per category for hardware + model — used by history_loader.

    We synthesize a minimal `HardwareProfile` from query params; the
    repository only inspects (gpus[0].name, gpus[0].vram_total_mb,
    gpu_count, nvlink_available, model_name).
    """
    hardware = HardwareProfile(
        gpus=[GPUInfo(
            index=0,
            name=gpu_name,
            vram_total_mb=gpu_vram_mb,
            vram_free_mb=0,
        )],
        gpu_count=gpu_count,
        nvlink_available=nvlink_available,
        model_name=model_name,
    )
    summaries = await _repo(request).find_top_for_hardware(
        hardware=hardware,
        model_name=model_name,
        latency_threshold_ms=latency_threshold_ms,
        limit=limit,
    )
    return TopHistoryResponse(summaries=summaries)


@router.get("/baseline", response_model=BaselineResponse)
async def get_baseline(
    request: Request,
    gpu_name: str = Query(...),
    gpu_count: int = Query(..., ge=1),
    gpu_vram_mb: int = Query(..., ge=1),
    nvlink_available: bool = Query(...),
    model_name: str = Query(...),
) -> BaselineResponse:
    """The operator-defined baseline for this hardware+model, or null.

    Used by the agent to anchor the planner (and skip re-running an existing
    baseline) and by the dashboard to highlight the reference configuration.
    """
    hardware = HardwareProfile(
        gpus=[GPUInfo(
            index=0,
            name=gpu_name,
            vram_total_mb=gpu_vram_mb,
            vram_free_mb=0,
        )],
        gpu_count=gpu_count,
        nvlink_available=nvlink_available,
        model_name=model_name,
    )
    summary = await _repo(request).find_baseline(
        hardware=hardware,
        model_name=model_name,
    )
    return BaselineResponse(summary=summary)


# ── Dashboard projections ──────────────────────────────────────────────────


def _coerce_row_to_summary(row: dict[str, Any]) -> ExperimentSummaryRow:
    # Pydantic handles type coercion; benchmark_seed comes back as str|None
    # from JSONB ->>, and timestamp as datetime from the ORM.
    return ExperimentSummaryRow(**row)


@router.get("", response_model=list[ExperimentSummaryRow])
async def list_experiments(
    request: Request,
    gpu_name: str | None = Query(None),
    gpu_count: int | None = Query(None),
    gpu_vram_mb: int | None = Query(None),
    nvlink_available: bool | None = Query(None),
    model: list[str] | None = Query(None),
    engine: list[str] | None = Query(None),
    status_filter: list[str] | None = Query(None, alias="status"),
    date_from: datetime | None = Query(None),
    date_to: datetime | None = Query(None),
) -> list[ExperimentSummaryRow]:
    rows = await _repo(request).list_summary_rows(
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        gpu_vram_mb=gpu_vram_mb,
        nvlink_available=nvlink_available,
        models=model,
        engines=engine,
        statuses=status_filter,
        date_from=date_from,
        date_to=date_to,
    )
    return [_coerce_row_to_summary(r) for r in rows]


@router.post("/phases", response_model=list[ExperimentPhaseRow])
async def list_phases(
    body: IdListBody, request: Request
) -> list[ExperimentPhaseRow]:
    rows = await _repo(request).list_phase_rows(body.experiment_ids)
    return [ExperimentPhaseRow(**r) for r in rows]


@router.post("/agentic-turns", response_model=list[AgenticTurnRow])
async def list_agentic_turns(
    body: IdListBody, request: Request
) -> list[AgenticTurnRow]:
    rows = await _repo(request).list_agentic_turn_rows(body.experiment_ids)
    return [AgenticTurnRow(**r) for r in rows]


@router.get("/{experiment_id}", response_model=ExperimentDetailResponse)
async def get_experiment(
    experiment_id: str, request: Request
) -> ExperimentDetailResponse:
    payload = await _repo(request).get_payload(experiment_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    return ExperimentDetailResponse(data=payload)
