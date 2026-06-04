"""Metadata routes — distinct hardware/models/engines for dashboard filters."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from inference_api.auth import require_bearer
from inference_api.db.repository import ExperimentRepository
from inference_api.schemas import HardwareOption

router = APIRouter(prefix="/meta", dependencies=[Depends(require_bearer)])


def _repo(request: Request) -> ExperimentRepository:
    return request.app.state.repository


@router.get("/hardware", response_model=list[HardwareOption])
async def list_hardware(request: Request) -> list[HardwareOption]:
    rows = await _repo(request).list_distinct_hardware()
    return [HardwareOption(**r) for r in rows]


@router.get("/models", response_model=list[str])
async def list_models(request: Request) -> list[str]:
    return await _repo(request).list_distinct_models()


@router.get("/engines", response_model=list[str])
async def list_engines(request: Request) -> list[str]:
    return await _repo(request).list_distinct_engines()
