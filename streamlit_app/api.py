"""REST client for the Streamlit dashboard.

The dashboard never talks to Postgres directly — every query goes through
`inference-api`. Public functions retain the same names and DataFrame shapes
as the previous direct-DB module so `streamlit_app/app.py` only had to swap
its import line.

Configuration (env):
  INFERENCE_API_URL    — base URL of the service (e.g. http://localhost:8080)
  INFERENCE_API_TOKEN  — Bearer token
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import requests
import streamlit as st


@dataclass(frozen=True)
class HardwareKey:
    gpu_name: str
    gpu_count: int
    gpu_vram_mb: int
    nvlink_available: bool

    def label(self) -> str:
        nvlink = " NVLink" if self.nvlink_available else ""
        return f"{self.gpu_name} x{self.gpu_count} ({self.gpu_vram_mb}MB){nvlink}"


@dataclass(frozen=True)
class Filters:
    hardware: HardwareKey | None = None
    models: tuple[str, ...] = ()
    engines: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()
    date_from: datetime | None = None
    date_to: datetime | None = None


def _base_url() -> str:
    url = os.environ.get("INFERENCE_API_URL")
    if not url:
        raise RuntimeError(
            "INFERENCE_API_URL is not set — point the dashboard at an "
            "inference-api instance."
        )
    return url.rstrip("/")


def _token() -> str:
    token = os.environ.get("INFERENCE_API_TOKEN")
    if not token:
        raise RuntimeError(
            "INFERENCE_API_TOKEN is not set — the dashboard cannot authenticate."
        )
    return token


@st.cache_resource
def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {_token()}",
        "Accept": "application/json",
    })
    return s


def _get(path: str, params: dict[str, Any] | None = None) -> Any:
    resp = _session().get(f"{_base_url()}{path}", params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, json: Any) -> Any:
    resp = _session().post(f"{_base_url()}{path}", json=json, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _delete(path: str, json: Any) -> Any:
    resp = _session().delete(f"{_base_url()}{path}", json=json, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ── Meta lookups ───────────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def list_distinct_hardware() -> list[HardwareKey]:
    rows = _get("/meta/hardware")
    return [
        HardwareKey(
            gpu_name=r["gpu_name"],
            gpu_count=r["gpu_count"],
            gpu_vram_mb=r["gpu_vram_mb"],
            nvlink_available=r["nvlink_available"],
        )
        for r in rows
    ]


@st.cache_data(ttl=300)
def list_distinct_models() -> list[str]:
    return _get("/meta/models")


@st.cache_data(ttl=300)
def list_distinct_engines() -> list[str]:
    return _get("/meta/engines")


# ── Summaries ──────────────────────────────────────────────────────────────


def _summary_params(filters: Filters) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if filters.hardware is not None:
        params["gpu_name"] = filters.hardware.gpu_name
        params["gpu_count"] = filters.hardware.gpu_count
        params["gpu_vram_mb"] = filters.hardware.gpu_vram_mb
        params["nvlink_available"] = (
            "true" if filters.hardware.nvlink_available else "false"
        )
    if filters.models:
        params["model"] = list(filters.models)
    if filters.engines:
        params["engine"] = list(filters.engines)
    if filters.statuses:
        params["status"] = list(filters.statuses)
    if filters.date_from is not None:
        params["date_from"] = filters.date_from.isoformat()
    if filters.date_to is not None:
        params["date_to"] = filters.date_to.isoformat()
    return params


@st.cache_data(ttl=30)
def list_experiment_summaries(filters: Filters) -> pd.DataFrame:
    """Return the summary DataFrame (one row per experiment) with derived columns."""
    rows = _get("/experiments", params=_summary_params(filters))
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["parallelism"] = df["tp"] * df["pp"] * df["dp"]
    df["throughput_per_gpu"] = df["peak_throughput"] / df["gpu_count"].clip(lower=1)
    df["throughput_per_watt"] = (
        df["peak_throughput"] / df["gpu_power_total_w"]
    ).where(df["gpu_power_total_w"] > 0, 0.0)
    headroom = (df["gpu_memory_total_mb"] - df["gpu_memory_peak_mb"]).clip(lower=0)
    df["gpu_memory_headroom_mb"] = headroom.where(
        (df["gpu_memory_total_mb"] > 0) & (df["gpu_memory_peak_mb"] > 0),
        0.0,
    )
    return df


@st.cache_data(ttl=30)
def list_experiment_phases(experiment_ids: tuple[str, ...]) -> pd.DataFrame:
    if not experiment_ids:
        return pd.DataFrame()
    rows = _post("/experiments/phases", {"experiment_ids": list(experiment_ids)})
    return pd.DataFrame(rows)


@st.cache_data(ttl=30)
def list_agentic_turn_metrics(experiment_ids: tuple[str, ...]) -> pd.DataFrame:
    if not experiment_ids:
        return pd.DataFrame()
    rows = _post("/experiments/agentic-turns", {"experiment_ids": list(experiment_ids)})
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def get_experiment_payload(experiment_id: str) -> dict | None:
    try:
        body = _get(f"/experiments/{experiment_id}")
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return None
        raise
    return body.get("data") if isinstance(body, dict) else None


def delete_experiments(experiment_ids: list[str]) -> int:
    if not experiment_ids:
        return 0
    body = _delete("/experiments", {"experiment_ids": experiment_ids})
    deleted = int(body.get("deleted", 0)) if isinstance(body, dict) else 0

    list_experiment_summaries.clear()
    list_experiment_phases.clear()
    list_agentic_turn_metrics.clear()
    get_experiment_payload.clear()
    list_distinct_hardware.clear()
    list_distinct_models.clear()
    list_distinct_engines.clear()
    return deleted
