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
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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
    # All dashboard endpoints are idempotent (reads + dedup-DELETE), so it is
    # safe to retry GET/POST/DELETE on transient connection failures and
    # retryable 5xx/429 responses. macOS errno 89 ("Operation canceled") slips
    # through here when the OS tears down a half-open socket; without retries
    # the whole tab crashes.
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST", "DELETE"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
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


_SPEC_CFG_RE = re.compile(r"--speculative-config\s+(\{[^{}]*\})")
_SPEC_METHOD_RE = re.compile(r'"(?:method|model)"\s*:\s*"([^"]+)"')
_SPEC_NTOK_RE = re.compile(r'"num_speculative_tokens"\s*:\s*(\d+)')


def _derive_speculative(algo: Any, container_command: Any) -> Any:
    """Resolve the speculative-decoding label for display.

    The structured `speculative_algorithm` field is only set when a config used
    the dedicated speculative fields (`speculative_algorithm` +
    `speculative_draft_model`) — typically planner runs. Baselines and any config
    passing `--speculative-config {...}` through extra_engine_args (notably MTP,
    which has no draft model and can't use the structured slot) leave it
    None/"none", so the projection shows "none" even though speculation is on.
    Fall back to the method declared in the actual launch command. Applied at the
    data layer so EVERY table/selector (Full Comparison, inspector, Impact,
    search) reflects it, not just one view.
    """
    empty = (
        algo is None
        or (isinstance(algo, float) and pd.isna(algo))
        or str(algo).strip().lower() in ("", "none", "null")
    )
    if not empty:
        return algo
    blob = _SPEC_CFG_RE.search(str(container_command or ""))
    if not blob:
        return algo
    method = _SPEC_METHOD_RE.search(blob.group(1))
    if not method:
        return algo
    ntok = _SPEC_NTOK_RE.search(blob.group(1))
    label = method.group(1)
    if ntok:
        label += f" (n={ntok.group(1)})"
    return label


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
    # Backfill speculative_algorithm from the launch command when the structured
    # field is empty (e.g. MTP passed via --speculative-config). One place →
    # every consumer of the summary df shows the real method, not "none".
    if "speculative_algorithm" in df.columns and "container_command" in df.columns:
        df["speculative_algorithm"] = [
            _derive_speculative(a, c)
            for a, c in zip(df["speculative_algorithm"], df["container_command"])
        ]
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


# ── Quality runs (prod-readiness) ──────────────────────────────────────────


@st.cache_data(ttl=30)
def list_quality_runs(
    hardware: HardwareKey | None = None,
    models: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Quality suite runs (so-testing / terminal-bench) for the finalists.

    Filtered by hardware server-side; model filtering is applied client-side so
    a multi-model dashboard selection works against the single-model endpoint.
    """
    params: dict[str, Any] = {}
    if hardware is not None:
        params["gpu_name"] = hardware.gpu_name
        params["gpu_count"] = hardware.gpu_count
        params["gpu_vram_mb"] = hardware.gpu_vram_mb
        params["nvlink_available"] = "true" if hardware.nvlink_available else "false"
    try:
        body = _get("/quality/runs", params=params)
    except requests.HTTPError as exc:
        # An inference-api that predates the quality endpoints returns 404 for
        # the whole route. Degrade to "no runs" instead of crashing the tab —
        # the fix is to update + restart inference-api (migration 0005 adds the
        # quality_runs table on startup).
        if exc.response is not None and exc.response.status_code == 404:
            return pd.DataFrame()
        raise
    runs = body.get("runs", []) if isinstance(body, dict) else []
    df = pd.DataFrame(runs)
    if df.empty:
        return df
    if models and "model_name" in df.columns:
        df = df[df["model_name"].isin(list(models))]
    if "updated_at" in df.columns:
        df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce", utc=True)
    return df


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
