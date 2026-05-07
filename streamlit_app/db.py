"""Synchronous DB access for the Streamlit dashboard.

Streamlit is sync-only, so this module uses a separate sync engine via
psycopg + DatabaseConfig.sync_url. ORM models are reused from
`inference_agent.db.models`.

The dashboard never pulls full JSONB payloads in bulk: `list_experiment_summaries`
projects only the scalars needed for the summary dataframe at SQL level,
`list_experiment_phases` expands `concurrency_results` per experiment, and
`get_experiment_payload` loads the full payload lazily for one experiment_id
(used by the Reproduce tab).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, delete, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from inference_agent.db.models import ExperimentRow
from inference_agent.models_pkg.config import DatabaseConfig


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


def _load_db_config() -> DatabaseConfig:
    """Build DatabaseConfig from DATABASE_* env vars (same names as the agent)."""
    raw: dict[str, Any] = {}
    for field_name in (
        "host", "port", "database", "user",
        "password", "password_env",
        "pool_size", "pool_max_overflow", "pool_timeout_sec", "echo",
    ):
        env_name = f"DATABASE_{field_name.upper()}"
        if env_name in os.environ:
            raw[field_name] = os.environ[env_name]
    return DatabaseConfig(**raw)


@st.cache_resource
def get_engine() -> Engine:
    cfg = _load_db_config()
    return create_engine(cfg.sync_url, pool_pre_ping=True)


def _get_sessionmaker():
    return sessionmaker(get_engine(), expire_on_commit=False)


@st.cache_data(ttl=300)
def list_distinct_hardware() -> list[HardwareKey]:
    Session = _get_sessionmaker()
    with Session() as session:
        rows = session.execute(
            select(
                ExperimentRow.gpu_name,
                ExperimentRow.gpu_count,
                ExperimentRow.gpu_vram_mb,
                ExperimentRow.nvlink_available,
            ).distinct()
        ).all()
    return [HardwareKey(*row) for row in rows]


@st.cache_data(ttl=300)
def list_distinct_models() -> list[str]:
    Session = _get_sessionmaker()
    with Session() as session:
        rows = session.execute(
            select(ExperimentRow.model_name).distinct()
        ).all()
    return sorted({row[0] for row in rows})


@st.cache_data(ttl=300)
def list_distinct_engines() -> list[str]:
    Session = _get_sessionmaker()
    with Session() as session:
        rows = session.execute(
            select(ExperimentRow.engine).distinct()
        ).all()
    return sorted({row[0] for row in rows})


# ---- Summary projection ----
#
# This SELECT is the hot path for the dashboard: the eager call that runs on
# every Streamlit rerun. It must NOT pull `data` (the JSONB blob) — that column
# can be megabytes per row when `concurrency_results` has many phases. We
# project just the scalars the summary dataframe needs.
#
# COALESCE/NULLIF defaults mirror the original Python helpers (`_as_float`
# defaults to 0.0, `_as_bool` to False, `_none_label` to "none" / "auto" / "fcfs").
# Array aggregates (gpu_*) are evaluated in SQL via `jsonb_array_elements_text`
# — arrays are tiny (one entry per GPU), so this is cheap.
_SUMMARY_SELECT_SQL = """
SELECT
    experiment_id,
    created_at AS timestamp,
    engine,
    engine_version,
    model_name AS model,
    status,
    correctness_gate_passed,
    gpu_count,
    peak_throughput,
    low_concurrency_ttft_p95 AS ttft_p95,
    max_model_len,
    docker_command,
    docker_image_digest,
    COALESCE(NULLIF(data->>'failure_classification', ''), 'none') AS failure_classification,
    COALESCE(NULLIF(data->>'optimization_classification', ''), 'none') AS classification,
    COALESCE(data->>'llm_commentary', '') AS commentary,
    COALESCE((data->>'duration_seconds')::float, 0) AS duration_s,
    COALESCE((data->>'time_to_healthy_sec')::float, 0) AS time_to_healthy_sec,
    data->>'benchmark_seed' AS benchmark_seed,
    COALESCE((data->'post_benchmark_correctness'->>'basic_chat')::bool, false) AS post_basic_chat,
    COALESCE((data->'config'->>'tensor_parallel_size')::int, 1) AS tp,
    COALESCE((data->'config'->>'pipeline_parallel_size')::int, 1) AS pp,
    COALESCE((data->'config'->>'data_parallel_size')::int, 1) AS dp,
    (data->'config'->>'gpu_memory_utilization')::float AS gpu_memory_utilization,
    (data->'config'->>'mem_fraction_static')::float AS mem_fraction_static,
    (data->'config'->>'max_num_seqs')::int AS max_num_seqs,
    (data->'config'->>'max_running_requests')::int AS max_running_requests,
    (data->'config'->>'max_num_batched_tokens')::int AS max_num_batched_tokens,
    (data->'config'->>'max_prefill_tokens')::int AS max_prefill_tokens,
    COALESCE(NULLIF(data->'config'->>'quantization', ''), 'none') AS quantization,
    COALESCE(NULLIF(data->'config'->>'dtype', ''), 'auto') AS dtype,
    COALESCE(NULLIF(data->'config'->>'kv_cache_dtype', ''), 'auto') AS kv_cache_dtype,
    COALESCE((data->'config'->>'enable_chunked_prefill')::bool, false) AS chunked_prefill,
    COALESCE((data->'config'->>'enable_prefix_caching')::bool, false) AS prefix_caching,
    COALESCE((data->'config'->>'enforce_eager')::bool, false) AS enforce_eager,
    COALESCE(NULLIF(data->'config'->>'scheduling_policy', ''), 'fcfs') AS scheduling_policy,
    COALESCE(NULLIF(data->'config'->>'attention_backend', ''), 'none') AS attention_backend,
    COALESCE(NULLIF(data->'config'->>'speculative_algorithm', ''), 'none') AS speculative_algorithm,
    COALESCE(data->'config'->>'rationale', '') AS rationale,
    COALESCE((data->'benchmark'->>'peak_total_tokens_per_sec')::float, 0) AS peak_total_throughput,
    COALESCE((data->'benchmark'->>'peak_requests_per_sec')::float, 0) AS peak_requests_per_sec,
    COALESCE((data->'benchmark'->>'low_concurrency_tpot_p95_ms')::float, 0) AS tpot_p95,
    COALESCE((data->'benchmark'->>'peak_throughput_e2e_cv')::float, 0) AS peak_throughput_e2e_cv,
    COALESCE((data->'benchmark'->>'low_concurrency_ttft_cv')::float, 0) AS low_concurrency_ttft_cv,
    COALESCE((data->'benchmark'->>'kv_cache_usage_percent')::float, 0) AS kv_cache_usage,
    COALESCE((data->'benchmark'->>'prefix_cache_hit_rate')::float, 0) AS prefix_hit_rate,
    COALESCE((SELECT SUM(v::float) FROM jsonb_array_elements_text(
        data->'benchmark'->'gpu_power_draw_watts') v), 0) AS gpu_power_total_w,
    COALESCE((SELECT MAX(v::float) FROM jsonb_array_elements_text(
        data->'benchmark'->'gpu_memory_used_mb') v), 0) AS gpu_memory_peak_mb,
    COALESCE((SELECT AVG(v::float) FROM jsonb_array_elements_text(
        data->'benchmark'->'gpu_utilization_percent') v), 0) AS gpu_util_avg,
    COALESCE((data->'hardware'->'gpus'->0->>'vram_total_mb')::float, 0) AS gpu_memory_total_mb,
    COALESCE((data->'smoke_tests'->>'basic_chat')::bool, false) AS smoke_basic,
    COALESCE((data->'smoke_tests'->>'tool_calling')::bool, false) AS smoke_tool,
    COALESCE((data->'smoke_tests'->>'tool_required')::bool, false) AS smoke_tool_required,
    COALESCE((data->'smoke_tests'->>'json_mode')::bool, false) AS smoke_json,
    COALESCE((data->'smoke_tests'->>'json_schema')::bool, false) AS smoke_schema,
    COALESCE((data->'scores'->>'throughput_score')::float, 0) AS throughput_score,
    COALESCE((data->'scores'->>'latency_score')::float, 0) AS latency_score,
    COALESCE((data->'scores'->>'balanced_score')::float, 0) AS balanced_score,
    COALESCE((data->'scores'->>'is_pareto_optimal')::bool, false) AS is_pareto
FROM experiments
"""


def _build_summary_filters(filters: Filters) -> tuple[str, dict[str, Any]]:
    """Render the WHERE clause + bind params for `Filters`."""
    clauses: list[str] = []
    params: dict[str, Any] = {}
    if filters.hardware is not None:
        hw = filters.hardware
        clauses.append(
            "gpu_name = :gpu_name AND gpu_count = :gpu_count "
            "AND gpu_vram_mb = :gpu_vram_mb AND nvlink_available = :nvlink_available"
        )
        params.update({
            "gpu_name": hw.gpu_name,
            "gpu_count": hw.gpu_count,
            "gpu_vram_mb": hw.gpu_vram_mb,
            "nvlink_available": hw.nvlink_available,
        })
    if filters.models:
        clauses.append("model_name = ANY(:models)")
        params["models"] = list(filters.models)
    if filters.engines:
        clauses.append("engine = ANY(:engines)")
        params["engines"] = list(filters.engines)
    if filters.statuses:
        clauses.append("status = ANY(:statuses)")
        params["statuses"] = list(filters.statuses)
    if filters.date_from is not None:
        clauses.append("created_at >= :date_from")
        params["date_from"] = filters.date_from
    if filters.date_to is not None:
        clauses.append("created_at <= :date_to")
        params["date_to"] = filters.date_to
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


@st.cache_data(ttl=30)
def list_experiment_summaries(filters: Filters) -> pd.DataFrame:
    """Return a flat summary DataFrame, projected from JSONB at SQL level.

    Derived columns (parallelism, throughput_per_gpu, throughput_per_watt,
    gpu_memory_headroom_mb) are computed in pandas after the fetch — they
    depend only on already-projected scalars.
    """
    where, params = _build_summary_filters(filters)
    sql = text(_SUMMARY_SELECT_SQL + where + " ORDER BY created_at DESC")
    Session = _get_sessionmaker()
    with Session() as session:
        rows = session.execute(sql, params).mappings().all()

    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["parallelism"] = df["tp"] * df["pp"] * df["dp"]
    df["throughput_per_gpu"] = df["peak_throughput"] / df["gpu_count"].clip(lower=1)
    df["throughput_per_watt"] = (
        df["peak_throughput"] / df["gpu_power_total_w"]
    ).where(df["gpu_power_total_w"] > 0, 0.0)
    # Match the original "if memory_total and memory_used else 0" semantics:
    # if either side was missing/empty, headroom is 0, not a phantom positive.
    headroom = (df["gpu_memory_total_mb"] - df["gpu_memory_peak_mb"]).clip(lower=0)
    df["gpu_memory_headroom_mb"] = headroom.where(
        (df["gpu_memory_total_mb"] > 0) & (df["gpu_memory_peak_mb"] > 0),
        0.0,
    )
    return df


# ---- Per-phase projection (concurrency_results) ----
_PHASE_SELECT_SQL = """
SELECT
    e.experiment_id,
    e.engine,
    e.status,
    e.correctness_gate_passed,
    COALESCE(NULLIF(e.data->'config'->>'quantization', ''), 'none') AS quantization,
    COALESCE((e.data->'config'->>'tensor_parallel_size')::int, 1) AS tp,
    COALESCE(NULLIF(p.value->>'workload_id', ''), 'unknown') AS workload_id,
    COALESCE(p.value->>'phase_id', '') AS phase_id,
    COALESCE((p.value->>'concurrency')::int, 0) AS concurrency,
    COALESCE((p.value->>'prompt_length')::int, 0) AS prompt_length,
    COALESCE((p.value->>'max_output_tokens')::int, 0) AS max_output_tokens,
    COALESCE((p.value->>'num_requests')::int, 0) AS num_requests,
    COALESCE((p.value->>'requests_per_sec')::float, 0) AS requests_per_sec,
    COALESCE((p.value->>'output_tokens_per_sec')::float, 0) AS output_tokens_per_sec,
    COALESCE((p.value->>'total_tokens_per_sec')::float, 0) AS total_tokens_per_sec,
    COALESCE((p.value->'ttft_ms'->>'median')::float, 0) AS ttft_p50,
    COALESCE((p.value->'ttft_ms'->>'p95')::float, 0) AS ttft_p95,
    COALESCE((p.value->'ttft_ms'->>'p99')::float, 0) AS ttft_p99,
    COALESCE((p.value->'ttft_ms'->>'cv')::float, 0) AS ttft_cv,
    COALESCE((p.value->'tpot_ms'->>'p95')::float, 0) AS tpot_p95,
    COALESCE((p.value->'e2e_latency_ms'->>'p95')::float, 0) AS e2e_p95,
    COALESCE((p.value->'e2e_latency_ms'->>'cv')::float, 0) AS e2e_cv,
    COALESCE((p.value->>'errors')::int, 0) AS errors,
    COALESCE((p.value->>'error_rate')::float, 0) AS error_rate
FROM experiments e
CROSS JOIN LATERAL jsonb_array_elements(
    COALESCE(e.data->'benchmark'->'concurrency_results', '[]'::jsonb)
) AS p(value)
WHERE e.experiment_id = ANY(:ids)
ORDER BY e.experiment_id,
         (p.value->>'concurrency')::int,
         (p.value->>'prompt_length')::int
"""


@st.cache_data(ttl=30)
def list_experiment_phases(experiment_ids: tuple[str, ...]) -> pd.DataFrame:
    """Return per-phase rows for the given experiments.

    Each row is one entry of `benchmark.concurrency_results`, expanded via
    `jsonb_array_elements`. Avoids shipping the whole JSONB blob.
    """
    if not experiment_ids:
        return pd.DataFrame()
    Session = _get_sessionmaker()
    with Session() as session:
        rows = session.execute(
            text(_PHASE_SELECT_SQL),
            {"ids": list(experiment_ids)},
        ).mappings().all()
    return pd.DataFrame([dict(r) for r in rows])


@st.cache_data(ttl=300)
def get_experiment_payload(experiment_id: str) -> dict | None:
    """Lazy-load the full `ExperimentResult` JSONB for one experiment.

    Used only by the Reproduce tab where the user picks a single experiment;
    the listing path never calls this.
    """
    Session = _get_sessionmaker()
    with Session() as session:
        return session.execute(
            select(ExperimentRow.data).where(
                ExperimentRow.experiment_id == experiment_id,
            )
        ).scalar_one_or_none()


def delete_experiments(experiment_ids: list[str]) -> int:
    """Delete experiments by id and invalidate dashboard caches.

    Returns the number of rows actually deleted. No-op for an empty list.
    """
    if not experiment_ids:
        return 0
    Session = _get_sessionmaker()
    with Session() as session:
        result = session.execute(
            delete(ExperimentRow).where(
                ExperimentRow.experiment_id.in_(experiment_ids)
            )
        )
        session.commit()
        deleted = result.rowcount or 0

    list_experiment_summaries.clear()
    list_experiment_phases.clear()
    get_experiment_payload.clear()
    list_distinct_hardware.clear()
    list_distinct_models.clear()
    list_distinct_engines.clear()
    return deleted
