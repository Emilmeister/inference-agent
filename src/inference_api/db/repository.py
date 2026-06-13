"""Async repository — domain ops + dashboard projections.

Two groups of methods:

  * Domain (agent-facing): `insert_experiment`, `find_top_for_hardware`.
  * Dashboard projections: rich JSONB-projecting SELECTs that mirror what the
    Streamlit dashboard used to run directly against Postgres. They live here
    so the dashboard never sees the DB itself — it talks to the REST service
    instead.

All methods are async.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from inference_agent.models_pkg.domain import (
    ExperimentResult,
    ExperimentSummary,
    HardwareProfile,
)
from inference_api.db.mappers import (
    _assert_homogeneous,
    result_to_row,
    row_to_summary,
)
from inference_api.db.models import ExperimentRow

logger = logging.getLogger(__name__)


# ── Dashboard projections ──────────────────────────────────────────────────
#
# Ported verbatim from the original streamlit_app/db.py. JSONB scalars are
# extracted server-side so the dashboard never pulls full payloads in bulk.
# COALESCE/NULLIF defaults reproduce the old Python helper semantics.

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
    container_command,
    container_image_digest,
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
    COALESCE((data->'smoke_tests'->>'json_schema')::bool, false) AS smoke_schema,
    COALESCE((data->'scores'->>'throughput_score')::float, 0) AS throughput_score,
    COALESCE((data->'scores'->>'latency_score')::float, 0) AS latency_score,
    COALESCE((data->'scores'->>'balanced_score')::float, 0) AS balanced_score,
    COALESCE((data->'scores'->>'is_pareto_optimal')::bool, false) AS is_pareto,
    COALESCE((data->'benchmark'->>'max_viable_agentic_concurrency')::int, 0) AS max_viable_agentic_concurrency,
    COALESCE((data->'benchmark'->>'agentic_concurrency_ceiling_hit')::bool, false) AS agentic_ceiling_hit,
    COALESCE((data->'benchmark'->>'agentic_saturation_concurrency')::int, 0) AS agentic_saturation_concurrency,
    COALESCE((data->'benchmark'->>'agentic_peak_output_tokens_per_sec')::float, 0) AS agentic_peak_throughput,
    COALESCE((data->'benchmark'->>'agentic_tpot_p95_ms')::float, 0) AS agentic_tpot_p95,
    COALESCE((data->'benchmark'->>'agentic_ttft_p95_ms')::float, 0) AS agentic_ttft_p95,
    COALESCE((data->'scores'->>'agentic_score')::float, 0) AS agentic_score,
    COALESCE((data->'scores'->>'is_agentic_pareto_optimal')::bool, false) AS is_agentic_pareto,
    is_baseline
FROM experiments
"""


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
    COALESCE((p.value->>'error_rate')::float, 0) AS error_rate,
    COALESCE((p.value->>'viable')::boolean, true) AS viable,
    COALESCE(
        ARRAY(SELECT jsonb_array_elements_text(p.value->'slo_violations')),
        ARRAY[]::text[]
    ) AS slo_violations
FROM experiments e
CROSS JOIN LATERAL jsonb_array_elements(
    COALESCE(e.data->'benchmark'->'concurrency_results', '[]'::jsonb)
) AS p(value)
WHERE e.experiment_id = ANY(:ids)
ORDER BY e.experiment_id,
         (p.value->>'concurrency')::int,
         (p.value->>'prompt_length')::int
"""


_AGENTIC_TURNS_SQL = """
SELECT
    e.experiment_id,
    e.engine,
    COALESCE(NULLIF(e.data->'config'->>'quantization', ''), 'none') AS quantization,
    COALESCE(p.value->>'phase_id', '') AS phase_id,
    COALESCE((p.value->>'concurrency')::int, 0) AS concurrency,
    COALESCE((t.value->>'session_idx')::int, 0) AS session_idx,
    COALESCE((t.value->>'turn_idx')::int, 0) AS turn_idx,
    COALESCE((t.value->>'ttft_ms')::float, 0) AS ttft_ms,
    COALESCE((t.value->>'tpot_ms')::float, 0) AS tpot_ms,
    COALESCE((t.value->>'e2e_latency_ms')::float, 0) AS e2e_latency_ms,
    COALESCE((t.value->>'output_tokens')::int, 0) AS output_tokens,
    COALESCE((t.value->>'input_tokens')::int, 0) AS input_tokens,
    NULLIF(t.value->>'error', '') AS error
FROM experiments e
CROSS JOIN LATERAL jsonb_array_elements(
    COALESCE(e.data->'benchmark'->'concurrency_results', '[]'::jsonb)
) AS p(value)
CROSS JOIN LATERAL jsonb_array_elements(
    COALESCE(p.value->'agentic_turn_metrics', '[]'::jsonb)
) AS t(value)
WHERE e.experiment_id = ANY(:ids)
  AND p.value->>'workload_id' = 'agentic_long_context'
ORDER BY e.experiment_id,
         (p.value->>'concurrency')::int,
         (t.value->>'session_idx')::int,
         (t.value->>'turn_idx')::int
"""


class ExperimentRepository:
    """Persist experiment results, load top-N history, and serve dashboard projections."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]):
        self._sessionmaker = sessionmaker

    # ── Domain ─────────────────────────────────────────────────────────────

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
            & (ExperimentRow.status.in_(["success", "partial"]))
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

    async def find_baseline(
        self,
        hardware: HardwareProfile,
        model_name: str,
    ) -> ExperimentSummary | None:
        """Return the most-recent baseline run for this hardware+model, or None.

        Used by the agent's `find_baseline` lookup (anchor + skip re-running)
        and by the dashboard to highlight the reference configuration.
        """
        _assert_homogeneous(hardware)
        primary = hardware.gpus[0]

        query = (
            select(ExperimentRow)
            .where(
                ExperimentRow.is_baseline.is_(True),
                ExperimentRow.gpu_name == primary.name,
                ExperimentRow.gpu_count == hardware.gpu_count,
                ExperimentRow.gpu_vram_mb == primary.vram_total_mb,
                ExperimentRow.nvlink_available == hardware.nvlink_available,
                ExperimentRow.model_name == model_name,
            )
            .order_by(ExperimentRow.created_at.desc())
            .limit(1)
        )
        async with self._sessionmaker() as session:
            row = (await session.execute(query)).scalars().first()
        return row_to_summary(row) if row is not None else None

    # ── Dashboard projections ──────────────────────────────────────────────

    async def list_summary_rows(
        self,
        *,
        gpu_name: str | None = None,
        gpu_count: int | None = None,
        gpu_vram_mb: int | None = None,
        nvlink_available: bool | None = None,
        models: list[str] | None = None,
        engines: list[str] | None = None,
        statuses: list[str] | None = None,
        date_from: Any | None = None,
        date_to: Any | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if gpu_name is not None and gpu_count is not None and gpu_vram_mb is not None and nvlink_available is not None:
            clauses.append(
                "gpu_name = :gpu_name AND gpu_count = :gpu_count "
                "AND gpu_vram_mb = :gpu_vram_mb AND nvlink_available = :nvlink_available"
            )
            params.update({
                "gpu_name": gpu_name,
                "gpu_count": gpu_count,
                "gpu_vram_mb": gpu_vram_mb,
                "nvlink_available": nvlink_available,
            })
        if models:
            clauses.append("model_name = ANY(:models)")
            params["models"] = list(models)
        if engines:
            clauses.append("engine = ANY(:engines)")
            params["engines"] = list(engines)
        if statuses:
            clauses.append("status = ANY(:statuses)")
            params["statuses"] = list(statuses)
        if date_from is not None:
            clauses.append("created_at >= :date_from")
            params["date_from"] = date_from
        if date_to is not None:
            clauses.append("created_at <= :date_to")
            params["date_to"] = date_to
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = text(_SUMMARY_SELECT_SQL + where + " ORDER BY created_at DESC")
        async with self._sessionmaker() as session:
            rows = (await session.execute(sql, params)).mappings().all()
        return [dict(r) for r in rows]

    async def list_phase_rows(self, experiment_ids: list[str]) -> list[dict[str, Any]]:
        if not experiment_ids:
            return []
        async with self._sessionmaker() as session:
            rows = (
                await session.execute(text(_PHASE_SELECT_SQL), {"ids": experiment_ids})
            ).mappings().all()
        return [dict(r) for r in rows]

    async def list_agentic_turn_rows(self, experiment_ids: list[str]) -> list[dict[str, Any]]:
        if not experiment_ids:
            return []
        async with self._sessionmaker() as session:
            rows = (
                await session.execute(text(_AGENTIC_TURNS_SQL), {"ids": experiment_ids})
            ).mappings().all()
        return [dict(r) for r in rows]

    async def list_distinct_hardware(self) -> list[dict[str, Any]]:
        async with self._sessionmaker() as session:
            rows = (await session.execute(
                select(
                    ExperimentRow.gpu_name,
                    ExperimentRow.gpu_count,
                    ExperimentRow.gpu_vram_mb,
                    ExperimentRow.nvlink_available,
                ).distinct()
            )).all()
        return [
            {
                "gpu_name": r[0],
                "gpu_count": r[1],
                "gpu_vram_mb": r[2],
                "nvlink_available": r[3],
            }
            for r in rows
        ]

    async def list_distinct_models(self) -> list[str]:
        async with self._sessionmaker() as session:
            rows = (await session.execute(select(ExperimentRow.model_name).distinct())).all()
        return sorted({r[0] for r in rows})

    async def list_distinct_engines(self) -> list[str]:
        async with self._sessionmaker() as session:
            rows = (await session.execute(select(ExperimentRow.engine).distinct())).all()
        return sorted({r[0] for r in rows})

    async def get_payload(self, experiment_id: str) -> dict[str, Any] | None:
        async with self._sessionmaker() as session:
            return (await session.execute(
                select(ExperimentRow.data).where(
                    ExperimentRow.experiment_id == experiment_id
                )
            )).scalar_one_or_none()

    async def delete_experiments(self, experiment_ids: list[str]) -> int:
        if not experiment_ids:
            return 0
        async with self._sessionmaker() as session:
            result = await session.execute(
                delete(ExperimentRow).where(
                    ExperimentRow.experiment_id.in_(experiment_ids)
                )
            )
            await session.commit()
            return result.rowcount or 0
