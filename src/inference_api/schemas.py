"""Request/response schemas for the REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from inference_agent.models_pkg.domain import (
    ExperimentResult,
    ExperimentSummary,
)


class CreateExperimentResponse(BaseModel):
    experiment_id: str


class TopHistoryQuery(BaseModel):
    """Hardware + model filter for the history_loader query.

    Mirrors `ExperimentRepository.find_top_for_hardware` arguments but sent as
    a flat query rather than a full `HardwareProfile` (the repository only
    looks at the primary GPU's name/vram + gpu_count + nvlink).
    """

    gpu_name: str
    gpu_count: int
    gpu_vram_mb: int
    nvlink_available: bool
    model_name: str
    latency_threshold_ms: float
    limit: int = 2


class TopHistoryResponse(BaseModel):
    summaries: list[ExperimentSummary]


class BaselineResponse(BaseModel):
    """The operator-defined baseline for a hardware+model, or null if none."""

    summary: ExperimentSummary | None = None


class ExperimentSummaryRow(BaseModel):
    """Compact dashboard projection — one row per experiment.

    Mirrors the previous Streamlit SQL projection so the dashboard can swap
    direct DB access for HTTP without reshaping its dataframe code.
    """

    experiment_id: str
    timestamp: datetime | None = None
    engine: str
    engine_version: str = ""
    model: str
    status: str
    correctness_gate_passed: bool = False
    gpu_count: int = 0
    peak_throughput: float = 0.0
    ttft_p95: float = 0.0
    max_model_len: int = 0
    container_command: str = ""
    container_image_digest: str = ""

    failure_classification: str = "none"
    classification: str = "none"
    commentary: str = ""
    duration_s: float = 0.0
    time_to_healthy_sec: float = 0.0
    benchmark_seed: str | None = None
    post_basic_chat: bool = False

    tp: int = 1
    pp: int = 1
    dp: int = 1
    gpu_memory_utilization: float | None = None
    mem_fraction_static: float | None = None
    max_num_seqs: int | None = None
    max_running_requests: int | None = None
    max_num_batched_tokens: int | None = None
    max_prefill_tokens: int | None = None
    quantization: str = "none"
    dtype: str = "auto"
    kv_cache_dtype: str = "auto"
    chunked_prefill: bool = False
    prefix_caching: bool = False
    enforce_eager: bool = False
    scheduling_policy: str = "fcfs"
    attention_backend: str = "none"
    speculative_algorithm: str = "none"
    rationale: str = ""

    peak_total_throughput: float = 0.0
    peak_requests_per_sec: float = 0.0
    tpot_p95: float = 0.0
    peak_throughput_e2e_cv: float = 0.0
    low_concurrency_ttft_cv: float = 0.0
    kv_cache_usage: float = 0.0
    prefix_hit_rate: float = 0.0

    gpu_power_total_w: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    gpu_util_avg: float = 0.0
    gpu_memory_total_mb: float = 0.0

    smoke_basic: bool = False
    smoke_tool: bool = False
    smoke_tool_required: bool = False
    smoke_schema: bool = False

    throughput_score: float = 0.0
    latency_score: float = 0.0
    balanced_score: float = 0.0
    is_pareto: bool = False

    max_viable_agentic_concurrency: int = 0
    agentic_ceiling_hit: bool = False
    agentic_saturation_concurrency: int = 0
    agentic_peak_throughput: float = 0.0
    agentic_tpot_p95: float = 0.0
    agentic_ttft_p95: float = 0.0
    agentic_score: float = 0.0
    is_agentic_pareto: bool = False

    is_baseline: bool = False


class ExperimentPhaseRow(BaseModel):
    experiment_id: str
    engine: str
    status: str
    correctness_gate_passed: bool = False
    quantization: str = "none"
    tp: int = 1
    workload_id: str = "unknown"
    phase_id: str = ""
    concurrency: int = 0
    prompt_length: int = 0
    max_output_tokens: int = 0
    num_requests: int = 0
    requests_per_sec: float = 0.0
    output_tokens_per_sec: float = 0.0
    total_tokens_per_sec: float = 0.0
    cached_tokens_per_sec: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0
    ttft_cv: float = 0.0
    tpot_p95: float = 0.0
    e2e_p95: float = 0.0
    e2e_cv: float = 0.0
    errors: int = 0
    error_rate: float = 0.0
    # False for agentic phases that breached the SLO (the ceiling level). These
    # are now persisted with full metrics so the dashboard can plot them, marked
    # distinctly from viable phases.
    viable: bool = True
    slo_violations: list[str] = Field(default_factory=list)


class AgenticTurnRow(BaseModel):
    experiment_id: str
    engine: str
    quantization: str = "none"
    phase_id: str = ""
    concurrency: int = 0
    session_idx: int = 0
    turn_idx: int = 0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    e2e_latency_ms: float = 0.0
    output_tokens: int = 0
    input_tokens: int = 0
    error: str | None = None


class HardwareOption(BaseModel):
    gpu_name: str
    gpu_count: int
    gpu_vram_mb: int
    nvlink_available: bool


class IdListBody(BaseModel):
    experiment_ids: list[str] = Field(default_factory=list)


class DeleteResponse(BaseModel):
    deleted: int


class ExperimentDetailResponse(BaseModel):
    """Full ExperimentResult payload — used by the dashboard Reproduce tab."""

    data: dict[str, Any]


# Re-export so route signatures read cleanly.
__all__ = [
    "AgenticTurnRow",
    "BaselineResponse",
    "CreateExperimentResponse",
    "DeleteResponse",
    "ExperimentDetailResponse",
    "ExperimentPhaseRow",
    "ExperimentResult",
    "ExperimentSummary",
    "ExperimentSummaryRow",
    "HardwareOption",
    "IdListBody",
    "TopHistoryQuery",
    "TopHistoryResponse",
]
