"""Domain models — enums, hardware, experiment, benchmark, scoring."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────


class EngineType(str, Enum):
    VLLM = "vllm"
    SGLANG = "sglang"


class OptimizationGoal(str, Enum):
    THROUGHPUT = "optimize_throughput"
    LATENCY = "optimize_latency"
    BALANCED = "optimize_balanced"
    EXPLORE = "explore"


class ExperimentStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    FAILED_CORRECTNESS = "failed_correctness"


class OptimizationClassification(str, Enum):
    BEST_THROUGHPUT = "best_throughput"
    BEST_LATENCY = "best_latency"
    BEST_BALANCED = "best_balanced"
    NONE = "none"


# ── Hardware ───────────────────────────────────────────────────────────────


class GPUInfo(BaseModel):
    index: int
    name: str
    vram_total_mb: int
    vram_free_mb: int


class HardwareProfile(BaseModel):
    gpus: list[GPUInfo]
    gpu_count: int
    nvlink_available: bool
    model_name: str
    model_size_params: int | None = None
    model_architecture: str | None = None
    model_max_context: int = 4096
    is_vlm: bool = False           # vision-language model (needs --language-model-only for vllm)
    has_mtp: bool = False           # has Multi-Token Prediction layers (SGLang NEXTN)
    available_engines: list[EngineType] = []


class GPUMetricsSnapshot(BaseModel):
    gpu_index: int
    utilization_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    power_draw_watts: float = 0.0
    temperature_celsius: float = 0.0


# ── Experiment Config ──────────────────────────────────────────────────────


class ExperimentConfig(BaseModel):
    engine: EngineType
    experiment_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])

    # Parallelism
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1

    # Context & Memory
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.90  # vllm
    mem_fraction_static: float | None = None  # sglang

    # Batching
    max_num_seqs: int | None = None  # vllm
    max_running_requests: int | None = None  # sglang
    max_num_batched_tokens: int | None = None  # vllm
    max_prefill_tokens: int | None = None  # sglang
    scheduling_policy: str = "fcfs"

    # Quantization
    quantization: str | None = None
    dtype: str = "auto"
    kv_cache_dtype: str = "auto"

    # Features
    enable_chunked_prefill: bool = False
    chunked_prefill_size: int | None = None  # sglang
    enable_prefix_caching: bool = False
    enforce_eager: bool = False  # vllm
    attention_backend: str | None = None  # both engines: --attention-backend

    # Speculative decoding
    speculative_algorithm: str | None = None
    speculative_draft_model: str | None = None
    speculative_num_steps: int | None = None

    # SGLang-specific
    num_continuous_decode_steps: int = 1
    dp_size: int | None = None  # sglang data parallelism

    # LLM-generated extra args (for flags not covered by dedicated fields)
    extra_engine_args: list[str] = Field(default_factory=list)
    extra_env: dict[str, str] = Field(default_factory=dict)

    # LLM rationale
    rationale: str = ""


# ── Benchmark Results ──────────────────────────────────────────────────────


class PercentileStats(BaseModel):
    mean: float = 0.0
    median: float = 0.0
    p75: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    min: float = 0.0
    max: float = 0.0


class ConcurrencyResult(BaseModel):
    concurrency: int
    prompt_length: int
    max_output_tokens: int
    num_requests: int = 0

    # Phase identification
    workload_id: str = ""    # agent_short | throughput | stress | long_context
    phase_id: str = ""       # unique id: e.g. "c1_p512"

    ttft_ms: PercentileStats = Field(default_factory=PercentileStats)
    tpot_ms: PercentileStats = Field(default_factory=PercentileStats)
    itl_ms: PercentileStats = Field(default_factory=PercentileStats)
    e2e_latency_ms: PercentileStats = Field(default_factory=PercentileStats)

    requests_per_sec: float = 0.0
    input_tokens_per_sec: float = 0.0
    output_tokens_per_sec: float = 0.0
    total_tokens_per_sec: float = 0.0

    queue_time_ms: PercentileStats = Field(default_factory=PercentileStats)
    prefill_time_ms: PercentileStats = Field(default_factory=PercentileStats)
    decode_time_ms: PercentileStats = Field(default_factory=PercentileStats)

    errors: int = 0
    error_rate: float = 0.0  # errors / num_requests
    error_details: list[str] = Field(default_factory=list)


class BenchmarkResult(BaseModel):
    # Aggregate timing (across all concurrency levels)
    ttft_ms: PercentileStats = Field(default_factory=PercentileStats)
    tpot_ms: PercentileStats = Field(default_factory=PercentileStats)
    itl_ms: PercentileStats = Field(default_factory=PercentileStats)
    e2e_latency_ms: PercentileStats = Field(default_factory=PercentileStats)

    # Peak throughput (best across concurrency levels)
    peak_requests_per_sec: float = 0.0
    peak_output_tokens_per_sec: float = 0.0
    peak_total_tokens_per_sec: float = 0.0

    # Latency at low concurrency (concurrency=1)
    low_concurrency_ttft_p95_ms: float = 0.0
    low_concurrency_tpot_p95_ms: float = 0.0

    # Queue & scheduling (aggregate)
    queue_time_ms: PercentileStats = Field(default_factory=PercentileStats)
    prefill_time_ms: PercentileStats = Field(default_factory=PercentileStats)
    decode_time_ms: PercentileStats = Field(default_factory=PercentileStats)

    # KV Cache
    kv_cache_usage_percent: float = 0.0
    prefix_cache_hit_rate: float = 0.0

    # GPU metrics (averaged over benchmark duration)
    gpu_utilization_percent: list[float] = Field(default_factory=list)
    gpu_memory_used_mb: list[float] = Field(default_factory=list)
    gpu_power_draw_watts: list[float] = Field(default_factory=list)
    gpu_temperature_celsius: list[float] = Field(default_factory=list)

    # Per-concurrency breakdown
    concurrency_results: list[ConcurrencyResult] = Field(default_factory=list)


# ── Smoke Tests ────────────────────────────────────────────────────────────


class SmokeTestResult(BaseModel):
    basic_chat: bool = False
    basic_chat_detail: str = ""
    tool_calling: bool = False
    tool_calling_detail: str = ""
    tool_required: bool = False
    tool_required_detail: str = ""
    json_mode: bool = False
    json_mode_detail: str = ""
    json_schema: bool = False
    json_schema_detail: str = ""

    @property
    def gate_passed(self) -> bool:
        """Correctness gate: basic_chat AND tool_calling AND json_schema must pass."""
        return self.basic_chat and self.tool_calling and self.json_schema


# ── Experiment Errors ─────────────────────────────────────────────────────


class ExperimentError(BaseModel):
    """Structured error from a specific stage of experiment execution."""
    stage: str  # startup | healthcheck | benchmark_phase | metrics | smoke | cleanup
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


# ── Experiment Result ──────────────────────────────────────────────────────


class ExperimentScores(BaseModel):
    throughput_score: float = 0.0  # normalized 0-1
    latency_score: float = 0.0
    balanced_score: float = 0.0
    is_pareto_optimal: bool = False


class ExperimentResult(BaseModel):
    experiment_id: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    engine: EngineType
    model: str
    hardware: HardwareProfile
    config: ExperimentConfig
    status: ExperimentStatus = ExperimentStatus.SUCCESS
    error: str | None = None
    errors: list[ExperimentError] = Field(default_factory=list)
    benchmark: BenchmarkResult = Field(default_factory=BenchmarkResult)
    smoke_tests: SmokeTestResult = Field(default_factory=SmokeTestResult)
    llm_commentary: str = ""
    optimization_classification: OptimizationClassification = (
        OptimizationClassification.NONE
    )
    scores: ExperimentScores = Field(default_factory=ExperimentScores)
    docker_command: str = ""  # one-liner docker run command for reproduction
    docker_args: list[str] = Field(default_factory=list)  # full argv for reproduction
    docker_image_digest: str = ""  # immutable image digest for reproducibility
    engine_version: str = ""  # engine version string (from /version or --version)
    benchmark_seed: int | None = None  # seed used for prompt generation
    duration_seconds: float = 0.0
    time_to_healthy_sec: float = 0.0  # seconds from container start to healthy
    failure_classification: str | None = None  # startup_crash | healthcheck_timeout | oom | correctness_failure | runtime_crash | benchmark_error
    correctness_gate_passed: bool = False
    post_benchmark_correctness: SmokeTestResult | None = None  # re-check after load


# ── Experiment Summary (compact, for LLM context) ─────────────────────────


class ExperimentSummary(BaseModel):
    experiment_id: str
    engine: EngineType
    status: ExperimentStatus
    config_digest: dict[str, Any] = Field(default_factory=dict)

    # Key metrics
    peak_throughput: float = 0.0
    low_concurrency_ttft_p95: float = 0.0
    low_concurrency_tpot_p95: float = 0.0
    smoke_tests_passed: int = 0
    smoke_tests_total: int = 5
    correctness_gate_passed: bool = False
    failure_classification: str | None = None
    error: str | None = None  # error message + container logs for failed experiments

    optimization_classification: OptimizationClassification = (
        OptimizationClassification.NONE
    )
    scores: ExperimentScores = Field(default_factory=ExperimentScores)
    llm_commentary: str = ""
    docker_command: str = ""
    rationale: str = ""

    @classmethod
    def from_result(cls, result: ExperimentResult) -> ExperimentSummary:
        config = result.config
        digest = {
            "tp": config.tensor_parallel_size,
            "pp": config.pipeline_parallel_size,
            "dp": config.data_parallel_size,
            "max_model_len": config.max_model_len,
            "quantization": config.quantization,
            "dtype": config.dtype,
            "kv_cache_dtype": config.kv_cache_dtype,
            "chunked_prefill": config.enable_chunked_prefill,
            "prefix_caching": config.enable_prefix_caching,
            "enforce_eager": config.enforce_eager,
            "scheduling_policy": config.scheduling_policy,
            "attention_backend": config.attention_backend,
        }
        if config.engine == EngineType.VLLM:
            digest["gpu_mem_util"] = config.gpu_memory_utilization
            digest["max_num_seqs"] = config.max_num_seqs
            digest["max_num_batched_tokens"] = config.max_num_batched_tokens
        else:
            digest["mem_fraction_static"] = config.mem_fraction_static
            digest["max_running_requests"] = config.max_running_requests
            digest["max_prefill_tokens"] = config.max_prefill_tokens
            digest["schedule_policy"] = config.scheduling_policy
            digest["continuous_decode_steps"] = config.num_continuous_decode_steps

        # Surface extra_engine_args / extra_env so the planner can compare
        # tail-flags across runs (otherwise these are only visible via the
        # full docker_command string, which is harder to diff).
        if config.extra_engine_args:
            digest["extra_args"] = list(config.extra_engine_args)
        if config.extra_env:
            digest["extra_env"] = dict(config.extra_env)

        smoke_passed = sum([
            result.smoke_tests.basic_chat,
            result.smoke_tests.tool_calling,
            result.smoke_tests.tool_required,
            result.smoke_tests.json_mode,
            result.smoke_tests.json_schema,
        ])

        return cls(
            experiment_id=result.experiment_id,
            engine=result.engine,
            status=result.status,
            config_digest=digest,
            peak_throughput=result.benchmark.peak_output_tokens_per_sec,
            low_concurrency_ttft_p95=result.benchmark.low_concurrency_ttft_p95_ms,
            low_concurrency_tpot_p95=result.benchmark.low_concurrency_tpot_p95_ms,
            smoke_tests_passed=smoke_passed,
            correctness_gate_passed=result.correctness_gate_passed,
            failure_classification=result.failure_classification,
            error=result.error if result.error else None,
            optimization_classification=result.optimization_classification,
            scores=result.scores,
            llm_commentary=result.llm_commentary,
            docker_command=result.docker_command,
            rationale=result.config.rationale,
        )


# ── Pareto Point ───────────────────────────────────────────────────────────


class ParetoPoint(BaseModel):
    config_id: str
    engine: EngineType
    throughput: float
    ttft_p95: float
