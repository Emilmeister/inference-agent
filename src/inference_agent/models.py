"""Data models for the inference benchmark agent."""

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

    # Speculative decoding
    speculative_algorithm: str | None = None
    speculative_draft_model: str | None = None
    speculative_num_steps: int | None = None

    # SGLang-specific
    num_continuous_decode_steps: int = 1
    dp_size: int | None = None  # sglang data parallelism

    # NOTE: tool_call_parser, reasoning_parser, enable_auto_tool_choice
    # are now managed via docker.vllm_extra_args / sglang_extra_args in config.yaml
    # to avoid duplication.

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
    error_details: list[str] = Field(default_factory=list)


class GPUMetricsSnapshot(BaseModel):
    gpu_index: int
    utilization_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    power_draw_watts: float = 0.0
    temperature_celsius: float = 0.0


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
    tool_calling: bool = False
    tool_calling_detail: str = ""
    json_mode: bool = False
    json_mode_detail: str = ""
    json_schema: bool = False
    json_schema_detail: str = ""


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
    benchmark: BenchmarkResult = Field(default_factory=BenchmarkResult)
    smoke_tests: SmokeTestResult = Field(default_factory=SmokeTestResult)
    llm_commentary: str = ""
    optimization_classification: OptimizationClassification = (
        OptimizationClassification.NONE
    )
    scores: ExperimentScores = Field(default_factory=ExperimentScores)
    docker_command: str = ""  # one-liner docker run command for reproduction
    duration_seconds: float = 0.0


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
    smoke_tests_total: int = 3

    optimization_classification: OptimizationClassification = (
        OptimizationClassification.NONE
    )
    scores: ExperimentScores = Field(default_factory=ExperimentScores)
    llm_commentary: str = ""

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

        smoke_passed = sum([
            result.smoke_tests.tool_calling,
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
            optimization_classification=result.optimization_classification,
            scores=result.scores,
            llm_commentary=result.llm_commentary,
        )


# ── LLM Structured Output Schemas ──────────────────────────────────────────


class PlannerOutput(BaseModel):
    """Schema for the Planner LLM response."""

    engine: str = Field(description="Engine to use: 'vllm' or 'sglang'")
    tensor_parallel_size: int = Field(default=1, description="Tensor parallelism size")
    pipeline_parallel_size: int = Field(default=1, description="Pipeline parallelism size")
    data_parallel_size: int = Field(default=1, description="Data parallelism size")
    max_model_len: int | None = Field(default=None, description="Max model context length override")
    gpu_memory_utilization: float = Field(default=0.9, description="GPU memory fraction (vLLM)")
    mem_fraction_static: float | None = Field(default=None, description="Static memory fraction (SGLang)")
    max_num_seqs: int | None = Field(default=None, description="Max concurrent sequences (vLLM)")
    max_running_requests: int | None = Field(default=None, description="Max running requests (SGLang)")
    max_num_batched_tokens: int | None = Field(default=None, description="Max batched tokens (vLLM)")
    max_prefill_tokens: int | None = Field(default=None, description="Max prefill tokens (SGLang)")
    scheduling_policy: str = Field(default="fcfs", description="Scheduling policy")
    quantization: str | None = Field(default=None, description="Quantization method: fp8, awq, gptq, or null")
    dtype: str = Field(default="auto", description="Data type: auto, float16, bfloat16")
    kv_cache_dtype: str = Field(default="auto", description="KV cache dtype")
    enable_chunked_prefill: bool = Field(default=False, description="Enable chunked prefill")
    chunked_prefill_size: int | None = Field(default=None, description="Chunked prefill size (SGLang)")
    enable_prefix_caching: bool = Field(default=False, description="Enable prefix caching")
    enforce_eager: bool = Field(default=False, description="Disable CUDA graphs (vLLM)")
    speculative_algorithm: str | None = Field(default=None, description="Speculative decoding algorithm")
    speculative_draft_model: str | None = Field(default=None, description="Draft model path")
    speculative_num_steps: int | None = Field(default=None, description="Speculative decode steps")
    num_continuous_decode_steps: int = Field(default=1, description="Continuous decode steps (SGLang)")
    dp_size: int | None = Field(default=None, description="Data parallelism size (SGLang)")
    rationale: str = Field(description="Explanation of why these parameters were chosen")


class AnalyzerOutput(BaseModel):
    """Schema for the Analyzer LLM response."""

    commentary: str = Field(description="2-4 sentence analysis of the experiment across all 3 goals")
    classification: str = Field(description="One of: best_throughput, best_latency, best_balanced, none")
    decision: str = Field(description="One of: continue, stop")
    stop_reason: str | None = Field(default=None, description="Reason for stopping, if decision=stop")
    next_goal: str = Field(description="One of: optimize_throughput, optimize_latency, optimize_balanced, explore")
    planner_hint: str = Field(default="", description="Hint for the planner on what to try next")


# ── Pareto Point ───────────────────────────────────────────────────────────


class ParetoPoint(BaseModel):
    config_id: str
    engine: EngineType
    throughput: float
    ttft_p95: float


# ── Agent Config ───────────────────────────────────────────────────────────


class AgentLLMConfig(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"


class DockerConfig(BaseModel):
    vllm_image: str = "vllm/vllm-openai:latest"
    sglang_image: str = "lmsysorg/sglang:latest"
    network: str = "host"
    shm_size: str = "16g"
    model_cache_dir: str = "/root/.cache/huggingface"

    # Fixed engine flags (not varied by LLM, always applied)
    vllm_extra_args: list[str] = Field(default_factory=list)
    sglang_extra_args: list[str] = Field(default_factory=list)


class BenchmarkConfig(BaseModel):
    warmup_requests: int = 10
    concurrency_levels: list[int] = Field(
        default_factory=lambda: [1, 4, 16, 64, 128, 256, 512]
    )
    prompt_lengths: list[int] = Field(
        default_factory=lambda: [128, 512, 2048, 4096, 32768, 65536, 100000]
    )
    max_output_tokens: int = 256
    long_context_max_output_tokens: int = 8192
    duration_per_level_sec: int = 60
    timeout_sec: int = 600
    latency_threshold_ms: float = 500.0


class ExperimentsConfig(BaseModel):
    max_experiments: int = 30
    plateau_threshold: float = 0.02
    plateau_window: int = 5
    engines: list[EngineType] = Field(
        default_factory=lambda: [EngineType.VLLM, EngineType.SGLANG]
    )


class StorageConfig(BaseModel):
    experiments_dir: str = "./experiments"
    logs_dir: str = "./logs"


class AgentConfig(BaseModel):
    model_name: str = "Qwen/Qwen2.5-72B-Instruct"
    model_revision: str | None = None
    agent_llm: AgentLLMConfig = Field(default_factory=AgentLLMConfig)
    docker: DockerConfig = Field(default_factory=DockerConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    experiments: ExperimentsConfig = Field(default_factory=ExperimentsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    # Natural language instructions for the LLM planner
    # e.g. "Focus on fp8 quantization. Try chunked_prefill_size=4096 with SGLang."
    planner_instructions: str = ""
