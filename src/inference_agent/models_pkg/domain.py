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
    # AGENTIC — primary goal: maximize parallel agents under SLO. Replaces
    # THROUGHPUT in new leaderboards but THROUGHPUT stays in the enum so the
    # API can still decode old DB records.
    AGENTIC = "optimize_agentic"
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
    # BEST_AGENTIC — primary classification for the agentic-first ranking.
    # BEST_THROUGHPUT remains in the enum to deserialize historical DB rows.
    BEST_AGENTIC = "best_agentic"
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
    mtp_num_layers: int = 0         # number of native Multi-Token Prediction heads built into the model
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
    # Dispersion of the per-request distribution within the phase. `stdev` is
    # the sample standard deviation; `cv` is the coefficient of variation
    # (stdev/mean) — dimensionless, comparable across metrics with different
    # scales. Use to flag noisy phases where ranking on the mean/p95 alone
    # would be unstable.
    stdev: float = 0.0
    cv: float = 0.0


class AgenticTurnMetric(BaseModel):
    """Per-turn metric for one ход одной agentic-сессии.

    Заполняется только в run_agentic_long_context_phase. Дашборд использует это
    поле, чтобы построить TTFT vs turn_idx box-plot (главный график agentic
    аналитики: turn 0 — cold prefill, turn 1+ — должен быть кэш-хит).
    """
    session_idx: int
    turn_idx: int
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    e2e_latency_ms: float = 0.0
    output_tokens: int = 0
    input_tokens: int = 0  # реальный input этого хода (растёт от хода к ходу)
    error: str | None = None


class ConcurrencyResult(BaseModel):
    concurrency: int
    prompt_length: int
    max_output_tokens: int
    num_requests: int = 0

    # Phase identification
    workload_id: str = ""    # agent_short | throughput | stress | long_context | agentic_long_context
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

    # Per-turn metrics — пустой для не-agentic фаз.
    agentic_turn_metrics: list[AgenticTurnMetric] = Field(default_factory=list)

    # SLO gate (agentic-only; True by default so non-agentic phases never get
    # filtered as "non-viable"). For agentic phases the runner sets viable
    # based on AgenticSLO (TTFT p95, tpot p95, session error_rate, per-turn
    # timeout). `slo_violations` carries human-readable reasons used by the
    # executor to build a CeilingProbeInfo entry.
    viable: bool = True
    slo_violations: list[str] = Field(default_factory=list)


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

    # Noise indicators (coefficient of variation = stdev/mean) for the phases
    # that produced the headline metrics. High cv (>0.5) means the underlying
    # per-request distribution was wide and the headline number is noisy —
    # ranking decisions on that experiment should be discounted accordingly.
    # See _aggregate_benchmark and _compute_scores in the analyzer.
    peak_throughput_e2e_cv: float = 0.0      # cv of e2e_latency at the peak-throughput phase
    low_concurrency_ttft_cv: float = 0.0     # median cv of TTFT across c=1 agent_short phases

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

    # Agentic long-context derived metrics. 0/False, если agentic-фаз не было
    # или ни одна не прошла гейты. См. _aggregate_benchmark в executor.
    #   max_viable_agentic_concurrency — наибольший concurrency, прошедший
    #     error_rate gate + TTFT p95 SLO + E2E p95 SLO. Главное число для
    #     прод-сайзинга: сколько code-агентов параллельно выдержит конфиг.
    #   ceiling_hit — True если max passed == max sweep level: реальный потолок
    #     может быть выше, sweep его не нашёл (включи agentic_concurrency_ceiling_search).
    #   saturation_concurrency — concurrency той agentic-фазы, где
    #     output_tokens_per_sec максимален (точка насыщения throughput).
    #   peak_output_tokens_per_sec — сам пик throughput внутри agentic-фаз
    #     (отдельно от общего peak, т.к. agentic исключён из peak_output_tokens_per_sec).
    max_viable_agentic_concurrency: int = 0
    agentic_concurrency_ceiling_hit: bool = False
    agentic_saturation_concurrency: int = 0
    agentic_peak_output_tokens_per_sec: float = 0.0

    # Latency at the max-viable-agentic-concurrency phase. Lets the analyzer
    # tie-break between two configs with the same max_viable_c on user-facing
    # responsiveness (tpot is the "smoothness" axis, ttft is the cold-start
    # axis). Both default to 0 when no agentic phase passed SLO.
    agentic_tpot_p95_ms: float = 0.0
    agentic_ttft_p95_ms: float = 0.0


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
    stage: str  # prefetch | startup | healthcheck | benchmark_phase | metrics | smoke | cleanup
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class FailedPhaseInfo(BaseModel):
    """Compact view of one benchmark phase that genuinely malfunctioned.

    Only phases that look like real bugs (HTTP 5xx, connection refused,
    parse errors, or any non-agentic phase that broke) land here. Agentic
    ceiling-probe failures (timeouts at high concurrency) go to
    `ceiling_probe_phases` instead — they are EXPECTED outcomes of the
    sweep, not malfunctions. See executor._classify_phase_outcome.

    Surfaced in `ExperimentSummary.failed_phases` so the LLM analyzer/planner
    sees per-phase failure detail (workload, concurrency, sample error text)
    instead of just a concatenated error string. Without this the LLM tends
    to hallucinate "no benchmark errors" when headline metrics look fine.
    """
    phase_id: str
    workload_id: str
    concurrency: int = 0
    prompt_length: int = 0
    error_rate: float = 0.0
    errors: int = 0
    error_sample: str = ""  # first actual error string from runner (timeout, HTTP code, …)


class CeilingProbeInfo(BaseModel):
    """Compact view of one agentic phase that did NOT meet SLO at this concurrency.

    Distinct from `FailedPhaseInfo`: a ceiling probe is the EXPECTED outcome
    of pushing agentic concurrency past what the config can serve while
    holding SLO. Treating it as a malfunction (the prior behavior) leaks
    "5 phase errors" noise into every experiment and makes the LLM
    hallucinate engine-side problems. Routed here when:
      - workload_id == "agentic_long_context", AND
      - errors are dominated by timeout-shaped messages (per-turn or
        session timeout) — the SLO-violation signature.
    """
    phase_id: str
    workload_id: str = "agentic_long_context"
    concurrency: int = 0
    prompt_length: int = 0
    error_rate: float = 0.0
    errors: int = 0
    reason: str = ""  # short human-readable cause: "per-turn timeout", "session timeout", …


# ── Experiment Result ──────────────────────────────────────────────────────


class ExperimentScores(BaseModel):
    # Primary score for the agentic-first goal: how many parallel agents the
    # config sustained under SLO, normalized against the best-known config.
    agentic_score: float = 0.0
    # Throughput score remains for backward compatibility with historical
    # rows and the dashboard "raw throughput" view — it is NO LONGER used for
    # leaderboards or Pareto front; the agentic_score is.
    throughput_score: float = 0.0
    latency_score: float = 0.0
    balanced_score: float = 0.0
    is_pareto_optimal: bool = False
    # Pareto optimality in the agentic axis: (max_viable_agentic_c ↑,
    # agentic_tpot_p95 ↓). Independent of `is_pareto_optimal` so historical
    # records keep working.
    is_agentic_pareto_optimal: bool = False


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
    container_command: str = ""  # one-liner nerdctl run command for reproduction
    container_args: list[str] = Field(default_factory=list)  # full argv for reproduction
    container_image_digest: str = ""  # immutable image digest for reproducibility
    engine_version: str = ""  # engine version string (from /version or --version)
    benchmark_seed: int | None = None  # seed used for prompt generation
    duration_seconds: float = 0.0
    time_to_healthy_sec: float = 0.0  # seconds from container start to healthy
    failure_classification: str | None = None  # startup_crash | healthcheck_timeout | oom | correctness_failure | runtime_crash | benchmark_error
    correctness_gate_passed: bool = False
    post_benchmark_correctness: SmokeTestResult | None = None  # re-check after load

    # Agentic phases that hit SLO at a given concurrency without indicating
    # an engine bug. Informational, NOT counted in `errors` and NOT promoted
    # to status=partial. See executor._classify_phase_outcome.
    ceiling_probe_phases: list[CeilingProbeInfo] = Field(default_factory=list)


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
    # Noise indicators for the phases that produced the headline metrics
    # (see BenchmarkResult.peak_throughput_e2e_cv / low_concurrency_ttft_cv).
    peak_throughput_e2e_cv: float = 0.0
    low_concurrency_ttft_cv: float = 0.0
    smoke_tests_passed: int = 0
    smoke_tests_total: int = 5
    correctness_gate_passed: bool = False
    failure_classification: str | None = None
    error: str | None = None  # error message + container logs for failed experiments

    # Per-phase failures (error-rate gate violations). Populated from
    # ExperimentResult.errors where stage == "benchmark_phase". The LLM uses
    # this to detect chronic systemic failures (e.g. agentic phases failing
    # 100% across many experiments) that headline aggregates hide.
    failed_phases: list[FailedPhaseInfo] = Field(default_factory=list)

    # Agentic ceiling-probe sweep summary. NOT failures — these are SLO-bound
    # probes that simply identify the max concurrency a config can serve while
    # meeting the agentic SLO. Surface separately so the planner doesn't
    # mistake them for engine-side problems.
    agentic_max_viable_concurrency: int = 0
    agentic_peak_throughput: float = 0.0          # total output tok/s at peak agentic phase
    agentic_tpot_p95: float = 0.0                  # tpot p95 ms at max-viable phase
    agentic_ttft_p95: float = 0.0                  # ttft p95 ms at max-viable phase
    agentic_concurrencies_probed: list[int] = Field(default_factory=list)
    agentic_concurrencies_viable: list[int] = Field(default_factory=list)
    agentic_concurrencies_ceiling: list[int] = Field(default_factory=list)

    optimization_classification: OptimizationClassification = (
        OptimizationClassification.NONE
    )
    scores: ExperimentScores = Field(default_factory=ExperimentScores)
    llm_commentary: str = ""
    container_command: str = ""
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
        # full container_command string, which is harder to diff).
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

        # Extract per-phase failures. executor pushes one ExperimentError per
        # phase that breached the error-rate threshold AND was classified as a
        # real malfunction (not an agentic ceiling probe). We project its
        # details into a typed, compact view that the LLM cannot ignore.
        failed_phases: list[FailedPhaseInfo] = []
        for err in result.errors:
            if err.stage != "benchmark_phase":
                continue
            d = err.details or {}
            samples = d.get("error_samples") or []
            sample = str(samples[0])[:240] if samples else err.message[:240]
            failed_phases.append(FailedPhaseInfo(
                phase_id=str(d.get("phase_id", "")),
                workload_id=str(d.get("workload_id", "")),
                concurrency=int(d.get("concurrency") or 0),
                prompt_length=int(d.get("prompt_length") or 0),
                error_rate=float(d.get("error_rate") or 0.0),
                errors=int(d.get("errors") or 0),
                error_sample=sample,
            ))

        # Agentic ceiling-probe view: union of passing agentic phases (in
        # benchmark.concurrency_results) and SLO-bound probes (in
        # ceiling_probe_phases). Lets the LLM see "tried [8,16,32,64,128],
        # viable [8,16], ceiling [32,64,128]" — clean ceiling signal.
        viable_agentic_c = sorted({
            r.concurrency
            for r in result.benchmark.concurrency_results
            if r.workload_id == "agentic_long_context"
        })
        ceiling_agentic_c = sorted({
            p.concurrency for p in result.ceiling_probe_phases
        })
        probed_agentic_c = sorted(set(viable_agentic_c) | set(ceiling_agentic_c))

        return cls(
            experiment_id=result.experiment_id,
            engine=result.engine,
            status=result.status,
            config_digest=digest,
            peak_throughput=result.benchmark.peak_output_tokens_per_sec,
            low_concurrency_ttft_p95=result.benchmark.low_concurrency_ttft_p95_ms,
            low_concurrency_tpot_p95=result.benchmark.low_concurrency_tpot_p95_ms,
            peak_throughput_e2e_cv=result.benchmark.peak_throughput_e2e_cv,
            low_concurrency_ttft_cv=result.benchmark.low_concurrency_ttft_cv,
            smoke_tests_passed=smoke_passed,
            correctness_gate_passed=result.correctness_gate_passed,
            failure_classification=result.failure_classification,
            error=result.error if result.error else None,
            optimization_classification=result.optimization_classification,
            scores=result.scores,
            llm_commentary=result.llm_commentary,
            container_command=result.container_command,
            rationale=result.config.rationale,
            failed_phases=failed_phases,
            agentic_max_viable_concurrency=result.benchmark.max_viable_agentic_concurrency,
            agentic_peak_throughput=result.benchmark.agentic_peak_output_tokens_per_sec,
            agentic_tpot_p95=result.benchmark.agentic_tpot_p95_ms,
            agentic_ttft_p95=result.benchmark.agentic_ttft_p95_ms,
            agentic_concurrencies_probed=probed_agentic_c,
            agentic_concurrencies_viable=viable_agentic_c,
            agentic_concurrencies_ceiling=ceiling_agentic_c,
        )


# ── Pareto Point ───────────────────────────────────────────────────────────


class ParetoPoint(BaseModel):
    """One point on a 2D Pareto front.

    Two fronts coexist in the analyzer:
      - Throughput vs TTFT (`throughput`, `ttft_p95`) — historical/dashboard view.
      - Agentic vs tpot   (`agentic_max_viable_c`, `agentic_tpot_p95`) — the
        primary front for the agentic-first goal. Points on the agentic
        front populate the latter pair; on the throughput front the former.
    The other pair is left at 0 on each point so consumers can detect which
    front the point belongs to without an extra discriminator.
    """
    config_id: str
    engine: EngineType
    throughput: float = 0.0
    ttft_p95: float = 0.0
    agentic_max_viable_c: int = 0
    agentic_tpot_p95: float = 0.0
