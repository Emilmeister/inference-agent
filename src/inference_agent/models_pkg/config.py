"""Configuration models — agent, container runtime, benchmark, storage settings."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from inference_agent.models_pkg.domain import EngineType, ExperimentConfig


class AgentLLMConfig(BaseModel):
    """Agent LLM (planner/analyzer) — any OpenAI-compatible Chat Completions endpoint.

    Supports OpenAI, Cloud.ru Foundation Models, Together, OpenRouter, vLLM/SGLang
    OpenAI-compatible servers, etc. Structured output is requested via the
    `response_format` field per the OpenAI spec.
    """

    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"

    # Provide either api_key directly or api_key_env (env var name to read from).
    # api_key takes precedence if both are set.
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"

    temperature: float = 0.0
    max_tokens: int | None = None
    timeout_sec: int = 600

    # "json_schema" — strict structured output (recommended; OpenAI / modern providers)
    # "json_object" — provider only enforces valid JSON; schema is inlined into prompt
    structured_output_mode: Literal["json_schema", "json_object"] = "json_schema"

    max_budget_usd: float | None = None

    @model_validator(mode="after")
    def _resolve_api_key(self) -> "AgentLLMConfig":
        if not self.api_key and self.api_key_env:
            self.api_key = os.environ.get(self.api_key_env)
        return self


class ContainerConfig(BaseModel):
    """nerdctl/containerd runtime settings for engine containers."""

    vllm_image: str = "vllm/vllm-openai:latest"
    sglang_image: str = "lmsysorg/sglang:latest"
    network: str = "host"
    shm_size: str = "16g"

    # Number of GPUs to expose to engine containers, counted from index 0
    # (i.e. GPUs 0..gpu_count-1). None = use every GPU nvidia-smi reports.
    # Lets a shared 8xH100 stand be split: set gpu_count: 4 to benchmark on
    # half the box. Discovery slices HardwareProfile.gpus to this subset, so
    # the planner's tensor_parallel_size constraint, validator divisibility
    # checks, and analyzer all reason about the GPUs we actually expose.
    # We restrict visibility via CUDA_VISIBLE_DEVICES (honored by vLLM and
    # SGLang) rather than nerdctl's ambiguous --gpus device= syntax.
    gpu_count: int | None = Field(default=None, ge=1)

    # HuggingFace cache paths.
    #   host_cache_dir: where the AGENT writes prefetched weights on the host.
    #     MUST point at the `hub/` subdir of an HF cache root — that is what
    #     vLLM / SGLang / transformers all resolve to (HF_HUB_CACHE defaults
    #     to $HF_HOME/hub). If you point this at the cache root itself,
    #     snapshot_download writes to `<root>/models--*` while engines look in
    #     `<root>/hub/models--*`, the files exist on disk but no engine sees
    #     them, and every experiment fails with `Cannot find any model
    #     weights` or `Couldn't instantiate the backend tokenizer` until
    #     someone hand-edits a symlink. Don't relearn that lesson.
    #   model_cache_dir: where the cache is mounted INSIDE the container. The
    #     stock vLLM / SGLang images run as root and HF_HUB_CACHE defaults to
    #     /root/.cache/huggingface/hub — point this at THAT subdir, not at
    #     the cache root, for the same reason as host_cache_dir.
    # The bind mount becomes `-v host_cache_dir:model_cache_dir`, mapping the
    # host's hub subdir to the container's hub subdir.
    host_cache_dir: str = Field(
        default_factory=lambda: os.path.expanduser("~/.cache/huggingface/hub")
    )
    model_cache_dir: str = "/root/.cache/huggingface/hub"

    # Fixed engine flags (not varied by LLM, always applied)
    vllm_extra_args: list[str] = Field(default_factory=list)
    sglang_extra_args: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _expand_cache_paths(self) -> "ContainerConfig":
        # Allow users to write `~/cache` literally in YAML and have it expand.
        self.host_cache_dir = os.path.expanduser(self.host_cache_dir)
        return self

    def cuda_visible_devices(self) -> str | None:
        """Comma-separated device indices to expose, or None for all GPUs.

        Matches the "first N GPUs" semantics of `gpu_count`: returns
        "0,1,...,gpu_count-1". Injected as CUDA_VISIBLE_DEVICES so the engine
        sees exactly that many GPUs, renumbered 0..N-1.
        """
        if self.gpu_count is None:
            return None
        return ",".join(str(i) for i in range(self.gpu_count))


class StartupConfig(BaseModel):
    """Engine startup / healthcheck behavior.

    The total wall time spent waiting for an engine to become healthy is bounded
    by `hard_timeout_sec`. To handle big-model loads that legitimately take long
    but do progress, we ALSO track an idle deadline: if no progress marker shows
    up in container logs for `idle_timeout_sec`, we abort. Idle is reset every
    time a known progress marker appears, so the wait can extend beyond the
    nominal timeout for slow loads, but stalls are still caught quickly.
    """

    hard_timeout_sec: int = 1800        # 30 min — generous for big models
    idle_timeout_sec: int = 300         # 5 min without log progress = stall
    log_scan_interval_sec: float = 10.0  # how often to fetch + scan container logs

    # Timeout for `nerdctl pull` of the engine image when not present locally.
    # Multi-GB engine images on slow links can take 10+ min — bump this if
    # registry is slow or image is large.
    image_pull_timeout_sec: int = 900   # 15 min

    # Timeout for `nerdctl run -d` itself (returns container ID, does NOT wait
    # for healthcheck — that has its own budget above). With NVIDIA runtime,
    # multi-GPU setups, or partially-pulled images, container creation can take
    # well over a minute. Bump this if you see startup_timeout errors despite
    # the image being locally present.
    container_run_timeout_sec: int = 180   # 3 min

    # Pre-download model weights into the host HF cache before launching any
    # container. Eliminates the "first launch takes 15+ min downloading 60GB"
    # failure mode and amortizes download cost across all experiments.
    prefetch_model: bool = True
    prefetch_allow_patterns: list[str] = Field(
        default_factory=lambda: [
            "*.json",
            "*.txt",
            "*.jinja",
            "*.safetensors",
            "*.bin",          # legacy pytorch_model.bin and GPTQ .bin shards
            "tokenizer.model",
            "tokenizer.json",
        ]
    )


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
    latency_threshold_ms: float = 2000.0  # SLO: TTFT p95 budget for balanced goal
    phase_error_rate_threshold: float = 0.1  # phases with error_rate above this are invalid
    seed: int | None = None  # seed for reproducible prompt generation

    # ── Agentic long-context workload — primary optimization target ────────
    # Real code-agent traffic shape: a shared system+tools prefix (REUSED
    # across all parallel sessions via prefix caching) + a small unique user
    # block per session + N multi-turn rounds where each turn appends a
    # synthetic tool_result. Context grows per turn but the bulk of the
    # prompt KV is shared, so the engine can pack many sessions in parallel.
    #
    # The derived headline metric — max_viable_agentic_concurrency — is the
    # largest concurrency at which a phase met ALL of the SLOs below.
    enable_agentic_long_context: bool = False

    # Prompt split — shared prefix is built ONCE per phase with deterministic
    # seed so every session sees the SAME tokens (→ prefix-cache reuse).
    # Unique part is per-session.
    agentic_shared_prefix_tokens: int = 16_000
    agentic_unique_prompt_tokens: int = 8_000

    agentic_max_output_tokens: int = 2_400    # cap per turn; engine usually returns less
    agentic_tool_result_min_tokens: int = 1024
    agentic_tool_result_max_tokens: int = 1536
    agentic_turns_per_session: int = 4

    # Full sweep, no ceiling shortcut: executor walks these in order and stops
    # after two consecutive non-viable phases. Tight grid around H100×8/122B
    # FP8 expectations; bump on bigger hardware.
    agentic_concurrency_levels: list[int] = Field(
        default_factory=lambda: [8, 12, 16, 20, 24, 32, 48]
    )
    agentic_session_timeout_sec: int = 600   # 10 min per session ceiling
    agentic_per_turn_timeout_sec: int = 90   # one HTTP turn budget (was 60s — too tight under prefill burst)

    # SLO gate — phase is `viable` iff ALL of these hold. Tight defaults that
    # match production agent UX: ~3s before first token, ~16 tok/s sustained,
    # <5% session failures. Override per-config when the product SLO differs.
    agentic_ttft_p95_slo_ms: float = 3_000.0
    agentic_tpot_p95_slo_ms: float = 60.0
    agentic_session_error_rate_slo: float = 0.05
    agentic_e2e_p95_slo_ms: float = 0.0       # 0 → auto = 0.8 * session_timeout * 1000

    # Legacy ceiling-search escape hatch — kept for parity, no-op in the new
    # sweep logic (executor walks the configured levels and stops on SLO).
    agentic_concurrency_ceiling_search: bool = False


class ExperimentsConfig(BaseModel):
    max_experiments: int = 30
    plateau_threshold: float = 0.02
    plateau_window: int = 5
    engines: list[EngineType] = Field(
        default_factory=lambda: [EngineType.VLLM, EngineType.SGLANG]
    )

    # How many config parameters the planner may change per experiment relative
    # to its starting point (the baseline anchor or the prior top result). This
    # is a SOFT guideline injected into the planner prompt, not a hard validator
    # constraint — it controls search locality: smaller = more isolated A/B
    # deltas (easier attribution of which knob moved the metric), larger = the
    # planner explores in bigger jumps. Must be >= 1.
    max_params_per_step: int = 2

    @field_validator("max_params_per_step")
    @classmethod
    def _at_least_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_params_per_step must be >= 1")
        return v


class StorageConfig(BaseModel):
    logs_dir: str = "./logs"


class ApiClientConfig(BaseModel):
    """Inference-api REST service endpoint and auth.

    The agent never speaks to Postgres directly — every persistence and
    history query goes through `inference_agent.api_client.ExperimentApiClient`,
    which talks to the FastAPI service (`src/inference_api/`).

    Token may be set directly or via `token_env` (env var name). The agent
    refuses to construct the client without a token.
    """

    base_url: str = "http://localhost:8080"
    token: str | None = None
    token_env: str = "INFERENCE_API_TOKEN"
    timeout_sec: float = 30.0

    @model_validator(mode="after")
    def _resolve_token(self) -> "ApiClientConfig":
        if not self.token and self.token_env:
            self.token = os.environ.get(self.token_env)
        return self


class AgentConfig(BaseModel):
    model_name: str = "Qwen/Qwen2.5-72B-Instruct"
    model_revision: str | None = None
    hf_token: str | None = None  # HuggingFace token for private models

    # Quantization is fixed across all experiments — the planner does not vary
    # it. Set to e.g. "fp8" / "awq" / "gptq", or null to disable.
    quantization: str | None = None

    # max_model_len is fixed across all experiments — the planner does not
    # vary it. Treat as a hardware/product constraint. When None the planner
    # picks per experiment (legacy behavior). When set, the planner is told
    # the value upfront and the validator's agentic oversize gate is skipped
    # (user explicitly chose this context window).
    max_model_len: int | None = None

    # kv_cache_dtype fixed across all experiments — same contract as
    # `quantization`. When set (e.g. "fp8" / "fp8_e4m3" / "fp8_e5m2") the
    # planner cannot vary it: every planned experiment gets this exact value
    # (engine flag forced, recorded config == what actually ran), and the LLM
    # is told it is fixed. None → legacy behavior: the planner defaults per its
    # Rule 0b and may A/B auto vs fp8.
    kv_cache_dtype: str | None = None

    agent_llm: AgentLLMConfig = Field(default_factory=AgentLLMConfig)
    container: ContainerConfig = Field(default_factory=ContainerConfig)
    startup: StartupConfig = Field(default_factory=StartupConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    experiments: ExperimentsConfig = Field(default_factory=ExperimentsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    api: ApiClientConfig = Field(default_factory=ApiClientConfig)

    # Natural language instructions for the LLM planner
    # e.g. "Try chunked_prefill_size=4096 with SGLang."
    planner_instructions: str = ""

    # Operator-defined baseline launch config (loaded from baseline.yaml by the
    # CLI). When set AND no baseline yet exists in the DB for this hardware +
    # model, the agent runs it deterministically as experiment #1 — NOT via the
    # LLM planner — so the benchmark harness measures a real anchor. The planner
    # then iterates ON these measured numbers, and the dashboard highlights the
    # agent's impact relative to them. None → legacy behavior (LLM plans every
    # experiment, including the first "baseline").
    baseline: ExperimentConfig | None = None
