"""Configuration models — agent, Docker, benchmark, storage settings."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from inference_agent.models_pkg.domain import EngineType


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


class DockerConfig(BaseModel):
    vllm_image: str = "vllm/vllm-openai:latest"
    sglang_image: str = "lmsysorg/sglang:latest"
    network: str = "host"
    shm_size: str = "16g"

    # HuggingFace cache paths.
    #   host_cache_dir: where the AGENT writes prefetched weights on the host.
    #     Defaults to the current user's HF cache so the agent works without
    #     root. Override if you have a shared cache on a separate volume.
    #   model_cache_dir: where the cache is mounted INSIDE the container. The
    #     stock vLLM / SGLang images run as root and look at /root/.cache.
    # The Docker mount becomes `-v host_cache_dir:model_cache_dir`.
    host_cache_dir: str = Field(
        default_factory=lambda: os.path.expanduser("~/.cache/huggingface")
    )
    model_cache_dir: str = "/root/.cache/huggingface"

    # Fixed engine flags (not varied by LLM, always applied)
    vllm_extra_args: list[str] = Field(default_factory=list)
    sglang_extra_args: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _expand_cache_paths(self) -> "DockerConfig":
        # Allow users to write `~/cache` literally in YAML and have it expand.
        self.host_cache_dir = os.path.expanduser(self.host_cache_dir)
        return self


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
    latency_threshold_ms: float = 500.0
    phase_error_rate_threshold: float = 0.1  # phases with error_rate above this are invalid
    seed: int | None = None  # seed for reproducible prompt generation


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
    hf_token: str | None = None  # HuggingFace token for private models
    agent_llm: AgentLLMConfig = Field(default_factory=AgentLLMConfig)
    docker: DockerConfig = Field(default_factory=DockerConfig)
    startup: StartupConfig = Field(default_factory=StartupConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    experiments: ExperimentsConfig = Field(default_factory=ExperimentsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    # Natural language instructions for the LLM planner
    # e.g. "Focus on fp8 quantization. Try chunked_prefill_size=4096 with SGLang."
    planner_instructions: str = ""
