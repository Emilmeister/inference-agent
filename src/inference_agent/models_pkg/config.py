"""Configuration models — agent, Docker, benchmark, storage settings."""

from __future__ import annotations

from pydantic import BaseModel, Field

from inference_agent.models_pkg.domain import EngineType


class AgentLLMConfig(BaseModel):
    model: str = "claude-sonnet-4-6"
    max_budget_usd: float | None = None


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
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    experiments: ExperimentsConfig = Field(default_factory=ExperimentsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    # Natural language instructions for the LLM planner
    # e.g. "Focus on fp8 quantization. Try chunked_prefill_size=4096 with SGLang."
    planner_instructions: str = ""
