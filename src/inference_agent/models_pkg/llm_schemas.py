"""LLM structured output schemas — adapter-level DTOs for Planner/Analyzer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PlannerOutput(BaseModel):
    """Schema for the Planner LLM response."""

    engine: str = Field(description="Engine to use: 'vllm' or 'sglang'")
    tensor_parallel_size: int = Field(default=1, description="Tensor parallelism size")
    pipeline_parallel_size: int = Field(default=1, description="Pipeline parallelism size")
    data_parallel_size: int = Field(default=1, description="Data parallelism size")
    max_model_len: int = Field(description="Max context length / KV-cache window. Choose ONE of {16384, 32768, 65536, 131072, 262144}, capped at the model's max_context. The OBJECTIVE is to find the LARGEST power-of-2 context that fits VRAM and still benchmarks well — do NOT anchor at 32768 by default. Pick the largest plausibly-fitting value first; halve only after a confirmed OOM.")
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
    attention_backend: str | None = Field(
        default=None,
        description="Attention kernel backend (both engines map to --attention-backend). "
        "SGLang choices include flashinfer, triton, fa3, fa4, flashmla, torch_native, etc. "
        "vLLM accepts FLASHINFER, FLASH_ATTN, XFORMERS, TORCH_SDPA, etc. "
        "Leave null to use the engine default.",
    )
    speculative_algorithm: str | None = Field(default=None, description="Speculative decoding algorithm")
    speculative_draft_model: str | None = Field(default=None, description="Draft model path")
    speculative_num_steps: int | None = Field(default=None, description="Speculative decode steps")
    num_continuous_decode_steps: int = Field(default=1, description="Continuous decode steps (SGLang)")
    dp_size: int | None = Field(default=None, description="Data parallelism size (SGLang)")
    extra_engine_args: list[str] = Field(
        default_factory=list,
        description="Extra CLI flags for the engine that are NOT covered by other fields in this schema. "
        "Each element is a single CLI token, e.g. ['--mamba-scheduler-strategy', 'extra_buffer', '--disable-radix-cache']. "
        "Use ONLY for parameters that have no dedicated field above. Never duplicate flags already set by other fields.",
    )
    extra_env: dict[str, str] = Field(
        default_factory=dict,
        description="Extra environment variables to pass to the container via -e, e.g. {'SGLANG_ENABLE_SPEC_V2': '1'}. "
        "Use ONLY when the engine requires env vars not covered by CLI flags.",
    )
    rationale: str = Field(description="Explanation of why these parameters were chosen")


class AnalyzerOutput(BaseModel):
    """Schema for the Analyzer LLM response."""

    commentary: str = Field(description="2-4 sentence analysis of the experiment across all 3 goals")
    classification: str = Field(description="One of: best_throughput, best_latency, best_balanced, none")
    decision: str = Field(description="One of: continue, stop")
    stop_reason: str | None = Field(default=None, description="Reason for stopping, if decision=stop")
    next_goal: str = Field(description="One of: optimize_throughput, optimize_latency, optimize_balanced, explore")
    planner_hint: str = Field(default="", description="Hint for the planner on what to try next")
