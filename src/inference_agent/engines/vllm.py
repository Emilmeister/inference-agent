"""vLLM Docker engine implementation."""

from __future__ import annotations

from inference_agent.engines.base import BaseEngine, dedup_flags
from inference_agent.models import AgentConfig, ExperimentConfig


class VLLMEngine(BaseEngine):
    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)

    def image(self) -> str:
        return self.config.docker.vllm_image

    def container_name(self, experiment: ExperimentConfig) -> str:
        return f"bench-vllm-{experiment.experiment_id}"

    def health_url(self) -> str:
        return f"http://localhost:{self.default_port()}/health"

    def metrics_url(self) -> str:
        return f"http://localhost:{self.default_port()}/metrics"

    def api_base_url(self) -> str:
        return f"http://localhost:{self.default_port()}/v1"

    def build_docker_args(self, experiment: ExperimentConfig) -> list[str]:
        args = self.build_common_docker_args(experiment)
        args.append(self.image())

        # vllm serve command
        serve_args = [
            "--model", self.config.model_name,
            "--host", "0.0.0.0",
            "--port", str(self.default_port()),
            "--tensor-parallel-size", str(experiment.tensor_parallel_size),
            "--dtype", experiment.dtype,
            "--gpu-memory-utilization", str(experiment.gpu_memory_utilization),
        ]

        if experiment.pipeline_parallel_size > 1:
            serve_args.extend([
                "--pipeline-parallel-size",
                str(experiment.pipeline_parallel_size),
            ])

        if experiment.data_parallel_size > 1:
            serve_args.extend([
                "--data-parallel-size",
                str(experiment.data_parallel_size),
            ])

        if experiment.max_model_len is not None:
            serve_args.extend(["--max-model-len", str(experiment.max_model_len)])

        if experiment.max_num_seqs is not None:
            serve_args.extend(["--max-num-seqs", str(experiment.max_num_seqs)])

        if experiment.max_num_batched_tokens is not None:
            serve_args.extend([
                "--max-num-batched-tokens",
                str(experiment.max_num_batched_tokens),
            ])

        if experiment.quantization:
            serve_args.extend(["--quantization", experiment.quantization])

        if experiment.kv_cache_dtype != "auto":
            serve_args.extend(["--kv-cache-dtype", experiment.kv_cache_dtype])

        if experiment.scheduling_policy != "fcfs":
            serve_args.extend(["--scheduling-policy", experiment.scheduling_policy])

        if experiment.enable_chunked_prefill:
            serve_args.append("--enable-chunked-prefill")

        if experiment.enable_prefix_caching:
            serve_args.append("--enable-prefix-caching")

        if experiment.enforce_eager:
            serve_args.append("--enforce-eager")

        if experiment.attention_backend:
            serve_args.extend(["--attention-backend", experiment.attention_backend])

        if (
            experiment.speculative_algorithm
            and experiment.speculative_draft_model
            and experiment.speculative_draft_model.lower() not in ("none", "null", "")
        ):
            # vLLM uses --speculative-config as JSON
            import json
            spec_config = {"model": experiment.speculative_draft_model}
            if experiment.speculative_num_steps:
                spec_config["num_speculative_tokens"] = experiment.speculative_num_steps
            serve_args.extend([
                "--speculative-config",
                json.dumps(spec_config),
            ])

        # LLM-generated extra args
        if experiment.extra_engine_args:
            serve_args.extend(experiment.extra_engine_args)

        # Fixed user-defined args from config (tool parsers, reasoning, etc.)
        if self.config.docker.vllm_extra_args:
            serve_args.extend(self.config.docker.vllm_extra_args)

        # Deduplicate flags (extra_engine_args / config may overlap)
        serve_args = dedup_flags(serve_args)

        args.extend(serve_args)
        return args
