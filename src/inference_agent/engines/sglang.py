"""SGLang Docker engine implementation."""

from __future__ import annotations

from inference_agent.engines.base import BaseEngine
from inference_agent.models import AgentConfig, ExperimentConfig


class SGLangEngine(BaseEngine):
    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)

    def image(self) -> str:
        return self.config.docker.sglang_image

    def container_name(self, experiment: ExperimentConfig) -> str:
        return f"bench-sglang-{experiment.experiment_id}"

    def health_url(self) -> str:
        return f"http://localhost:{self.default_port()}/health"

    def metrics_url(self) -> str:
        return f"http://localhost:{self.default_port()}/metrics"

    def api_base_url(self) -> str:
        return f"http://localhost:{self.default_port()}/v1"

    def build_docker_args(self, experiment: ExperimentConfig) -> list[str]:
        args = self.build_common_docker_args(experiment)
        args.append(self.image())

        # sglang launch_server command
        serve_args = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", self.config.model_name,
            "--host", "0.0.0.0",
            "--port", str(self.default_port()),
            "--tp-size", str(experiment.tensor_parallel_size),
            "--dtype", experiment.dtype,
            "--enable-metrics",
        ]

        if experiment.pipeline_parallel_size > 1:
            serve_args.extend([
                "--pp-size", str(experiment.pipeline_parallel_size),
            ])

        if experiment.dp_size and experiment.dp_size > 1:
            serve_args.extend(["--dp-size", str(experiment.dp_size)])

        if experiment.mem_fraction_static is not None:
            serve_args.extend([
                "--mem-fraction-static",
                str(experiment.mem_fraction_static),
            ])

        if experiment.max_model_len is not None:
            # SGLang uses --context-length
            serve_args.extend(["--context-length", str(experiment.max_model_len)])

        if experiment.max_running_requests is not None:
            serve_args.extend([
                "--max-running-requests",
                str(experiment.max_running_requests),
            ])

        if experiment.max_prefill_tokens is not None:
            serve_args.extend([
                "--max-prefill-tokens",
                str(experiment.max_prefill_tokens),
            ])

        if experiment.quantization:
            serve_args.extend(["--quantization", experiment.quantization])

        if experiment.kv_cache_dtype != "auto":
            serve_args.extend(["--kv-cache-dtype", experiment.kv_cache_dtype])

        if experiment.scheduling_policy != "fcfs":
            serve_args.extend(["--schedule-policy", experiment.scheduling_policy])

        if experiment.enable_chunked_prefill:
            if experiment.chunked_prefill_size is not None:
                serve_args.extend([
                    "--chunked-prefill-size",
                    str(experiment.chunked_prefill_size),
                ])
            else:
                serve_args.extend(["--chunked-prefill-size", "8192"])

        if not experiment.enable_prefix_caching:
            serve_args.append("--disable-radix-cache")

        if experiment.num_continuous_decode_steps > 1:
            serve_args.extend([
                "--num-continuous-decode-steps",
                str(experiment.num_continuous_decode_steps),
            ])

        if experiment.speculative_algorithm:
            serve_args.extend([
                "--speculative-algorithm",
                experiment.speculative_algorithm,
            ])
            if experiment.speculative_draft_model:
                serve_args.extend([
                    "--speculative-draft-model-path",
                    experiment.speculative_draft_model,
                ])
            if experiment.speculative_num_steps:
                serve_args.extend([
                    "--speculative-num-steps",
                    str(experiment.speculative_num_steps),
                ])

        # Extra user-defined args from config
        if self.config.docker.sglang_extra_args:
            serve_args.extend(self.config.docker.sglang_extra_args)

        args.extend(serve_args)
        return args
