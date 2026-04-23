"""SGLang Docker engine implementation."""

from __future__ import annotations

import logging

from inference_agent.engines.base import BaseEngine, dedup_flags
from inference_agent.models import AgentConfig, ExperimentConfig

logger = logging.getLogger(__name__)

# Flags automatically managed for NEXTN — strip from extra_engine_args
_NEXTN_AUTO_FLAGS = {
    "--mamba-scheduler-strategy",
    "--speculative-algorithm",
    "--speculative-algo",
    "--speculative-eagle-topk",
    "--speculative-num-draft-tokens",
}


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

        is_nextn = (
            experiment.speculative_algorithm
            and experiment.speculative_algorithm.upper() == "NEXTN"
        )

        # NEXTN speculative decoding needs SGLANG_ENABLE_SPEC_V2=1 env var
        if is_nextn:
            # Only add if not already in extra_env
            if "SGLANG_ENABLE_SPEC_V2" not in experiment.extra_env:
                args.extend(["-e", "SGLANG_ENABLE_SPEC_V2=1"])

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

        # NEXTN + disable-radix-cache requires no_buffer strategy.
        # NEXTN + radix-cache (default) requires extra_buffer.
        # So: if NEXTN is enabled, force the correct mamba strategy
        # and DON'T add --disable-radix-cache when using extra_buffer.
        if is_nextn:
            if not experiment.enable_prefix_caching:
                # User wants radix cache off → must use no_buffer
                serve_args.append("--disable-radix-cache")
                serve_args.extend([
                    "--mamba-scheduler-strategy", "no_buffer",
                ])
                logger.info(
                    "NEXTN + disable-radix-cache: using no_buffer strategy"
                )
            else:
                # Radix cache ON (default for NEXTN) → extra_buffer
                serve_args.extend([
                    "--mamba-scheduler-strategy", "extra_buffer",
                ])
            # Avoid SGLang bug: speculative_eagle_topk must not be None
            serve_args.extend(["--speculative-eagle-topk", "1"])
        else:
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

        # LLM-generated extra args (filter out NEXTN auto-managed flags)
        if experiment.extra_engine_args:
            filtered = _filter_auto_flags(
                experiment.extra_engine_args, _NEXTN_AUTO_FLAGS if is_nextn else set()
            )
            serve_args.extend(filtered)

        # Fixed user-defined args from config
        if self.config.docker.sglang_extra_args:
            serve_args.extend(self.config.docker.sglang_extra_args)

        # Deduplicate flags (extra_engine_args / config may overlap)
        serve_args = dedup_flags(serve_args)

        args.extend(serve_args)
        return args


def _filter_auto_flags(
    extra_args: list[str], auto_flags: set[str]
) -> list[str]:
    """Remove flags from extra_args that are auto-managed by the engine."""
    if not auto_flags:
        return list(extra_args)
    result: list[str] = []
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        if arg in auto_flags:
            # Skip this flag and its value if present
            if i + 1 < len(extra_args) and not extra_args[i + 1].startswith("-"):
                i += 2
            else:
                i += 1
            continue
        result.append(arg)
        i += 1
    return result
