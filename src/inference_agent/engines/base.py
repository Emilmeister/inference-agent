"""Abstract base class for inference engine container management (nerdctl)."""

from __future__ import annotations

import abc
import logging

from inference_agent.models import AgentConfig, ExperimentConfig

logger = logging.getLogger(__name__)


def dedup_flags(args: list[str]) -> list[str]:
    """Remove duplicate CLI flags, keeping the first occurrence.

    Handles both value flags (``--flag value``) and boolean flags (``--flag``).
    """
    seen: set[str] = set()
    result: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            if arg in seen:
                # Skip duplicate flag (and its value if present)
                if i + 1 < len(args) and not args[i + 1].startswith("-"):
                    i += 2
                else:
                    i += 1
                continue
            seen.add(arg)
        result.append(arg)
        i += 1
    return result


class BaseEngine(abc.ABC):
    """Interface for managing an inference engine via nerdctl."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def build_container_args(self, experiment: ExperimentConfig) -> list[str]:
        """Build the full `nerdctl run` argument list for this experiment."""

    @abc.abstractmethod
    def container_name(self, experiment: ExperimentConfig) -> str:
        """Return a deterministic container name for this experiment."""

    @abc.abstractmethod
    def image(self) -> str:
        """Return the container image to use."""

    @abc.abstractmethod
    def health_url(self) -> str:
        """Return the URL for the health check endpoint."""

    @abc.abstractmethod
    def metrics_url(self) -> str:
        """Return the URL for the Prometheus /metrics endpoint."""

    @abc.abstractmethod
    def api_base_url(self) -> str:
        """Return the base URL for the OpenAI-compatible API."""

    def default_port(self) -> int:
        return 8000

    def build_common_container_args(self, experiment: ExperimentConfig) -> list[str]:
        """Build container arguments common to all engines."""
        cc = self.config.container
        args = [
            "nerdctl", "run",
            "--name", self.container_name(experiment),
            "--gpus", "all",
            "--shm-size", cc.shm_size,
            "--network", cc.network,
            "-v", f"{cc.host_cache_dir}:{cc.model_cache_dir}",
            "-d",  # detached
            # NOTE: no --rm so we can read logs from crashed containers
        ]
        if self.config.hf_token:
            args.extend(["-e", f"HF_TOKEN={self.config.hf_token}"])
        # LLM-generated env vars
        for key, val in experiment.extra_env.items():
            args.extend(["-e", f"{key}={val}"])
        # Restrict GPU visibility to the configured subset. Placed AFTER
        # extra_env (last -e wins) so a planner-generated CUDA_VISIBLE_DEVICES
        # cannot widen the box past what the operator allocated in config.
        cvd = cc.cuda_visible_devices()
        if cvd is not None:
            args.extend(["-e", f"CUDA_VISIBLE_DEVICES={cvd}"])
        # Force HF offline mode when we've already prefetched the model.
        # Two reasons:
        #   1. Some engines call huggingface_hub APIs (e.g. model_info) on
        #      startup and parse responses through StrictDataclass — when the
        #      upstream config has int literals where a float is expected
        #      (DeepSeekV3-style `routed_scaling_factor: 1`), this crashes the
        #      server before it can read our patched local config.json.
        #   2. HF re-validates cached blobs against expected SHA — our
        #      in-place int→float coercion in discovery breaks the hash, so
        #      online mode would re-download and overwrite the patch.
        # Putting these AFTER extra_env so the LLM cannot accidentally re-enable
        # online mode via planner-generated env vars.
        if self.config.startup.prefetch_model:
            args.extend([
                "-e", "HF_HUB_OFFLINE=1",
                "-e", "TRANSFORMERS_OFFLINE=1",
            ])
        return args
