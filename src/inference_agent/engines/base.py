"""Abstract base class for inference engine Docker management."""

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
    """Interface for managing an inference engine in Docker."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def build_docker_args(self, experiment: ExperimentConfig) -> list[str]:
        """Build the full `docker run` argument list for this experiment."""

    @abc.abstractmethod
    def container_name(self, experiment: ExperimentConfig) -> str:
        """Return a deterministic container name for this experiment."""

    @abc.abstractmethod
    def image(self) -> str:
        """Return the Docker image to use."""

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

    def build_common_docker_args(self, experiment: ExperimentConfig) -> list[str]:
        """Build Docker arguments common to all engines."""
        dc = self.config.docker
        args = [
            "docker", "run",
            "--name", self.container_name(experiment),
            "--gpus", "all",
            "--shm-size", dc.shm_size,
            "--network", dc.network,
            "-v", f"{dc.model_cache_dir}:{dc.model_cache_dir}",
            "-d",  # detached
            # NOTE: no --rm so we can read logs from crashed containers
        ]
        if self.config.hf_token:
            args.extend(["-e", f"HF_TOKEN={self.config.hf_token}"])
        # LLM-generated env vars
        for key, val in experiment.extra_env.items():
            args.extend(["-e", f"{key}={val}"])
        return args
