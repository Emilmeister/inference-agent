"""Abstract base class for inference engine Docker management."""

from __future__ import annotations

import abc

from inference_agent.models import AgentConfig, ExperimentConfig


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
        if self.config.model_revision:
            args.extend(["-e", f"HF_TOKEN={self.config.model_revision}"])
        return args
