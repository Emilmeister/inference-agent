"""Mappers between domain objects (ExperimentResult / ExperimentSummary) and ORM rows."""

from __future__ import annotations

from inference_agent.db.models import ExperimentRow
from inference_agent.models_pkg.domain import (
    ExperimentResult,
    ExperimentSummary,
    HardwareProfile,
)


class HeterogeneousClusterError(ValueError):
    """Raised when GPUs in HardwareProfile have differing name/vram.

    The schema assumes a homogeneous cluster (one row per cluster signature).
    Heterogeneous setups are out of scope for this version.
    """


def _assert_homogeneous(hw: HardwareProfile) -> None:
    if not hw.gpus:
        raise HeterogeneousClusterError("HardwareProfile has empty gpus list")
    first = hw.gpus[0]
    for gpu in hw.gpus[1:]:
        if gpu.name != first.name or gpu.vram_total_mb != first.vram_total_mb:
            raise HeterogeneousClusterError(
                f"Heterogeneous GPU cluster not supported: "
                f"GPU 0 is {first.name} ({first.vram_total_mb}MB) but GPU "
                f"{gpu.index} is {gpu.name} ({gpu.vram_total_mb}MB)"
            )


def result_to_row(result: ExperimentResult) -> ExperimentRow:
    """Build an `ExperimentRow` from a domain `ExperimentResult`.

    Raises `HeterogeneousClusterError` if the cluster's GPUs aren't uniform.
    """
    _assert_homogeneous(result.hardware)
    primary = result.hardware.gpus[0]

    return ExperimentRow(
        experiment_id=result.experiment_id,
        engine=result.engine.value,
        engine_version=result.engine_version,
        model_name=result.model,
        gpu_name=primary.name,
        gpu_count=result.hardware.gpu_count,
        gpu_vram_mb=primary.vram_total_mb,
        nvlink_available=result.hardware.nvlink_available,
        docker_image_digest=result.docker_image_digest,
        docker_command=result.docker_command,
        docker_args=list(result.docker_args),
        status=result.status.value,
        correctness_gate_passed=result.correctness_gate_passed,
        peak_throughput=result.benchmark.peak_output_tokens_per_sec,
        low_concurrency_ttft_p95=result.benchmark.low_concurrency_ttft_p95_ms,
        data=result.model_dump(mode="json"),
    )


def row_to_summary(row: ExperimentRow) -> ExperimentSummary:
    """Reconstruct an `ExperimentSummary` from a stored row.

    Re-validates the JSONB `data` payload as `ExperimentResult` so we get
    the same digest/scores logic as freshly built summaries.
    """
    result = ExperimentResult.model_validate(row.data)
    return ExperimentSummary.from_result(result)
