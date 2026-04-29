"""Unit tests for DB mappers — domain <-> ORM conversions."""

from __future__ import annotations

import pytest

from inference_agent.db.mappers import (
    HeterogeneousClusterError,
    result_to_row,
    row_to_summary,
)
from inference_agent.models import (
    BenchmarkResult,
    EngineType,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    GPUInfo,
    HardwareProfile,
)


def _make_result(
    *,
    gpus: list[GPUInfo] | None = None,
    gpu_count: int = 1,
    nvlink: bool = False,
    docker_command: str = "docker run vllm/vllm-openai",
    docker_args: list[str] | None = None,
) -> ExperimentResult:
    if gpus is None:
        gpus = [GPUInfo(index=0, name="NVIDIA H100", vram_total_mb=81920, vram_free_mb=80000)]
    return ExperimentResult(
        experiment_id="exp_abc",
        engine=EngineType.VLLM,
        engine_version="0.6.0",
        model="Qwen/Qwen2.5-7B-Instruct",
        hardware=HardwareProfile(
            gpus=gpus,
            gpu_count=gpu_count,
            nvlink_available=nvlink,
            model_name="Qwen/Qwen2.5-7B-Instruct",
        ),
        config=ExperimentConfig(engine=EngineType.VLLM, tensor_parallel_size=2),
        status=ExperimentStatus.SUCCESS,
        correctness_gate_passed=True,
        benchmark=BenchmarkResult(
            peak_output_tokens_per_sec=512.5,
            low_concurrency_ttft_p95_ms=120.0,
        ),
        docker_command=docker_command,
        docker_args=docker_args or ["--tensor-parallel-size", "2"],
        docker_image_digest="sha256:abc",
    )


def test_result_to_row_basic_fields():
    result = _make_result(gpu_count=2, nvlink=True)
    row = result_to_row(result)

    assert row.experiment_id == "exp_abc"
    assert row.engine == "vllm"
    assert row.engine_version == "0.6.0"
    assert row.model_name == "Qwen/Qwen2.5-7B-Instruct"
    assert row.gpu_name == "NVIDIA H100"
    assert row.gpu_count == 2
    assert row.gpu_vram_mb == 81920
    assert row.nvlink_available is True
    assert row.status == "success"
    assert row.correctness_gate_passed is True
    assert row.peak_throughput == pytest.approx(512.5)
    assert row.low_concurrency_ttft_p95 == pytest.approx(120.0)
    assert row.docker_command.startswith("docker run")
    assert row.docker_args == ["--tensor-parallel-size", "2"]
    assert row.docker_image_digest == "sha256:abc"
    # Full payload preserved in JSONB
    assert row.data["experiment_id"] == "exp_abc"
    assert row.data["benchmark"]["peak_output_tokens_per_sec"] == pytest.approx(512.5)


def test_result_to_row_rejects_heterogeneous_cluster():
    result = _make_result(
        gpus=[
            GPUInfo(index=0, name="NVIDIA H100", vram_total_mb=81920, vram_free_mb=80000),
            GPUInfo(index=1, name="NVIDIA A100", vram_total_mb=40960, vram_free_mb=40000),
        ],
        gpu_count=2,
    )
    with pytest.raises(HeterogeneousClusterError):
        result_to_row(result)


def test_result_to_row_rejects_mixed_vram():
    result = _make_result(
        gpus=[
            GPUInfo(index=0, name="NVIDIA H100", vram_total_mb=81920, vram_free_mb=80000),
            GPUInfo(index=1, name="NVIDIA H100", vram_total_mb=40960, vram_free_mb=40000),
        ],
        gpu_count=2,
    )
    with pytest.raises(HeterogeneousClusterError):
        result_to_row(result)


def test_result_to_row_rejects_empty_gpus():
    result = _make_result(gpus=[], gpu_count=0)
    with pytest.raises(HeterogeneousClusterError):
        result_to_row(result)


def test_row_to_summary_roundtrip():
    result = _make_result()
    row = result_to_row(result)

    summary = row_to_summary(row)
    assert summary.experiment_id == "exp_abc"
    assert summary.engine == EngineType.VLLM
    assert summary.peak_throughput == pytest.approx(512.5)
    assert summary.low_concurrency_ttft_p95 == pytest.approx(120.0)
    assert summary.correctness_gate_passed is True
