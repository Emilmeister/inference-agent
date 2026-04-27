"""Tests for ExperimentSummary.from_result config_digest construction."""

from __future__ import annotations

from inference_agent.models import (
    BenchmarkResult,
    EngineType,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    ExperimentSummary,
    GPUInfo,
    HardwareProfile,
    SmokeTestResult,
)


def _make_result(engine: EngineType = EngineType.VLLM, **config_overrides) -> ExperimentResult:
    config = ExperimentConfig(
        engine=engine,
        max_model_len=4096,
        **config_overrides,
    )
    hw = HardwareProfile(
        gpus=[GPUInfo(index=0, name="test", vram_total_mb=80000, vram_free_mb=70000)],
        gpu_count=1,
        nvlink_available=False,
        model_name="test/model",
        model_max_context=8192,
    )
    return ExperimentResult(
        experiment_id="x",
        engine=config.engine,
        model="test/model",
        hardware=hw,
        config=config,
        status=ExperimentStatus.SUCCESS,
        smoke_tests=SmokeTestResult(),
        benchmark=BenchmarkResult(),
    )


class TestDigestCommonFields:
    def test_includes_attention_backend(self):
        digest = ExperimentSummary.from_result(
            _make_result(attention_backend="FLASHINFER")
        ).config_digest
        assert digest["attention_backend"] == "FLASHINFER"

    def test_attention_backend_null_by_default(self):
        digest = ExperimentSummary.from_result(_make_result()).config_digest
        assert digest["attention_backend"] is None


class TestDigestExtraArgs:
    def test_extra_args_surfaced_when_present(self):
        digest = ExperimentSummary.from_result(_make_result(
            extra_engine_args=["--mamba-scheduler-strategy", "extra_buffer"],
        )).config_digest
        assert digest["extra_args"] == ["--mamba-scheduler-strategy", "extra_buffer"]

    def test_extra_args_omitted_when_empty(self):
        digest = ExperimentSummary.from_result(_make_result()).config_digest
        assert "extra_args" not in digest

    def test_extra_env_includes_values(self):
        """Both keys and values surfaced so planner can compare across runs."""
        digest = ExperimentSummary.from_result(_make_result(
            extra_env={"SGLANG_ENABLE_SPEC_V2": "1", "VLLM_USE_V1": "0"},
        )).config_digest
        assert digest["extra_env"] == {"SGLANG_ENABLE_SPEC_V2": "1", "VLLM_USE_V1": "0"}

    def test_extra_env_omitted_when_empty(self):
        digest = ExperimentSummary.from_result(_make_result()).config_digest
        assert "extra_env" not in digest


class TestDigestEngineSpecific:
    def test_vllm_includes_gpu_mem_util(self):
        digest = ExperimentSummary.from_result(
            _make_result(engine=EngineType.VLLM, gpu_memory_utilization=0.85)
        ).config_digest
        assert digest["gpu_mem_util"] == 0.85
        assert "mem_fraction_static" not in digest

    def test_sglang_includes_mem_fraction(self):
        digest = ExperimentSummary.from_result(
            _make_result(engine=EngineType.SGLANG, mem_fraction_static=0.8)
        ).config_digest
        assert digest["mem_fraction_static"] == 0.8
        assert "gpu_mem_util" not in digest
