"""Tests for experiment config validation."""

from inference_agent.models import EngineType, ExperimentConfig, HardwareProfile, GPUInfo
from inference_agent.nodes.validator import validate_experiment


def _make_hardware(**overrides) -> HardwareProfile:
    defaults = {
        "gpus": [GPUInfo(index=0, name="A100", vram_total_mb=81920, vram_free_mb=80000)],
        "gpu_count": 1,
        "nvlink_available": False,
        "model_name": "test/model",
        "model_max_context": 32768,
        "has_mtp": False,
        "available_engines": [EngineType.VLLM, EngineType.SGLANG],
    }
    defaults.update(overrides)
    return HardwareProfile(**defaults)


def _make_experiment(engine: EngineType = EngineType.VLLM, **overrides) -> ExperimentConfig:
    defaults = {
        "engine": engine,
        "tensor_parallel_size": 1,
        "max_model_len": 4096,
    }
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


class TestValidateExperiment:
    def test_valid_config(self):
        hw = _make_hardware()
        exp = _make_experiment()
        errors = validate_experiment(exp, hw)
        assert errors == []

    def test_tp_exceeds_gpu_count(self):
        hw = _make_hardware(gpu_count=2)
        exp = _make_experiment(tensor_parallel_size=4)
        errors = validate_experiment(exp, hw)
        assert any("tensor_parallel_size=4 exceeds" in e for e in errors)

    def test_tp_not_divisible(self):
        hw = _make_hardware(
            gpus=[
                GPUInfo(index=i, name="A100", vram_total_mb=81920, vram_free_mb=80000)
                for i in range(4)
            ],
            gpu_count=4,
        )
        exp = _make_experiment(tensor_parallel_size=3)
        errors = validate_experiment(exp, hw)
        assert any("does not divide evenly" in e for e in errors)

    def test_max_model_len_exceeds_context(self):
        hw = _make_hardware(model_max_context=8192)
        exp = _make_experiment(max_model_len=16384)
        errors = validate_experiment(exp, hw)
        assert any("exceeds model_max_context" in e for e in errors)

    def test_max_model_len_too_small(self):
        hw = _make_hardware()
        exp = _make_experiment(max_model_len=256)
        errors = validate_experiment(exp, hw)
        assert any("too small" in e for e in errors)

    def test_vllm_invalid_scheduling(self):
        hw = _make_hardware()
        exp = _make_experiment(scheduling_policy="lpm")
        errors = validate_experiment(exp, hw)
        assert any("scheduling_policy" in e for e in errors)

    def test_sglang_invalid_scheduling(self):
        hw = _make_hardware()
        exp = _make_experiment(EngineType.SGLANG, scheduling_policy="priority")
        errors = validate_experiment(exp, hw)
        assert any("scheduling_policy" in e for e in errors)

    def test_cross_engine_params_vllm(self):
        hw = _make_hardware()
        exp = _make_experiment(
            EngineType.VLLM,
            mem_fraction_static=0.8,
            max_running_requests=256,
        )
        errors = validate_experiment(exp, hw)
        assert any("mem_fraction_static" in e for e in errors)
        assert any("max_running_requests" in e for e in errors)

    def test_cross_engine_params_sglang(self):
        hw = _make_hardware()
        exp = _make_experiment(
            EngineType.SGLANG,
            max_num_seqs=256,
            max_num_batched_tokens=4096,
        )
        errors = validate_experiment(exp, hw)
        assert any("max_num_seqs" in e for e in errors)
        assert any("max_num_batched_tokens" in e for e in errors)

    def test_vllm_speculative_needs_draft_model(self):
        hw = _make_hardware()
        exp = _make_experiment(
            speculative_algorithm="some_algo",
            speculative_draft_model=None,
        )
        errors = validate_experiment(exp, hw)
        assert any("speculative_draft_model" in e for e in errors)

    def test_nextn_without_mtp(self):
        hw = _make_hardware(has_mtp=False)
        exp = _make_experiment(
            EngineType.SGLANG,
            speculative_algorithm="NEXTN",
        )
        errors = validate_experiment(exp, hw)
        assert any("MTP" in e for e in errors)

    def test_nextn_with_mtp(self):
        hw = _make_hardware(has_mtp=True)
        exp = _make_experiment(
            EngineType.SGLANG,
            speculative_algorithm="NEXTN",
        )
        errors = validate_experiment(exp, hw)
        assert not any("MTP" in e for e in errors)

    def test_gpu_memory_utilization_bounds(self):
        hw = _make_hardware()
        exp = _make_experiment(gpu_memory_utilization=1.5)
        errors = validate_experiment(exp, hw)
        assert any("gpu_memory_utilization" in e for e in errors)

    def test_engine_not_available(self):
        hw = _make_hardware(available_engines=[EngineType.VLLM])
        exp = _make_experiment(EngineType.SGLANG)
        errors = validate_experiment(exp, hw)
        assert any("not in available_engines" in e for e in errors)

    def test_total_parallelism_exceeds_gpus(self):
        hw = _make_hardware(gpu_count=4)
        exp = _make_experiment(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=2,
        )
        errors = validate_experiment(exp, hw)
        assert any("TP*PP*DP=8 exceeds" in e for e in errors)
