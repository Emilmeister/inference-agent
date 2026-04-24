"""Tests for Docker args building — vLLM and SGLang engines."""

from inference_agent.engines.base import dedup_flags
from inference_agent.engines.sglang import SGLangEngine, _filter_auto_flags
from inference_agent.engines.vllm import VLLMEngine
from inference_agent.models import AgentConfig, EngineType, ExperimentConfig


def _make_config(**overrides) -> AgentConfig:
    defaults = {
        "model_name": "test/model",
        "hf_token": None,
    }
    defaults.update(overrides)
    return AgentConfig(**defaults)


def _make_experiment(engine: EngineType, **overrides) -> ExperimentConfig:
    defaults = {
        "engine": engine,
        "experiment_id": "test123",
        "tensor_parallel_size": 1,
        "max_model_len": 4096,
    }
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


class TestDedupFlags:
    def test_no_duplicates(self):
        args = ["--model", "test", "--port", "8000"]
        assert dedup_flags(args) == args

    def test_dedup_value_flag(self):
        args = ["--model", "test", "--port", "8000", "--model", "other"]
        result = dedup_flags(args)
        assert result == ["--model", "test", "--port", "8000"]

    def test_dedup_boolean_flag(self):
        args = ["--enforce-eager", "--port", "8000", "--enforce-eager"]
        result = dedup_flags(args)
        assert result == ["--enforce-eager", "--port", "8000"]

    def test_non_flag_args_preserved(self):
        args = ["python3", "-m", "sglang", "--port", "8000"]
        assert dedup_flags(args) == args


class TestFilterAutoFlags:
    def test_empty_auto_flags(self):
        extra = ["--some-flag", "value"]
        assert _filter_auto_flags(extra, set()) == extra

    def test_strips_managed_flag_with_value(self):
        extra = ["--mamba-scheduler-strategy", "extra_buffer", "--some-other", "val"]
        result = _filter_auto_flags(extra, {"--mamba-scheduler-strategy"})
        assert result == ["--some-other", "val"]

    def test_strips_managed_boolean_flag(self):
        extra = ["--speculative-algorithm", "--some-other"]
        result = _filter_auto_flags(extra, {"--speculative-algorithm"})
        # "--some-other" starts with "-", so --speculative-algorithm is boolean
        assert result == ["--some-other"]


class TestVLLMEngine:
    def test_basic_args(self):
        config = _make_config()
        engine = VLLMEngine(config)
        experiment = _make_experiment(EngineType.VLLM)
        args = engine.build_docker_args(experiment)

        assert "docker" in args
        assert "run" in args
        assert "--model" in args
        assert "test/model" in args
        assert "--tensor-parallel-size" in args
        assert "--max-model-len" in args

    def test_quantization_flag(self):
        config = _make_config()
        engine = VLLMEngine(config)
        experiment = _make_experiment(EngineType.VLLM, quantization="fp8")
        args = engine.build_docker_args(experiment)

        assert "--quantization" in args
        idx = args.index("--quantization")
        assert args[idx + 1] == "fp8"

    def test_chunked_prefill_flag(self):
        config = _make_config()
        engine = VLLMEngine(config)
        experiment = _make_experiment(EngineType.VLLM, enable_chunked_prefill=True)
        args = engine.build_docker_args(experiment)

        assert "--enable-chunked-prefill" in args

    def test_no_hf_token_env(self):
        config = _make_config(hf_token=None)
        engine = VLLMEngine(config)
        experiment = _make_experiment(EngineType.VLLM)
        args = engine.build_docker_args(experiment)

        # Should not have HF_TOKEN
        env_args = [a for a in args if "HF_TOKEN" in a]
        assert len(env_args) == 0

    def test_hf_token_env(self):
        config = _make_config(hf_token="hf_test123")
        engine = VLLMEngine(config)
        experiment = _make_experiment(EngineType.VLLM)
        args = engine.build_docker_args(experiment)

        assert "HF_TOKEN=hf_test123" in " ".join(args)

    def test_container_name(self):
        config = _make_config()
        engine = VLLMEngine(config)
        experiment = _make_experiment(EngineType.VLLM)
        assert engine.container_name(experiment) == "bench-vllm-test123"


class TestSGLangEngine:
    def test_basic_args(self):
        config = _make_config()
        engine = SGLangEngine(config)
        experiment = _make_experiment(EngineType.SGLANG)
        args = engine.build_docker_args(experiment)

        assert "docker" in args
        assert "sglang.launch_server" in args
        assert "--tp-size" in args
        assert "--enable-metrics" in args

    def test_nextn_env_var(self):
        config = _make_config()
        engine = SGLangEngine(config)
        experiment = _make_experiment(
            EngineType.SGLANG,
            speculative_algorithm="NEXTN",
            speculative_num_steps=3,
            enable_prefix_caching=True,
        )
        args = engine.build_docker_args(experiment)

        assert "SGLANG_ENABLE_SPEC_V2=1" in " ".join(args)
        assert "--speculative-algorithm" in args
        assert "--mamba-scheduler-strategy" in args

    def test_nextn_bumps_mem_fraction(self):
        config = _make_config()
        engine = SGLangEngine(config)
        experiment = _make_experiment(
            EngineType.SGLANG,
            speculative_algorithm="NEXTN",
            mem_fraction_static=0.7,
            enable_prefix_caching=True,
        )
        args = engine.build_docker_args(experiment)

        idx = args.index("--mem-fraction-static")
        assert args[idx + 1] == "0.9"

    def test_container_name(self):
        config = _make_config()
        engine = SGLangEngine(config)
        experiment = _make_experiment(EngineType.SGLANG)
        assert engine.container_name(experiment) == "bench-sglang-test123"

    def test_disable_radix_cache(self):
        config = _make_config()
        engine = SGLangEngine(config)
        experiment = _make_experiment(
            EngineType.SGLANG,
            enable_prefix_caching=False,
        )
        args = engine.build_docker_args(experiment)
        assert "--disable-radix-cache" in args
