"""Tests for planner pure logic — context estimation, engine forcing, speculative disabling."""

from inference_agent.models import (
    EngineType,
    ExperimentStatus,
    ExperimentSummary,
    GPUInfo,
    HardwareProfile,
)
from inference_agent.models import PlannerOutput
from inference_agent.nodes.planner import (
    _build_experiment_config,
    _estimate_safe_context,
    _get_forced_engine,
    _load_curated_docs,
    _should_disable_speculative,
)


def _make_hardware(
    vram_mb: int = 81559,
    gpu_count: int = 1,
    model_params: int | None = 7_459_569_664,
    max_context: int = 262144,
) -> HardwareProfile:
    gpus = [
        GPUInfo(index=i, name="GPU", vram_total_mb=vram_mb, vram_free_mb=vram_mb - 500)
        for i in range(gpu_count)
    ]
    return HardwareProfile(
        gpus=gpus,
        gpu_count=gpu_count,
        nvlink_available=False,
        model_name="test/model",
        model_size_params=model_params,
        model_max_context=max_context,
        available_engines=[EngineType.VLLM, EngineType.SGLANG],
    )


def _make_summary(
    exp_id: str,
    engine: EngineType = EngineType.VLLM,
    status: ExperimentStatus = ExperimentStatus.SUCCESS,
) -> ExperimentSummary:
    return ExperimentSummary(
        experiment_id=exp_id,
        engine=engine,
        status=status,
        peak_throughput=100.0,
        low_concurrency_ttft_p95=50.0,
    )


# ── _estimate_safe_context ────────────────────────────────────────────────


_BUCKETS = (16384, 32768, 65536, 131072, 262144)


class TestEstimateSafeContext:
    def test_small_model_large_gpu(self):
        """7B model on H100 80GB — should pick a bucket >= 32768."""
        hw = _make_hardware(vram_mb=81559, model_params=7_459_569_664, max_context=262144)
        ctx = _estimate_safe_context(hw)
        assert ctx in _BUCKETS
        assert ctx >= 32768

    def test_large_model_barely_fits(self):
        """70B model on 80GB — barely fits, falls back to floor bucket."""
        hw = _make_hardware(vram_mb=81920, model_params=70_000_000_000, max_context=131072)
        ctx = _estimate_safe_context(hw)
        # 70B*2 ≈ 130GB > 80GB → no bucket fits → smallest (capped at max_context)
        assert ctx == 16384

    def test_medium_model_medium_gpu(self):
        """7B model on 24GB RTX 4090 — moderate context, in bucket set."""
        hw = _make_hardware(vram_mb=24576, model_params=7_000_000_000, max_context=32768)
        ctx = _estimate_safe_context(hw)
        assert ctx in _BUCKETS
        assert ctx <= 32768

    def test_no_gpus(self):
        """No GPUs — falls back to min(32768, max_context)."""
        hw = HardwareProfile(
            gpus=[], gpu_count=0, nvlink_available=False,
            model_name="test", model_max_context=32768,
        )
        ctx = _estimate_safe_context(hw)
        assert ctx == 32768

    def test_no_gpus_small_context_model(self):
        """No GPUs, model max context is small."""
        hw = HardwareProfile(
            gpus=[], gpu_count=0, nvlink_available=False,
            model_name="test", model_max_context=8192,
        )
        ctx = _estimate_safe_context(hw)
        assert ctx == 8192

    def test_unknown_model_size(self):
        """model_size_params=None — assumes half VRAM for model."""
        hw = _make_hardware(vram_mb=81920, model_params=None, max_context=262144)
        ctx = _estimate_safe_context(hw)
        assert ctx in _BUCKETS

    def test_multi_gpu_picks_large_bucket(self):
        """Multiple GPUs — total VRAM summed; should pick a large bucket."""
        hw = _make_hardware(vram_mb=81920, gpu_count=4, model_params=70_000_000_000, max_context=262144)
        ctx = _estimate_safe_context(hw)
        # 4x80GB - 130GB model ≈ 190GB → very large context fits
        assert ctx in _BUCKETS
        assert ctx >= 65536

    def test_returns_power_of_two_bucket(self):
        """Result is always a value from the bucket set (or capped fallback)."""
        hw = _make_hardware(vram_mb=30000, model_params=3_000_000_000, max_context=262144)
        ctx = _estimate_safe_context(hw)
        assert ctx in _BUCKETS

    def test_no_cap_at_65536_when_vram_allows(self):
        """With abundant VRAM, fallback can reach 262144."""
        hw = _make_hardware(vram_mb=200000, model_params=1_000_000_000, max_context=262144)
        ctx = _estimate_safe_context(hw)
        assert ctx == 262144

    def test_respects_model_max_context(self):
        """If model max context is small, don't exceed it."""
        hw = _make_hardware(vram_mb=81920, model_params=1_000_000_000, max_context=4096)
        ctx = _estimate_safe_context(hw)
        assert ctx <= 4096


# ── _get_forced_engine ────────────────────────────────────────────────────


class TestGetForcedEngine:
    def test_no_history(self):
        assert _get_forced_engine([], [EngineType.VLLM, EngineType.SGLANG]) is None

    def test_short_history(self):
        history = [_make_summary("a", EngineType.VLLM)]
        assert _get_forced_engine(history, [EngineType.VLLM, EngineType.SGLANG]) is None

    def test_same_engine_twice_forces_switch(self):
        history = [
            _make_summary("a", EngineType.VLLM),
            _make_summary("b", EngineType.VLLM),
        ]
        forced = _get_forced_engine(history, [EngineType.VLLM, EngineType.SGLANG])
        assert forced == EngineType.SGLANG

    def test_alternating_engines_no_force(self):
        history = [
            _make_summary("a", EngineType.VLLM),
            _make_summary("b", EngineType.SGLANG),
        ]
        assert _get_forced_engine(history, [EngineType.VLLM, EngineType.SGLANG]) is None

    def test_single_engine_available(self):
        """If only one engine available, don't force (can't switch)."""
        history = [
            _make_summary("a", EngineType.VLLM),
            _make_summary("b", EngineType.VLLM),
        ]
        assert _get_forced_engine(history, [EngineType.VLLM]) is None

    def test_sglang_twice_forces_vllm(self):
        history = [
            _make_summary("a", EngineType.SGLANG),
            _make_summary("b", EngineType.SGLANG),
        ]
        forced = _get_forced_engine(history, [EngineType.VLLM, EngineType.SGLANG])
        assert forced == EngineType.VLLM

    def test_only_last_two_matter(self):
        """Earlier history doesn't affect the decision."""
        history = [
            _make_summary("a", EngineType.VLLM),
            _make_summary("b", EngineType.VLLM),
            _make_summary("c", EngineType.SGLANG),
            _make_summary("d", EngineType.VLLM),
        ]
        # Last two are SGLANG, VLLM — different, no force
        assert _get_forced_engine(history, [EngineType.VLLM, EngineType.SGLANG]) is None


# ── _should_disable_speculative ───────────────────────────────────────────


class TestShouldDisableSpeculative:
    def test_no_history(self):
        assert _should_disable_speculative([], EngineType.VLLM) is False

    def test_not_enough_failures(self):
        history = [
            _make_summary("a", EngineType.VLLM, ExperimentStatus.FAILED),
            _make_summary("b", EngineType.VLLM, ExperimentStatus.FAILED),
        ]
        assert _should_disable_speculative(history, EngineType.VLLM, threshold=3) is False

    def test_three_consecutive_failures(self):
        history = [
            _make_summary("a", EngineType.VLLM, ExperimentStatus.FAILED),
            _make_summary("b", EngineType.VLLM, ExperimentStatus.FAILED),
            _make_summary("c", EngineType.VLLM, ExperimentStatus.FAILED),
        ]
        assert _should_disable_speculative(history, EngineType.VLLM, threshold=3) is True

    def test_mixed_results_not_disabled(self):
        history = [
            _make_summary("a", EngineType.VLLM, ExperimentStatus.FAILED),
            _make_summary("b", EngineType.VLLM, ExperimentStatus.SUCCESS),
            _make_summary("c", EngineType.VLLM, ExperimentStatus.FAILED),
        ]
        assert _should_disable_speculative(history, EngineType.VLLM, threshold=3) is False

    def test_only_counts_matching_engine(self):
        history = [
            _make_summary("a", EngineType.SGLANG, ExperimentStatus.FAILED),
            _make_summary("b", EngineType.SGLANG, ExperimentStatus.FAILED),
            _make_summary("c", EngineType.SGLANG, ExperimentStatus.FAILED),
        ]
        # Asking about VLLM, but failures are all SGLang
        assert _should_disable_speculative(history, EngineType.VLLM, threshold=3) is False
        # Asking about SGLang — should disable
        assert _should_disable_speculative(history, EngineType.SGLANG, threshold=3) is True

    def test_success_after_failures_resets(self):
        history = [
            _make_summary("a", EngineType.VLLM, ExperimentStatus.FAILED),
            _make_summary("b", EngineType.VLLM, ExperimentStatus.FAILED),
            _make_summary("c", EngineType.VLLM, ExperimentStatus.FAILED),
            _make_summary("d", EngineType.VLLM, ExperimentStatus.SUCCESS),
        ]
        # Last 3 are [FAILED, FAILED, SUCCESS] — not all failed
        assert _should_disable_speculative(history, EngineType.VLLM, threshold=3) is False


# ── _load_curated_docs ────────────────────────────────────────────────────


class TestBuildExperimentConfigSanitization:
    """LLMs in strict json_schema mode often emit literal 'null' strings or 0
    sentinels for optional fields. _build_experiment_config must absorb those
    so engine builders don't pass `--quantization null` to the engines."""

    def _hw(self) -> HardwareProfile:
        return _make_hardware(gpu_count=4, vram_mb=40960, max_context=262144)

    def _output(self, **overrides) -> PlannerOutput:
        defaults = dict(
            engine="vllm",
            tensor_parallel_size=4,
            max_model_len=32768,
            rationale="test",
        )
        defaults.update(overrides)
        return PlannerOutput(**defaults)

    def test_string_null_quantization_becomes_none(self):
        from inference_agent.models import AgentConfig
        cfg = AgentConfig()
        out = self._output(quantization="null")
        exp = _build_experiment_config(out, self._hw(), cfg)
        assert exp.quantization is None

    def test_string_none_attention_backend_becomes_none(self):
        from inference_agent.models import AgentConfig
        cfg = AgentConfig()
        out = self._output(attention_backend="None")
        exp = _build_experiment_config(out, self._hw(), cfg)
        assert exp.attention_backend is None

    def test_string_null_speculative_fields(self):
        from inference_agent.models import AgentConfig
        cfg = AgentConfig()
        out = self._output(
            speculative_algorithm="null",
            speculative_draft_model="null",
        )
        exp = _build_experiment_config(out, self._hw(), cfg)
        assert exp.speculative_algorithm is None
        assert exp.speculative_draft_model is None

    def test_real_quantization_value_preserved(self):
        from inference_agent.models import AgentConfig
        cfg = AgentConfig()
        out = self._output(quantization="fp8", attention_backend="FLASH_ATTN")
        exp = _build_experiment_config(out, self._hw(), cfg)
        assert exp.quantization == "fp8"
        assert exp.attention_backend == "FLASH_ATTN"

    def test_zero_sentinel_for_optional_numerics(self):
        from inference_agent.models import AgentConfig
        cfg = AgentConfig()
        out = self._output(
            engine="sglang",
            mem_fraction_static=0.0,
            max_running_requests=0,
        )
        exp = _build_experiment_config(out, self._hw(), cfg)
        assert exp.mem_fraction_static is None
        assert exp.max_running_requests is None

    def test_cross_engine_fields_stripped_for_vllm(self):
        from inference_agent.models import AgentConfig
        cfg = AgentConfig()
        out = self._output(
            engine="vllm",
            mem_fraction_static=0.85,
            max_running_requests=128,
        )
        exp = _build_experiment_config(out, self._hw(), cfg)
        assert exp.mem_fraction_static is None
        assert exp.max_running_requests is None

    def test_cross_engine_fields_stripped_for_sglang(self):
        from inference_agent.models import AgentConfig
        cfg = AgentConfig()
        out = self._output(
            engine="sglang",
            max_num_seqs=256,
            max_num_batched_tokens=8192,
        )
        exp = _build_experiment_config(out, self._hw(), cfg)
        assert exp.max_num_seqs is None
        assert exp.max_num_batched_tokens is None

    def test_dtype_null_falls_back_to_auto(self):
        from inference_agent.models import AgentConfig
        cfg = AgentConfig()
        out = self._output(dtype="null", kv_cache_dtype="None")
        exp = _build_experiment_config(out, self._hw(), cfg)
        assert exp.dtype == "auto"
        assert exp.kv_cache_dtype == "auto"


class TestLoadCuratedDocs:
    def test_vllm_docs_loadable(self):
        docs = _load_curated_docs(EngineType.VLLM)
        assert "vLLM CLI parameters" in docs
        assert "## ParallelConfig" in docs
        # Sanity-check key flags the planner relies on are present
        assert "--gpu-memory-utilization" in docs or "--gpu-memory-util" in docs

    def test_sglang_docs_loadable(self):
        docs = _load_curated_docs(EngineType.SGLANG)
        assert "SGLang CLI parameters" in docs
        assert "## Memory and scheduling" in docs
        assert "--mem-fraction-static" in docs

    def test_omitted_sections_absent(self):
        """Curated docs strip frontend/SSL/multi-node noise."""
        sglang = _load_curated_docs(EngineType.SGLANG)
        assert "## HTTP server" not in sglang
        assert "## LoRA" not in sglang
        assert "## Multi-node" not in sglang

        vllm = _load_curated_docs(EngineType.VLLM)
        assert "## Frontend" not in vllm
        assert "## LoRAConfig" not in vllm
        assert "## ObservabilityConfig" not in vllm
