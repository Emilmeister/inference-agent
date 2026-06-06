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
        "mtp_num_layers": 0,
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
            speculative_algorithm="eagle",
            speculative_draft_model=None,
        )
        errors = validate_experiment(exp, hw)
        assert any("speculative_draft_model" in e for e in errors)

    def test_vllm_ngram_does_not_need_draft_model(self):
        hw = _make_hardware()
        exp = _make_experiment(
            speculative_algorithm="ngram",
            speculative_draft_model=None,
        )
        errors = validate_experiment(exp, hw)
        assert not any("speculative_draft_model" in e for e in errors)

    def test_vllm_self_speculation_with_native_mtp(self):
        # mtp/eagle3 can self-speculate on a model with native MTP heads,
        # so no draft model is required.
        hw = _make_hardware(mtp_num_layers=1)
        for algo in ("mtp", "eagle3"):
            exp = _make_experiment(
                speculative_algorithm=algo,
                speculative_draft_model=None,
            )
            errors = validate_experiment(exp, hw)
            assert not any("speculative_draft_model" in e for e in errors), (
                f"{algo} on MTP model should not require a draft model: {errors}"
            )

    def test_vllm_self_speculation_without_mtp_still_needs_draft(self):
        hw = _make_hardware(mtp_num_layers=0)
        exp = _make_experiment(
            speculative_algorithm="eagle3",
            speculative_draft_model=None,
        )
        errors = validate_experiment(exp, hw)
        assert any("speculative_draft_model" in e for e in errors)

    def test_nextn_without_mtp(self):
        hw = _make_hardware(mtp_num_layers=0)
        exp = _make_experiment(
            EngineType.SGLANG,
            speculative_algorithm="NEXTN",
        )
        errors = validate_experiment(exp, hw)
        assert any("MTP" in e for e in errors)

    def test_nextn_with_mtp(self):
        hw = _make_hardware(mtp_num_layers=1)
        exp = _make_experiment(
            EngineType.SGLANG,
            speculative_algorithm="NEXTN",
        )
        errors = validate_experiment(exp, hw)
        assert not any("MTP" in e for e in errors)

    def test_speculative_num_steps_exceeds_mtp_heads(self):
        hw = _make_hardware(mtp_num_layers=1)
        exp = _make_experiment(
            EngineType.SGLANG,
            speculative_algorithm="NEXTN",
            speculative_num_steps=4,
        )
        errors = validate_experiment(exp, hw)
        assert any("native MTP head count" in e for e in errors)

    def test_speculative_num_steps_within_mtp_heads(self):
        hw = _make_hardware(mtp_num_layers=3)
        exp = _make_experiment(
            EngineType.SGLANG,
            speculative_algorithm="NEXTN",
            speculative_num_steps=2,
        )
        errors = validate_experiment(exp, hw)
        assert not any("native MTP head count" in e for e in errors)

    def test_speculative_num_steps_no_cap_with_external_draft(self):
        # With an external draft model the cap based on native MTP heads
        # does not apply — the draft can predict an arbitrary number of
        # tokens, and the engine itself decides the upper bound.
        hw = _make_hardware(mtp_num_layers=1)
        exp = _make_experiment(
            speculative_algorithm="eagle",
            speculative_draft_model="some/draft-model",
            speculative_num_steps=8,
        )
        errors = validate_experiment(exp, hw)
        assert not any("native MTP head count" in e for e in errors)

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


# ─── Agentic-goal gates ───────────────────────────────────────────────────

from inference_agent.models import AgentConfig, OptimizationGoal


class TestAgenticGoalGates:
    """When goal == OptimizationGoal.AGENTIC, two extra gates fire:
      1. enable_prefix_caching MUST be true (shared-prefix workload).
      2. max_model_len ≤ 2× the agentic workload budget.
    Other goals are unaffected (regression check)."""

    def _agent_config(self, **bench_overrides) -> AgentConfig:
        ac = AgentConfig()
        ac.benchmark.enable_agentic_long_context = True
        ac.benchmark.agentic_shared_prefix_tokens = 16_000
        ac.benchmark.agentic_unique_prompt_tokens = 8_000
        ac.benchmark.agentic_max_output_tokens = 2_400
        ac.benchmark.agentic_tool_result_max_tokens = 1_536
        ac.benchmark.agentic_turns_per_session = 4
        for k, v in bench_overrides.items():
            setattr(ac.benchmark, k, v)
        return ac

    def test_prefix_caching_required_under_agentic(self):
        hw = _make_hardware(model_max_context=131_072)
        exp = _make_experiment(
            max_model_len=32_768,
            enable_prefix_caching=False,
        )
        errors = validate_experiment(
            exp, hw, self._agent_config(), OptimizationGoal.AGENTIC,
        )
        assert any("enable_prefix_caching" in e for e in errors)

    def test_prefix_caching_required_only_under_agentic(self):
        """The same config passes when the goal is throughput/latency."""
        hw = _make_hardware(model_max_context=131_072)
        exp = _make_experiment(
            max_model_len=32_768,
            enable_prefix_caching=False,
        )
        errors = validate_experiment(
            exp, hw, self._agent_config(), OptimizationGoal.LATENCY,
        )
        assert errors == []

    def test_oversize_max_model_len_rejected_under_agentic(self):
        """Budget = 16k + 8k + 4*(2.4k + 1.5k) ≈ 40k. 131_072 > 2×40k=80k → reject."""
        hw = _make_hardware(model_max_context=262_144)
        exp = _make_experiment(
            max_model_len=131_072,
            enable_prefix_caching=True,
        )
        errors = validate_experiment(
            exp, hw, self._agent_config(), OptimizationGoal.AGENTIC,
        )
        assert any(
            "max_model_len" in e and "agentic workload budget" in e
            for e in errors
        )

    def test_tight_max_model_len_accepted_under_agentic(self):
        """65_536 sits below the 2×budget threshold and works for the workload."""
        hw = _make_hardware(model_max_context=131_072)
        exp = _make_experiment(
            max_model_len=65_536,
            enable_prefix_caching=True,
        )
        errors = validate_experiment(
            exp, hw, self._agent_config(), OptimizationGoal.AGENTIC,
        )
        assert errors == []

    def test_oversize_max_model_len_passes_under_other_goals(self):
        hw = _make_hardware(model_max_context=262_144)
        exp = _make_experiment(
            max_model_len=131_072,
            enable_prefix_caching=True,
        )
        errors = validate_experiment(
            exp, hw, self._agent_config(), OptimizationGoal.THROUGHPUT,
        )
        assert errors == []

    def test_legacy_no_goal_keeps_old_behavior(self):
        """validate_experiment(exp, hw) without goal/agent_config still works
        (e.g. ad-hoc callers in tests / other modules)."""
        hw = _make_hardware(model_max_context=131_072)
        exp = _make_experiment(
            max_model_len=131_072,
            enable_prefix_caching=False,
        )
        errors = validate_experiment(exp, hw)
        # Prefix-cache gate doesn't fire without goal=AGENTIC, oversize gate
        # doesn't fire without agent_config — legacy ad-hoc validation stays
        # permissive about the agentic-specific checks.
        assert errors == []
