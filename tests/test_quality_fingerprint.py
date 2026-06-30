"""Tests for the quality fingerprint."""

from __future__ import annotations

from inference_agent.models import EngineType, ExperimentConfig, GPUInfo, HardwareProfile
from inference_agent.quality.fingerprint import _extract_quality_args, quality_fingerprint


def _hw(gpu: str = "H100", count: int = 8, vram: int = 81920, nvlink: bool = True) -> HardwareProfile:
    return HardwareProfile(
        gpus=[GPUInfo(index=i, name=gpu, vram_total_mb=vram, vram_free_mb=vram) for i in range(count)],
        gpu_count=count,
        nvlink_available=nvlink,
        model_name="m",
    )


def _cfg(**kw) -> ExperimentConfig:
    base = dict(engine=EngineType.VLLM)
    base.update(kw)
    return ExperimentConfig(**base)


def test_fingerprint_is_deterministic():
    fp1 = quality_fingerprint(_cfg(quantization="fp8"), _hw(), "m")
    fp2 = quality_fingerprint(_cfg(quantization="fp8"), _hw(), "m")
    assert fp1 == fp2
    assert len(fp1) == 16


def test_batching_knobs_do_not_change_fingerprint():
    a = _cfg(max_num_seqs=64, gpu_memory_utilization=0.90, enable_chunked_prefill=True)
    b = _cfg(max_num_seqs=256, gpu_memory_utilization=0.95, enable_chunked_prefill=False)
    assert quality_fingerprint(a, _hw(), "m") == quality_fingerprint(b, _hw(), "m")


def test_quality_dimensions_change_fingerprint():
    base = quality_fingerprint(_cfg(quantization=None), _hw(), "m")
    assert quality_fingerprint(_cfg(quantization="fp8"), _hw(), "m") != base
    assert quality_fingerprint(_cfg(kv_cache_dtype="fp8"), _hw(), "m") != base
    assert quality_fingerprint(_cfg(dtype="float16"), _hw(), "m") != base
    assert quality_fingerprint(_cfg(), _hw(gpu="A100"), "m") != base
    assert quality_fingerprint(_cfg(), _hw(), "other-model") != base


def test_tool_parser_flag_changes_fingerprint_but_batching_arg_does_not():
    base = quality_fingerprint(_cfg(), _hw(), "m")
    with_parser = quality_fingerprint(
        _cfg(extra_engine_args=["--tool-call-parser", "hermes"]), _hw(), "m"
    )
    with_batch_arg = quality_fingerprint(
        _cfg(extra_engine_args=["--max-num-seqs", "512"]), _hw(), "m"
    )
    assert with_parser != base
    assert with_batch_arg == base


def test_extract_quality_args_forms():
    args = [
        "--tool-call-parser", "hermes",
        "--enable-auto-tool-choice",
        "--guided-decoding-backend=xgrammar",
        "--max-num-seqs", "512",            # excluded (not quality-relevant)
        "--enforce-eager",                   # excluded
    ]
    extracted = _extract_quality_args(args)
    assert extracted == {
        "--tool-call-parser": "hermes",
        "--enable-auto-tool-choice": True,
        "--guided-decoding-backend": "xgrammar",
    }
