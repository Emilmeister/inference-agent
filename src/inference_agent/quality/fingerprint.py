"""Quality fingerprint — a stable hash over quality-relevant config dimensions.

Model intelligence (tool-calling correctness, structured-output validity,
agentic task success) depends only on a SUBSET of the launch config: the
weights and how tokens are produced/decoded — NOT on batching, KV-cache sizing,
prefill chunking or memory knobs, which change throughput/latency but not the
output token distribution.

So we fingerprint exactly the dimensions that can change quality:
  - model + engine + hardware key (numerics differ across GPUs/engines)
  - quantization, dtype, kv_cache_dtype
  - speculative decoding (different draft models CAN shift outputs)
  - max_model_len (a different effective context = a different deployment)
  - a curated set of quality-affecting `extra_engine_args` flags
    (tool-call parser, reasoning parser, guided-decoding backend, chat
    template, speculative-config, generation overrides). Batching/memory
    flags are deliberately excluded so they don't fragment the fingerprint.

Two finalists with the same fingerprint are quality-equivalent, so the
expensive terminal-bench run (~2h) executes once per fingerprint and the
result is attributed to every finalist that shares it.
"""

from __future__ import annotations

import hashlib
import json

from inference_agent.models_pkg.domain import ExperimentConfig, HardwareProfile

# `extra_engine_args` flag names that change the produced tokens (and thus
# quality). Everything not listed here (batching, memory, scheduling, prefix
# caching, …) is excluded so it never splits an otherwise-equivalent config.
QUALITY_RELEVANT_FLAGS: frozenset[str] = frozenset({
    "--tool-call-parser",
    "--enable-auto-tool-choice",      # boolean
    "--reasoning-parser",
    "--enable-reasoning",             # boolean
    "--guided-decoding-backend",
    "--chat-template",
    "--chat-template-content-format",
    "--speculative-config",
    "--override-generation-config",
    "--generation-config",
    "--hf-overrides",
    "--rope-scaling",
    "--rope-theta",
})


def _extract_quality_args(extra_engine_args: list[str]) -> dict[str, object]:
    """Project the quality-relevant subset of `extra_engine_args` into a dict.

    Handles `--flag value`, `--flag=value`, and boolean `--flag` forms. Flags
    not in `QUALITY_RELEVANT_FLAGS` are dropped so batching/memory tuning never
    perturbs the fingerprint.
    """
    args = extra_engine_args or []
    out: dict[str, object] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if isinstance(arg, str) and arg.startswith("--"):
            name, eq, inline = arg.partition("=")
            if name in QUALITY_RELEVANT_FLAGS:
                if eq:
                    out[name] = inline
                elif i + 1 < len(args) and not str(args[i + 1]).startswith("--"):
                    out[name] = args[i + 1]
                    i += 1
                else:
                    out[name] = True  # boolean flag (no value)
        i += 1
    return out


def quality_fingerprint(
    config: ExperimentConfig,
    hardware: HardwareProfile,
    model_name: str,
) -> str:
    """Return a stable 16-char hex fingerprint of the quality-relevant config.

    Deterministic: same quality dimensions → same fingerprint, regardless of
    batching/memory knobs or dict ordering.
    """
    primary = hardware.gpus[0] if hardware.gpus else None
    payload = {
        "model": model_name,
        "engine": config.engine.value,
        "quantization": config.quantization,
        "dtype": config.dtype,
        "kv_cache_dtype": config.kv_cache_dtype,
        "max_model_len": config.max_model_len,
        "speculative_algorithm": config.speculative_algorithm,
        "speculative_draft_model": config.speculative_draft_model,
        "speculative_num_steps": config.speculative_num_steps,
        "quality_args": _extract_quality_args(config.extra_engine_args),
        "hardware": {
            "gpu_name": primary.name if primary else "",
            "gpu_count": hardware.gpu_count,
            "gpu_vram_mb": primary.vram_total_mb if primary else 0,
            "nvlink": hardware.nvlink_available,
        },
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
