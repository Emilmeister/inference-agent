"""Discovery node — detects hardware, model info, and available engines."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time

from huggingface_hub import hf_hub_download, snapshot_download

from inference_agent.models import (
    EngineType,
    GPUInfo,
    HardwareProfile,
)
from inference_agent.quality.preflight import preflight_quality
from inference_agent.state import AgentState
from inference_agent.utils.container import pull_image

logger = logging.getLogger(__name__)


def _detect_gpus() -> list[GPUInfo]:
    """Query nvidia-smi for GPU information."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            gpus.append(GPUInfo(
                index=int(parts[0]),
                name=parts[1],
                vram_total_mb=int(float(parts[2])),
                vram_free_mb=int(float(parts[3])),
            ))
        return gpus
    except (subprocess.TimeoutExpired, FileNotFoundError, IndexError) as e:
        logger.error("Failed to detect GPUs: %s", e)
        return []


def _detect_nvlink(attempts: int = 3, timeout_sec: float = 20.0) -> bool:
    """Check if NVLink is available between GPUs via `nvidia-smi topo -m`.

    For a fixed stand NVLink presence is a hardware CONSTANT, but the topo probe
    can be slow/contended while GPUs are busy (e.g. a large model is loading in a
    just-launched container). A transient failure here is dangerous, not
    cosmetic: `nvlink_available` is part of the exact HardwareProfile match key
    that history/baseline lookups use, so one flaky probe silently flipping it to
    False makes the agent think it's on different hardware — it then ignores all
    prior experiments and re-runs the baseline, splitting the DB into two
    incompatible nvlink=true/false profiles.

    So we (a) retry with a generous timeout, (b) only trust a probe that actually
    succeeded (rc==0, non-empty output) — a clean run with no "NV" links is a
    legitimate False, but a timeout / non-zero exit is a FAILED probe, not
    evidence of "no NVLink", and (c) log loudly on total failure instead of
    quietly returning False.
    """
    last_err: str | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            last_err = f"timeout after {timeout_sec}s"
        except FileNotFoundError:
            # nvidia-smi missing is not transient — retrying won't help.
            logger.warning("nvidia-smi not found — assuming nvlink_available=False")
            return False
        else:
            if result.returncode == 0 and result.stdout.strip():
                # Probe succeeded: trust its verdict (legitimate True or False).
                return "NV" in result.stdout
            last_err = (
                f"exit={result.returncode}, empty_stdout={not result.stdout.strip()}"
            )
        if attempt < attempts:
            logger.debug(
                "nvlink probe attempt %d/%d failed (%s) — retrying",
                attempt, attempts, last_err,
            )
            time.sleep(1.0)
    logger.warning(
        "NVLink detection failed after %d attempts (%s). Falling back to "
        "nvlink_available=False — if this stand actually has NVLink, the run "
        "will MISS prior experiments/baseline (history matches on nvlink). "
        "Re-run when the GPUs are less busy.",
        attempts, last_err,
    )
    return False


def _read_model_config(model_name: str, revision: str | None = None) -> dict:
    """Download and read the model's config.json from HuggingFace."""
    try:
        config_path = hf_hub_download(
            repo_id=model_name,
            filename="config.json",
            revision=revision,
        )
        with open(config_path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to read model config for %s: %s", model_name, e)
        return {}


_DTYPE_BYTES = {
    "bfloat16": 2,
    "bf16": 2,
    "float16": 2,
    "fp16": 2,
    "half": 2,
    "float32": 4,
    "fp32": 4,
    "float": 4,
    "float8": 1,
    "fp8": 1,
    "int8": 1,
    "uint8": 1,
    "int4": 0.5,
}


def _read_model_size_bytes(
    model_name: str, revision: str | None = None
) -> int | None:
    """Read total model weight bytes from model.safetensors.index.json.

    Sharded HF safetensors checkpoints publish a small index file with
    metadata.total_size summed across all shards — this gives an exact byte
    count without downloading any weight files. Returns None if the model is
    not sharded or the index is unavailable; caller should fall back to a
    formula-based estimate.
    """
    try:
        idx_path = hf_hub_download(
            repo_id=model_name,
            filename="model.safetensors.index.json",
            revision=revision,
        )
    except Exception as e:
        logger.debug(
            "No safetensors index for %s (likely single-file or non-safetensors): %s",
            model_name, e,
        )
        return None

    try:
        with open(idx_path) as f:
            idx = json.load(f)
        total_size = idx.get("metadata", {}).get("total_size")
        if isinstance(total_size, (int, float)) and total_size > 0:
            return int(total_size)
    except Exception as e:
        logger.warning("Failed to parse safetensors index for %s: %s", model_name, e)

    return None


def _bytes_to_params(total_bytes: int, dtype_str: str | None) -> int:
    """Convert weight-bytes to parameter count using the model's dtype."""
    key = (dtype_str or "bfloat16").lower().strip()
    bytes_per_param = _DTYPE_BYTES.get(key, 2)
    return int(total_bytes / bytes_per_param)


def prefetch_and_normalize_model(
    model_name: str,
    cache_dir: str,
    revision: str | None,
    token: str | None,
    allow_patterns: list[str],
    raise_on_failure: bool = False,
) -> str | None:
    """Download model weights into the host HF cache and normalize the config.

    snapshot_download is idempotent — if the model is already present at the
    target revision, this returns quickly. The discovery node calls this
    best-effort (`raise_on_failure=False`) so a transient HF outage does not
    abort startup; the executor calls it for speculative draft models with
    `raise_on_failure=True` so a bad draft model ID surfaces immediately as
    a structured prefetch error instead of a 20-minute container hang.

    Returns the snapshot path on success, None on best-effort failure.
    """
    import os

    # Make sure the target dir exists and is writable by us before invoking
    # snapshot_download. snapshot_download writes refs/ files unconditionally
    # and on a read-only path it surfaces a permission error mid-fetch instead
    # of failing fast.
    try:
        os.makedirs(cache_dir, exist_ok=True)
        if not os.access(cache_dir, os.W_OK):
            raise PermissionError(
                f"cache_dir={cache_dir} is not writable by current user. "
                f"Set container.host_cache_dir to a path you own (e.g. ~/.cache/huggingface)."
            )
    except Exception as e:
        if raise_on_failure:
            raise
        logger.warning(
            "Prefetch skipped for %s — host cache not usable: %s. "
            "Engine will download inside the container instead.",
            model_name, e,
        )
        return None

    logger.info(
        "Prefetching %s into cache_dir=%s (this may take a while on first run)...",
        model_name, cache_dir,
    )
    try:
        path = snapshot_download(
            repo_id=model_name,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            allow_patterns=allow_patterns or None,
        )
        logger.info("Prefetch complete: %s", path)
    except Exception as e:
        if raise_on_failure:
            raise
        logger.warning(
            "Prefetch failed for %s (engine will download inside container): %s",
            model_name, e,
        )
        return None

    # Sanity check: confirm the snapshot is actually navigable from cache_dir
    # and has a config.json at the snapshot root. We deliberately use a plain
    # startswith on the LITERAL (unresolved) path here, not realpath — legit
    # cache layouts may have symlinks (e.g. an operator pre-populated
    # `<cache_dir>/models--foo` as a symlink into a larger volume), and we
    # don't want to false-positive on those. snapshot_download returns the
    # path under cache_dir even when the underlying storage is symlinked,
    # so the unresolved comparison is what tells us "did it write where it
    # said it did".
    if not path.startswith(cache_dir.rstrip(os.sep) + os.sep):
        msg = (
            f"HF cache layout mismatch: snapshot_download claimed to write to "
            f"{path} but cache_dir is {cache_dir}. Engines will not find the "
            f"model. Check container.host_cache_dir."
        )
        if raise_on_failure:
            raise RuntimeError(msg)
        logger.error(msg)
        return None
    if not os.path.exists(os.path.join(path, "config.json")):
        msg = (
            f"Prefetched snapshot at {path} is missing config.json. "
            f"This usually means snapshot_download wrote into the wrong place "
            f"because container.host_cache_dir points at an HF cache root "
            f"instead of its hub/ subdir."
        )
        if raise_on_failure:
            raise RuntimeError(msg)
        logger.error(msg)
        return None

    _normalize_cached_config(path)
    return path


# Fields that some model configs publish as int (e.g. `1`) but downstream
# strict dataclass validators in newer huggingface_hub builds insist on float.
# DeepSeekV3-derived MoE configs (GigaChat3, etc.) regularly hit this on
# `routed_scaling_factor`. Coercing in-place in the local HF cache snapshot
# keeps the upstream model untouched and unblocks every engine that loads
# from this cache.
_FLOAT_COERCE_FIELDS = (
    "routed_scaling_factor",
    "partial_rotary_factor",
    "rope_theta",
    "router_aux_loss_coef",
    "rms_norm_eps",
    "initializer_range",
    "attention_dropout",
)


def _coerce_floats_in_place(node: object) -> int:
    """Walk a parsed JSON tree and coerce known float-typed fields from int.

    Returns the number of values rewritten. Recurses into dicts and lists so
    nested configs (text_config, vision_config, rope_scaling, …) are covered.
    """
    fixed = 0
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if (
                key in _FLOAT_COERCE_FIELDS
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                node[key] = float(value)
                fixed += 1
            else:
                fixed += _coerce_floats_in_place(value)
        # rope_scaling.factor is the only nested field where the *key* alone
        # ('factor') is too generic to coerce blindly across the whole tree —
        # handle it explicitly when we see a rope_scaling/rope_parameters dict.
        for parent_key in ("rope_scaling", "rope_parameters"):
            sub = node.get(parent_key)
            if isinstance(sub, dict):
                f = sub.get("factor")
                if isinstance(f, int) and not isinstance(f, bool):
                    sub["factor"] = float(f)
                    fixed += 1
    elif isinstance(node, list):
        for item in node:
            fixed += _coerce_floats_in_place(item)
    return fixed


def _normalize_cached_config(snapshot_path: str) -> None:
    """Patch the cached `config.json` to coerce ints into floats where engines expect float.

    huggingface_hub StrictDataclass validation in fresh vLLM/SGLang images
    rejects `routed_scaling_factor: 1` (int) and similar literals that some
    model authors publish as integers. The fix is purely local — we rewrite
    the file in this user's HF cache snapshot, the upstream HF model is
    untouched.
    """
    import os

    config_path = os.path.join(snapshot_path, "config.json")
    if not os.path.exists(config_path):
        return

    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except Exception as e:
        logger.warning("Could not read cached config.json at %s: %s", config_path, e)
        return

    fixed = _coerce_floats_in_place(cfg)
    if fixed == 0:
        return

    # `config.json` in HF cache is a symlink into blobs/<sha>; rewriting via
    # the symlink updates the blob and is exactly what we want (every snapshot
    # pointing at this blob picks up the fix). os.path.realpath resolves the
    # link so json.dump writes to the actual blob, not creating a new file.
    real_path = os.path.realpath(config_path)
    try:
        with open(real_path, "w") as f:
            json.dump(cfg, f, indent=2)
        logger.info(
            "Normalized cached config.json (%d int→float coercions) at %s",
            fixed, real_path,
        )
    except Exception as e:
        logger.warning(
            "Could not write normalized config.json to %s: %s. "
            "Engine may crash with strict-dataclass validation errors.",
            real_path, e,
        )


def _detect_available_engines() -> list[EngineType]:
    """Check which engine images are available locally via nerdctl."""
    engines = []
    for engine, image_check in [
        (EngineType.VLLM, "vllm/vllm-openai"),
        (EngineType.SGLANG, "lmsysorg/sglang"),
    ]:
        try:
            result = subprocess.run(
                ["nerdctl", "images", "-q", image_check],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.stdout.strip():
                engines.append(engine)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    if not engines:
        logger.warning(
            "No engine images found locally. "
            "Pull at least one image before running: "
            "'nerdctl pull vllm/vllm-openai:latest' or "
            "'nerdctl pull lmsysorg/sglang:latest'"
        )

    return engines


async def discovery_node(state: AgentState) -> dict:
    """Detect hardware, model info, and available engines."""
    config = state["config"]
    logger.info("Starting discovery for model: %s", config.model_name)

    # Fail fast on a misconfigured quality phase BEFORE the (hours-long)
    # optimization loop — don't discover that harbor/so-testing isn't runnable
    # only after the search converges.
    await preflight_quality(config.quality)

    # Run GPU detection, model config read, and weight-size lookup concurrently
    loop = asyncio.get_event_loop()
    gpus, nvlink, model_config, model_size_bytes, engines = await asyncio.gather(
        loop.run_in_executor(None, _detect_gpus),
        loop.run_in_executor(None, _detect_nvlink),
        loop.run_in_executor(
            None, _read_model_config, config.model_name, config.model_revision
        ),
        loop.run_in_executor(
            None, _read_model_size_bytes, config.model_name, config.model_revision
        ),
        loop.run_in_executor(None, _detect_available_engines),
    )

    # Restrict to the configured GPU subset (first N by index) before anything
    # downstream reads gpu_count. This keeps the planner's tensor_parallel_size
    # constraint, the validator's divisibility checks, and the analyzer all
    # reasoning about the GPUs we actually expose to the container (the engine
    # gets CUDA_VISIBLE_DEVICES=0..N-1 in engines/base.py).
    requested_gpu_count = config.container.gpu_count
    if requested_gpu_count is not None and gpus:
        if requested_gpu_count > len(gpus):
            raise RuntimeError(
                f"container.gpu_count={requested_gpu_count} exceeds detected "
                f"GPUs ({len(gpus)}). Lower gpu_count or check nvidia-smi."
            )
        if requested_gpu_count < len(gpus):
            logger.info(
                "Restricting to first %d of %d detected GPUs (container.gpu_count)",
                requested_gpu_count, len(gpus),
            )
            gpus = gpus[:requested_gpu_count]

    # Prefetch weights into the host cache (mounted into containers) so
    # subsequent container runs don't each re-download the model.
    #
    # Fail loud here when prefetch_model=True: engines/base.py forces
    # HF_HUB_OFFLINE=1 in that mode, so a silent prefetch failure becomes
    # a cryptic container crash later ("Can't load image processor",
    # "config.json not found"). Surfacing the real cause — wrong repo id,
    # gated model without token, registry down — at discovery time saves
    # hours of confused planning.
    if config.startup.prefetch_model:
        try:
            await loop.run_in_executor(
                None,
                prefetch_and_normalize_model,
                config.model_name,
                config.container.host_cache_dir,
                config.model_revision,
                config.hf_token,
                config.startup.prefetch_allow_patterns,
                True,  # raise_on_failure
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to prefetch model '{config.model_name}'"
                + (f" @ revision {config.model_revision}" if config.model_revision else "")
                + f": {e}. "
                f"With startup.prefetch_model=true the container runs in HF offline "
                f"mode and needs weights present in container.host_cache_dir. "
                f"Fix the model name / hf_token, or set startup.prefetch_model=false "
                f"to let the engine download inside the container at run time."
            ) from e

    # Extract model info from config.json
    model_architecture = None
    architectures = model_config.get("architectures", [])
    if architectures:
        model_architecture = architectures[0].lower().replace("forconditionalgeneration", "").replace("forcausallm", "")

    # For VLM/multimodal models, key params are often nested in text_config
    text_config = model_config.get("text_config", {})

    # Determine model size in parameters.
    # Preferred: sum tensor sizes from model.safetensors.index.json (exact).
    # Fallback: SwiGLU-aware formula from config.json (approximate, ±15%).
    hidden_size = text_config.get("hidden_size", model_config.get("hidden_size", 0))
    num_layers = text_config.get("num_hidden_layers", model_config.get("num_hidden_layers", 0))
    vocab_size = text_config.get("vocab_size", model_config.get("vocab_size", 0))
    intermediate_size = text_config.get(
        "intermediate_size", model_config.get("intermediate_size", 0)
    )
    dtype_str = text_config.get("torch_dtype") or model_config.get("torch_dtype")

    model_size_params: int | None = None
    if model_size_bytes:
        model_size_params = _bytes_to_params(model_size_bytes, dtype_str)
        logger.info(
            "Model size from safetensors index: %.1f GB (%s, %d params)",
            model_size_bytes / (1024**3),
            dtype_str or "bfloat16-assumed",
            model_size_params,
        )
    elif hidden_size and num_layers:
        # SwiGLU FFN: 3 matrices of (hidden, intermediate). If intermediate
        # is unknown, assume 4x hidden (matches Llama family). Attention is
        # ~4*h^2 per layer (Q/K/V/O, ignoring GQA reduction for simplicity).
        ffn = intermediate_size or (4 * hidden_size)
        per_layer = 4 * hidden_size * hidden_size + 3 * hidden_size * ffn
        model_size_params = per_layer * num_layers + vocab_size * hidden_size
        logger.info(
            "Model size estimated from config (no safetensors index): ~%d params",
            model_size_params,
        )

    # Determine max context — check both top-level and text_config
    max_context = max(
        model_config.get("max_position_embeddings", 0),
        text_config.get("max_position_embeddings", 0),
    )
    if max_context == 0:
        max_context = 4096
        logger.warning(
            "Could not determine max context length from model config for '%s'. "
            "Falling back to %d — this may be incorrect. "
            "Check model config.json on HuggingFace.",
            config.model_name,
            max_context,
        )

    # Check rope_scaling in both places
    rope_scaling = model_config.get("rope_scaling") or text_config.get("rope_scaling")
    if rope_scaling and isinstance(rope_scaling, dict):
        factor = rope_scaling.get("factor", 1.0)
        original_max = rope_scaling.get("original_max_position_embeddings", max_context)
        max_context = max(max_context, int(original_max * factor))

    # Check alternative fields
    max_context = max(
        max_context,
        model_config.get("max_sequence_length", 0),
        text_config.get("max_sequence_length", 0),
    )

    # Detect if model is multimodal (VLM)
    is_vlm = "vision_config" in model_config or "ForConditionalGeneration" in str(architectures)
    # Detect native Multi-Token Prediction heads.
    # Qwen3.5-MoE et al. use `mtp_num_hidden_layers`; DeepSeek-V3 family uses
    # `num_nextn_predict_layers`. Either spelling appears in `text_config` for
    # multimodal wrappers or at the top level for text-only models. A non-zero
    # count means the model can self-speculate (NEXTN/EAGLE3 without an external
    # draft model) and gives us an upper bound on `speculative_num_steps`.
    mtp_num_layers = max(
        int(text_config.get("mtp_num_hidden_layers", 0) or 0),
        int(text_config.get("num_nextn_predict_layers", 0) or 0),
        int(model_config.get("mtp_num_hidden_layers", 0) or 0),
        int(model_config.get("num_nextn_predict_layers", 0) or 0),
    )

    logger.info(
        "Model info: max_context=%d, is_vlm=%s, mtp_num_layers=%d, hidden=%d, layers=%d",
        max_context, is_vlm, mtp_num_layers, hidden_size, num_layers,
    )

    # Auto-pull images for requested engines that aren't present locally.
    # `_detect_available_engines` only checks for ANY tag of the engine's
    # image prefix; the exact tag in `config.container.*_image` may still
    # be missing and trigger a slow implicit pull at experiment-start time.
    # Doing the pull here keeps timeouts explicit and surfaces network /
    # registry problems before we waste discovery + planning work.
    image_for_engine = {
        EngineType.VLLM: config.container.vllm_image,
        EngineType.SGLANG: config.container.sglang_image,
    }
    missing = [e for e in config.experiments.engines if e not in engines]
    for engine in missing:
        image = image_for_engine.get(engine)
        if not image:
            continue
        pull_timeout = config.startup.image_pull_timeout_sec
        logger.info(
            "Engine %s image not present locally — pulling %s (timeout %ds)...",
            engine.value, image, pull_timeout,
        )
        try:
            await pull_image(image, timeout=pull_timeout)
            engines.append(engine)
            logger.info("Pulled %s for engine %s", image, engine.value)
        except RuntimeError as e:
            logger.warning(
                "Failed to pull %s for engine %s: %s",
                image, engine.value, e,
            )

    # Filter engines to only those requested in config
    available = [e for e in engines if e in config.experiments.engines]

    if not available:
        requested = [e.value for e in config.experiments.engines]
        found = [e.value for e in engines]
        raise RuntimeError(
            f"No usable engine images available even after auto-pull. "
            f"Requested engines: {requested}. "
            f"Images found locally: {found or 'none'}. "
            f"Check container.{{engine}}_image in config and registry connectivity."
        )

    hardware = HardwareProfile(
        gpus=gpus,
        gpu_count=len(gpus),
        nvlink_available=nvlink,
        model_name=config.model_name,
        model_size_params=model_size_params,
        model_architecture=model_architecture,
        model_max_context=max_context,
        is_vlm=is_vlm,
        mtp_num_layers=mtp_num_layers,
        available_engines=available,
    )

    logger.info(
        "Discovery complete: %d GPUs (%s), max_context=%d, engines=%s",
        hardware.gpu_count,
        gpus[0].name if gpus else "none",
        max_context,
        [e.value for e in available],
    )

    return {
        "hardware": hardware,
        "experiments_count": 0,
        "best_throughput": 0.0,
        "best_throughput_config_id": "",
        "best_latency_ttft_p95": float("inf"),
        "best_latency_config_id": "",
        "best_balanced_config_id": "",
        "best_balanced_throughput": 0.0,
        "best_balanced_latency": float("inf"),
        "pareto_front": [],
        "status": "running",
        "stop_reason": None,
    }
