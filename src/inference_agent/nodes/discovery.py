"""Discovery node — detects hardware, model info, and available engines."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess

from huggingface_hub import hf_hub_download

from inference_agent.models import (
    EngineType,
    GPUInfo,
    HardwareProfile,
)
from inference_agent.state import AgentState

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


def _detect_nvlink() -> bool:
    """Check if NVLink is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return "NV" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
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


def _detect_available_engines() -> list[EngineType]:
    """Check which Docker images are available locally."""
    engines = []
    for engine, image_check in [
        (EngineType.VLLM, "vllm/vllm-openai"),
        (EngineType.SGLANG, "lmsysorg/sglang"),
    ]:
        try:
            result = subprocess.run(
                ["docker", "images", "-q", image_check],
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
            "No engine Docker images found locally. "
            "Pull at least one image before running: "
            "'docker pull vllm/vllm-openai:latest' or "
            "'docker pull lmsysorg/sglang:latest'"
        )

    return engines


async def discovery_node(state: AgentState) -> dict:
    """Detect hardware, model info, and available engines."""
    config = state["config"]
    logger.info("Starting discovery for model: %s", config.model_name)

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
    # Detect MTP support
    has_mtp = text_config.get("mtp_num_hidden_layers", 0) > 0

    logger.info(
        "Model info: max_context=%d, is_vlm=%s, has_mtp=%s, hidden=%d, layers=%d",
        max_context, is_vlm, has_mtp, hidden_size, num_layers,
    )

    # Filter engines to only those requested in config
    available = [e for e in engines if e in config.experiments.engines]

    if not available:
        requested = [e.value for e in config.experiments.engines]
        found = [e.value for e in engines]
        raise RuntimeError(
            f"No usable engine images found. "
            f"Requested engines: {requested}. "
            f"Images found locally: {found or 'none'}. "
            f"Pull the required Docker images first."
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
        has_mtp=has_mtp,
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
