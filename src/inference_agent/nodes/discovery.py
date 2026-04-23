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

    # If no images found locally, assume both are pullable
    if not engines:
        logger.info("No engine images found locally, assuming both are available")
        engines = [EngineType.VLLM, EngineType.SGLANG]

    return engines


async def discovery_node(state: AgentState) -> dict:
    """Detect hardware, model info, and available engines."""
    config = state["config"]
    logger.info("Starting discovery for model: %s", config.model_name)

    # Run GPU detection and model config read concurrently
    loop = asyncio.get_event_loop()
    gpus, nvlink, model_config, engines = await asyncio.gather(
        loop.run_in_executor(None, _detect_gpus),
        loop.run_in_executor(None, _detect_nvlink),
        loop.run_in_executor(
            None, _read_model_config, config.model_name, config.model_revision
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

    # Estimate model size from hidden_size and num_layers
    hidden_size = text_config.get("hidden_size", model_config.get("hidden_size", 0))
    num_layers = text_config.get("num_hidden_layers", model_config.get("num_hidden_layers", 0))
    vocab_size = text_config.get("vocab_size", model_config.get("vocab_size", 0))
    model_size_params = None
    if hidden_size and num_layers:
        model_size_params = 12 * hidden_size**2 * num_layers + vocab_size * hidden_size

    # Determine max context — check both top-level and text_config
    max_context = max(
        model_config.get("max_position_embeddings", 0),
        text_config.get("max_position_embeddings", 0),
    )
    if max_context == 0:
        max_context = 4096  # absolute fallback

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
