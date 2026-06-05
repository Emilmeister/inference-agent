"""nerdctl helper functions for container lifecycle management.

The agent shells out to `nerdctl` (containerd's docker-compatible CLI) for
all container ops. Flag surface used here is identical to docker's, so all
`nerdctl run` arguments built by the engines work as-is. Requires nerdctl,
containerd, CNI plugins and (for GPU passthrough) nvidia-container-toolkit
installed on the host VM.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess

import aiohttp

logger = logging.getLogger(__name__)


async def run_container(args: list[str], timeout: int = 60) -> str:
    """Run a nerdctl command and return the container ID.

    Assumes the image is already pulled — `nerdctl run -d` on a present image
    returns in seconds. If the image isn't local, nerdctl would do an implicit
    pull here and a multi-GB image would blow this timeout. Use `pull_image`
    before calling to keep responsibilities separate and timeouts honest.
    """
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    if proc.returncode != 0:
        error = stderr.decode().strip()
        raise RuntimeError(f"nerdctl run failed (rc={proc.returncode}): {error}")
    return stdout.decode().strip()


async def image_exists_locally(image: str) -> bool:
    """Return True if nerdctl has the exact image:tag present locally."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nerdctl", "image", "inspect", image,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        return await asyncio.wait_for(proc.wait(), timeout=10) == 0
    except (asyncio.TimeoutError, FileNotFoundError):
        return False


async def pull_image(image: str, timeout: int = 900) -> None:
    """Pull a container image with a long, explicit timeout.

    Raises RuntimeError on failure (including timeout) so callers can classify
    the experiment as `image_pull_failed` rather than a generic startup crash.
    """
    proc = await asyncio.create_subprocess_exec(
        "nerdctl", "pull", image,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass
        raise RuntimeError(f"nerdctl pull {image} timed out after {timeout}s")
    if proc.returncode != 0:
        err = stderr.decode().strip()
        raise RuntimeError(f"nerdctl pull {image} failed (rc={proc.returncode}): {err}")


async def stop_container(name: str, timeout: int = 30) -> None:
    """Stop and remove a container by name. Ignores errors if not running."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nerdctl", "stop", "-t", "10", name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (asyncio.TimeoutError, Exception) as e:
        logger.debug("Container stop failed (may not exist): %s", e)

    # Force remove if still exists
    try:
        proc = await asyncio.create_subprocess_exec(
            "nerdctl", "rm", "-f", name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=10)
    except Exception:
        pass


async def _is_container_running(name: str) -> bool:
    """Check if a container is still running."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nerdctl", "inspect", "-f", "{{.State.Running}}", name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        return stdout.decode().strip() == "true"
    except Exception:
        return False


# Patterns indicating the engine has hit a non-recoverable error before the
# health endpoint comes up. Detecting these in container logs lets us abort
# immediately instead of waiting for the full timeout, and lets the planner
# see a precise classification instead of generic "healthcheck_timeout".
_FATAL_LOG_PATTERNS: tuple[tuple[str, str], ...] = (
    ("invalid choice:", "argparse_error"),
    ("unrecognized arguments", "argparse_error"),
    ("error: argument", "argparse_error"),
    ("the following arguments are required", "argparse_error"),
    ("torch.cuda.outofmemoryerror", "oom"),
    ("cuda out of memory", "oom"),
    ("out of memory", "oom"),
    ("cuda error", "cuda_error"),
    ("nccl error", "nccl_error"),
    ("repositorynotfounderror", "model_not_found"),
    ("gatedrepoerror", "model_gated"),
    ("no space left on device", "disk_full"),
    ("os error 28", "disk_full"),
    # HF cache layout / weight resolution failures. Both engines emit one
    # of these when host_cache_dir is misconfigured (the hub/ subdir trap)
    # or when offline mode can't resolve a repo id to a snapshot.
    ("cannot find any model weights", "weights_not_found"),
    ("does not appear to have a file named", "weights_not_found"),
    # Tokenizer load failures. Old transformers fall through to slow
    # conversion (the `tekken.json` AttributeError); new transformers raise
    # the "Couldn't instantiate the backend tokenizer" ValueError. Both
    # mean the same thing from the planner's POV: a setup bug, not a
    # hyperparameter to tune.
    ("couldn't instantiate the backend tokenizer", "tokenizer_init_failed"),
    ("vocab_file.endswith", "tokenizer_init_failed"),
    ("can't load image processor", "tokenizer_init_failed"),
    # vLLM v1 multiproc worker bootstrap — the wrapper traceback at the
    # tail says "Engine core initialization failed" but the real cause is
    # usually one of the patterns above, surfaced from a worker stderr.
    # Catching this explicitly stops it from masquerading as "startup_crash"
    # when no upper-stack marker matched.
    ("workerproc initialization failed", "worker_init_failed"),
    ("ranks did not initialize properly", "distributed_init_failed"),
    # Engine-side feature rejections we've hit in the wild.
    ("speculative decoding for", "speculative_incompatible"),
    ("reasoning parser could not locate", "reasoning_parser_incompatible"),
)

# Markers indicating active loading progress. Any new marker resets the
# idle-timeout clock so legitimately-slow loads (big models, slow disk) get
# the time they need without needing one giant nominal timeout.
_PROGRESS_MARKERS: tuple[str, ...] = (
    "loading safetensors",
    "loading weights",
    "loading model weights",
    "loading checkpoint shards",
    "downloading",
    "fetching",
    "capturing cuda graph",
    "capturing the model",
    "init torch distributed",
    "initializing model",
    "init engine",
    "starting api server",
    "loading multimodal weights",
    "model loaded",
    "warmup complete",
)


def scan_engine_logs(logs: str) -> dict:
    """Classify the state of a starting engine from its container logs.

    Returns one of:
      {"state": "fatal", "classification": "<kind>", "marker": "..."}
      {"state": "loading", "markers": [...]}
      {"state": "unknown"}
    """
    text = logs.lower()
    for needle, kind in _FATAL_LOG_PATTERNS:
        if needle in text:
            return {"state": "fatal", "classification": kind, "marker": needle}

    seen = [m for m in _PROGRESS_MARKERS if m in text]
    if seen:
        return {"state": "loading", "markers": seen}

    return {"state": "unknown"}


async def wait_for_healthy(
    url: str,
    timeout_sec: int = 1800,
    poll_interval: float = 5.0,
    container_name: str | None = None,
    idle_timeout_sec: int = 300,
    log_scan_interval_sec: float = 10.0,
) -> dict:
    """Poll a health endpoint until it returns 200 or we hit a stop condition.

    Returns a dict with keys:
      - healthy: bool — endpoint returned 200
      - reason: str — one of "ok", "hard_timeout", "idle_timeout",
        "fatal_in_logs", "container_dead"
      - classification: str | None — failure classification (e.g.
        "argparse_error", "oom") if available from log scan
      - marker: str | None — the log substring that triggered classification
      - elapsed_sec: float

    Two clocks bound the wait:
      hard deadline = start + timeout_sec  (absolute cap)
      idle deadline = last_progress + idle_timeout_sec  (resets on every
        new progress marker spotted in container logs)

    Big-model loads (60GB+) routinely take 10+ min just to load weights from
    a fast NVMe. The idle clock lets these complete as long as we keep seeing
    progress, while still failing fast if the engine actually stalls.
    """
    loop = asyncio.get_event_loop()
    start = loop.time()
    hard_deadline = start + timeout_sec
    last_progress_time = start
    seen_markers: set[str] = set()
    last_log_scan = 0.0
    attempts = 0

    async with aiohttp.ClientSession() as session:
        while True:
            now = loop.time()
            elapsed = now - start

            if now >= hard_deadline:
                logger.error(
                    "Health check hard timeout after %ds (%d attempts): %s",
                    timeout_sec, attempts, url,
                )
                return {
                    "healthy": False,
                    "reason": "hard_timeout",
                    "classification": "healthcheck_timeout",
                    "marker": None,
                    "elapsed_sec": elapsed,
                }

            if now - last_progress_time >= idle_timeout_sec:
                logger.error(
                    "Health check idle timeout: no progress for %ds (elapsed %ds): %s",
                    idle_timeout_sec, int(elapsed), url,
                )
                return {
                    "healthy": False,
                    "reason": "idle_timeout",
                    "classification": "healthcheck_idle",
                    "marker": None,
                    "elapsed_sec": elapsed,
                }

            attempts += 1

            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        logger.info(
                            "Health check passed after %ds (%d attempts): %s",
                            int(elapsed), attempts, url,
                        )
                        return {
                            "healthy": True,
                            "reason": "ok",
                            "classification": None,
                            "marker": None,
                            "elapsed_sec": elapsed,
                        }
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass

            # Periodic log scan + container alive check
            if container_name and (now - last_log_scan) >= log_scan_interval_sec:
                last_log_scan = now

                alive = await _is_container_running(container_name)
                if not alive:
                    final_logs = await get_container_logs(container_name)
                    scan = scan_engine_logs(final_logs)
                    classification = scan.get("classification") or "startup_crash"
                    logger.error(
                        "Container '%s' exited before becoming healthy "
                        "(elapsed %ds, classification=%s).",
                        container_name, int(elapsed), classification,
                    )
                    return {
                        "healthy": False,
                        "reason": "container_dead",
                        "classification": classification,
                        "marker": scan.get("marker"),
                        "elapsed_sec": elapsed,
                    }

                logs = await get_container_logs(container_name, tail=200)
                scan = scan_engine_logs(logs)

                if scan["state"] == "fatal":
                    logger.error(
                        "Detected fatal pattern in logs (elapsed %ds): %s — aborting wait.",
                        int(elapsed), scan["marker"],
                    )
                    return {
                        "healthy": False,
                        "reason": "fatal_in_logs",
                        "classification": scan["classification"],
                        "marker": scan["marker"],
                        "elapsed_sec": elapsed,
                    }

                if scan["state"] == "loading":
                    new_markers = set(scan["markers"]) - seen_markers
                    if new_markers:
                        for m in sorted(new_markers):
                            logger.info(
                                "  Engine progressing (elapsed %ds): %s",
                                int(elapsed), m,
                            )
                        seen_markers.update(new_markers)
                        last_progress_time = now

            if attempts % 12 == 0:
                logger.info(
                    "  Still waiting for health check... (%ds elapsed, hard cap %ds)",
                    int(elapsed), timeout_sec,
                )

            await asyncio.sleep(poll_interval)


async def get_container_logs(name: str, tail: int = 100) -> str:
    """Get container logs. tail=0 returns the full log."""
    cmd = ["nerdctl", "logs"]
    if tail > 0:
        cmd.extend(["--tail", str(tail)])
    cmd.append(name)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # Full-log fetch on a long-running engine can be tens of MB; bump the
        # timeout proportionally rather than truncating, so we don't lose the
        # actual root-cause traceback to a 10s I/O cap.
        timeout = 30 if tail == 0 else 10
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return stdout.decode(errors="replace") + stderr.decode(errors="replace")
    except Exception as e:
        return f"Failed to get logs: {e}"


# Substrings that mark the start of an interesting stack on engine crashes.
# Used by extract_error_excerpt to find the most diagnostic window of the
# log when the full output is too large to attach to an ExperimentError.
_ERROR_MARKERS: tuple[str, ...] = (
    "Traceback (most recent call last)",
    "ERROR",
    "RuntimeError:",
    "AssertionError:",
    "ValueError:",
    "TypeError:",
    "AttributeError:",
    "OSError:",
    "Exception:",
    "FATAL",
    "raise ",
    "assert ",
    "torch.cuda.OutOfMemoryError",
    "nccl",
)


def extract_error_excerpt(
    full_log: str,
    *,
    max_lines: int = 200,
    context_before: int = 20,
) -> str:
    """Return the most diagnostic slice of a container log.

    On engine crashes we want the first stack trace (the multiproc worker
    failure is usually the root cause; later tracebacks just unwind the
    wrapper), with a bit of context before it. If we can't find any marker
    we fall back to the last `max_lines` lines.

    The full log (potentially tens of MB) is always persisted separately;
    this excerpt is what we put into ExperimentError.details so it stays
    LLM-prompt-sized.
    """
    if not full_log:
        return ""
    lines = full_log.splitlines()
    first_match = -1
    for i, line in enumerate(lines):
        if any(m in line for m in _ERROR_MARKERS):
            first_match = i
            break
    if first_match < 0:
        return "\n".join(lines[-max_lines:])
    start = max(0, first_match - context_before)
    end = min(len(lines), start + max_lines)
    excerpt_lines = lines[start:end]
    prefix = f"[excerpt: lines {start + 1}-{end} of {len(lines)}]\n"
    return prefix + "\n".join(excerpt_lines)


async def get_image_digest(image: str) -> str:
    """Get the digest of a container image. Returns empty string on failure."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nerdctl", "inspect", "--format", "{{index .RepoDigests 0}}", image,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        digest = stdout.decode().strip()
        if digest and "@sha256:" in digest:
            return digest
        # Fallback to image ID
        proc2 = await asyncio.create_subprocess_exec(
            "nerdctl", "inspect", "--format", "{{.Id}}", image,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout2, _ = await asyncio.wait_for(proc2.communicate(), timeout=10)
        return stdout2.decode().strip()
    except Exception as e:
        logger.debug("Failed to get image digest for %s: %s", image, e)
        return ""


async def get_container_exit_code(name: str) -> int | None:
    """Get the exit code of a stopped container. Returns None if unknown."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nerdctl", "inspect", "-f", "{{.State.ExitCode}}", name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        code_str = stdout.decode().strip()
        if code_str.isdigit():
            return int(code_str)
        return None
    except Exception:
        return None


async def get_engine_version(api_base_url: str) -> str:
    """Get engine version from the /version endpoint. Returns empty string on failure."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{api_base_url}/version",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("version", str(data))
    except Exception:
        pass
    # Fallback: try /v1/models for basic info
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{api_base_url}/models",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    return "available"
    except Exception:
        pass
    return ""


def stop_all_bench_containers() -> None:
    """Stop all containers with bench- prefix. For cleanup."""
    try:
        result = subprocess.run(
            ["nerdctl", "ps", "-q", "--filter", "name=bench-"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        container_ids = result.stdout.strip().split("\n")
        for cid in container_ids:
            if cid.strip():
                subprocess.run(
                    ["nerdctl", "rm", "-f", cid.strip()],
                    capture_output=True,
                    timeout=10,
                )
    except Exception as e:
        logger.warning("Cleanup failed: %s", e)
