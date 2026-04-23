"""Docker helper functions for container lifecycle management."""

from __future__ import annotations

import asyncio
import logging
import subprocess

import aiohttp

logger = logging.getLogger(__name__)


async def run_container(args: list[str], timeout: int = 30) -> str:
    """Run a docker command and return the container ID."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    if proc.returncode != 0:
        error = stderr.decode().strip()
        raise RuntimeError(f"Docker run failed (rc={proc.returncode}): {error}")
    return stdout.decode().strip()


async def stop_container(name: str, timeout: int = 30) -> None:
    """Stop and remove a container by name. Ignores errors if not running."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "stop", "-t", "10", name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (asyncio.TimeoutError, Exception) as e:
        logger.debug("Container stop failed (may not exist): %s", e)

    # Force remove if still exists
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "rm", "-f", name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=10)
    except Exception:
        pass


async def wait_for_healthy(
    url: str,
    timeout_sec: int = 300,
    poll_interval: float = 5.0,
) -> bool:
    """Poll a health endpoint until it returns 200 or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout_sec
    attempts = 0
    async with aiohttp.ClientSession() as session:
        while asyncio.get_event_loop().time() < deadline:
            attempts += 1
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        logger.info("Health check passed: %s", url)
                        return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            if attempts % 12 == 0:  # every ~60 seconds
                elapsed = int(timeout_sec - (deadline - asyncio.get_event_loop().time()))
                logger.info("  Still waiting for health check... (%ds/%ds)", elapsed, timeout_sec)
            await asyncio.sleep(poll_interval)
    logger.error("Health check timed out after %ds (%d attempts): %s", timeout_sec, attempts, url)
    return False


async def get_container_logs(name: str, tail: int = 100) -> str:
    """Get the last N lines of container logs."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "logs", "--tail", str(tail), name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
        return stdout.decode() + stderr.decode()
    except Exception as e:
        return f"Failed to get logs: {e}"


def stop_all_bench_containers() -> None:
    """Stop all containers with bench- prefix. For cleanup."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", "name=bench-"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        container_ids = result.stdout.strip().split("\n")
        for cid in container_ids:
            if cid.strip():
                subprocess.run(
                    ["docker", "rm", "-f", cid.strip()],
                    capture_output=True,
                    timeout=10,
                )
    except Exception as e:
        logger.warning("Cleanup failed: %s", e)
