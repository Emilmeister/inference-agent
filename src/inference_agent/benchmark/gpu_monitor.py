"""GPU metrics collector using nvidia-smi."""

from __future__ import annotations

import asyncio
import csv
import io
import logging

from inference_agent.models import GPUMetricsSnapshot

logger = logging.getLogger(__name__)


class GPUMonitor:
    """Collects GPU metrics in the background using nvidia-smi dmon."""

    def __init__(self, interval_ms: int = 1000) -> None:
        self.interval_ms = interval_ms
        self._process: asyncio.subprocess.Process | None = None
        self._snapshots: list[list[GPUMetricsSnapshot]] = []
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start collecting GPU metrics in the background."""
        self._running = True
        self._snapshots = []
        self._task = asyncio.create_task(self._collect_loop())

    async def stop(self) -> list[list[GPUMetricsSnapshot]]:
        """Stop collecting and return all snapshots."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        return self._snapshots

    async def _collect_loop(self) -> None:
        """Periodically query nvidia-smi."""
        while self._running:
            try:
                snapshot = await self._query_gpus()
                if snapshot:
                    self._snapshots.append(snapshot)
            except Exception as e:
                logger.debug("GPU monitor query failed: %s", e)
            await asyncio.sleep(self.interval_ms / 1000)

    async def _query_gpus(self) -> list[GPUMetricsSnapshot]:
        """Run nvidia-smi and parse the output."""
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        output = stdout.decode().strip()
        if not output:
            return []

        snapshots = []
        reader = csv.reader(io.StringIO(output))
        for row in reader:
            if len(row) < 6:
                continue
            parts = [p.strip() for p in row]
            try:
                snapshots.append(GPUMetricsSnapshot(
                    gpu_index=int(parts[0]),
                    utilization_percent=float(parts[1]),
                    memory_used_mb=float(parts[2]),
                    memory_total_mb=float(parts[3]),
                    power_draw_watts=float(parts[4]),
                    temperature_celsius=float(parts[5]),
                ))
            except (ValueError, IndexError):
                continue
        return snapshots

    @staticmethod
    def aggregate_snapshots(
        all_snapshots: list[list[GPUMetricsSnapshot]],
    ) -> dict[int, dict]:
        """Aggregate GPU snapshots into per-GPU averages and peaks.

        Returns: {gpu_index: {util_avg, mem_peak, power_avg, temp_max}}
        """
        if not all_snapshots:
            return {}

        gpu_data: dict[int, dict[str, list[float]]] = {}
        for snapshot_group in all_snapshots:
            for snap in snapshot_group:
                if snap.gpu_index not in gpu_data:
                    gpu_data[snap.gpu_index] = {
                        "util": [],
                        "mem": [],
                        "power": [],
                        "temp": [],
                    }
                gpu_data[snap.gpu_index]["util"].append(snap.utilization_percent)
                gpu_data[snap.gpu_index]["mem"].append(snap.memory_used_mb)
                gpu_data[snap.gpu_index]["power"].append(snap.power_draw_watts)
                gpu_data[snap.gpu_index]["temp"].append(snap.temperature_celsius)

        result = {}
        for idx, data in sorted(gpu_data.items()):
            result[idx] = {
                "util_avg": sum(data["util"]) / len(data["util"]),
                "mem_peak": max(data["mem"]),
                "power_avg": sum(data["power"]) / len(data["power"]),
                "temp_max": max(data["temp"]),
            }
        return result
