"""quality_finalize node — prod-readiness validation of the finalists.

Runs ONLY after the optimization loop converges (analyzer decided stop). For
each finalist leaderboard winner it:

  1. recovers the finalist's launch config from the API,
  2. groups finalists by quality fingerprint (so the costly terminal-bench
     runs once per quality-equivalent group, not per finalist),
  3. relaunches the finalist container once per group,
  4. runs the enabled suites (so-testing, terminal-bench) against it,
  5. persists each suite's result as a quality_run (idempotent — a suite whose
     run is already `done` is skipped, so Ctrl+C mid-run resumes cleanly).

Report-only: never mutates leaderboards/Pareto or the planner. A suite failure
is recorded and the run continues with the other finalists/suites.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from inference_agent.api_client import APIClientError, ExperimentApiClient
from inference_agent.engines.sglang import SGLangEngine
from inference_agent.engines.vllm import VLLMEngine
from inference_agent.models import AgentConfig, EngineType, ExperimentResult, HardwareProfile
from inference_agent.models_pkg.config import QualityConfig
from inference_agent.nodes.executor import _start_engine
from inference_agent.quality.fingerprint import quality_fingerprint
from inference_agent.quality.finalists import distinct_experiment_ids, select_finalist_refs
from inference_agent.quality.runner import run_so_testing, run_terminal_bench
from inference_agent.state import AgentState
from inference_agent.utils.container import stop_container

logger = logging.getLogger(__name__)

QualityFinalizeNode = Callable[[AgentState], Awaitable[dict]]


def _engine_for(config: AgentConfig, engine_type: EngineType):
    return VLLMEngine(config) if engine_type == EngineType.VLLM else SGLangEngine(config)


def _row_payload(
    *,
    group_key: str,
    fingerprint: str,
    suite: str,
    hardware: HardwareProfile,
    model: str,
    experiment_ids: list[str],
    categories: list[str],
    status: str,
    score: float | None = None,
    suite_version: str = "",
    data: dict[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    primary = hardware.gpus[0] if hardware.gpus else None
    return {
        "id": f"{group_key}-{suite}",
        "fingerprint": fingerprint,
        "suite": suite,
        "suite_version": suite_version,
        "model_name": model,
        "gpu_name": primary.name if primary else "",
        "gpu_count": hardware.gpu_count,
        "gpu_vram_mb": primary.vram_total_mb if primary else 0,
        "nvlink_available": hardware.nvlink_available,
        "status": status,
        "score": score,
        "error": error,
        "experiment_ids": experiment_ids,
        "categories": categories,
        "data": data or {},
    }


async def _validate_group(
    client: ExperimentApiClient,
    config: AgentConfig,
    qcfg: QualityConfig,
    group_key: str,
    fingerprint: str,
    result: ExperimentResult,
    experiment_ids: list[str],
    categories: list[str],
) -> None:
    """Relaunch one finalist (representing its fingerprint group) and run the
    enabled, not-yet-done suites against it."""
    hardware = result.hardware
    model = result.model
    finalist_config = result.config

    wanted: list[str] = []
    if qcfg.so_testing.enabled:
        wanted.append("so_testing")
    if qcfg.terminal_bench.enabled:
        wanted.append("terminal_bench")

    # Idempotency: skip suites already marked done for this group.
    pending: list[str] = []
    for suite in wanted:
        existing = await client.get_quality_run(f"{group_key}-{suite}")
        if existing and existing.get("status") == "done":
            logger.info("quality_finalize: %s-%s already done — skipping", group_key, suite)
            continue
        pending.append(suite)
    if not pending:
        return

    def payload(suite: str, **kw) -> dict[str, Any]:
        return _row_payload(
            group_key=group_key, fingerprint=fingerprint, suite=suite,
            hardware=hardware, model=model,
            experiment_ids=experiment_ids, categories=categories, **kw,
        )

    # Mark running (resumability signal) before the expensive relaunch.
    for suite in pending:
        await client.upsert_quality_run(payload(suite, status="running"))

    engine = _engine_for(config, finalist_config.engine)
    container_name = engine.container_name(finalist_config)
    container_args = engine.build_container_args(finalist_config)

    logger.info(
        "quality_finalize: relaunching finalist %s (engine=%s, fp=%s) for %s",
        finalist_config.experiment_id, finalist_config.engine.value,
        fingerprint, ", ".join(pending),
    )
    container_id, errors, _tth = await _start_engine(
        engine, container_args, container_name, finalist_config.experiment_id,
    )
    if container_id is None:
        msg = "relaunch failed: " + "; ".join(e.message for e in errors)
        logger.warning("quality_finalize: %s", msg)
        for suite in pending:
            await client.upsert_quality_run(payload(suite, status="failed", error=msg))
        return

    base_url = engine.api_base_url()
    try:
        if "so_testing" in pending:
            try:
                sr = await run_so_testing(qcfg.so_testing, base_url, model)
            except Exception as e:  # noqa: BLE001 — never let one suite kill the node
                logger.exception("quality_finalize: so-testing crashed")
                sr = None
                await client.upsert_quality_run(
                    payload("so_testing", status="failed", error=f"runner crashed: {e}")
                )
            if sr is not None:
                await client.upsert_quality_run(payload(
                    "so_testing", status=sr.status, score=sr.score,
                    suite_version=sr.suite_version, data=sr.data, error=sr.error,
                ))

        if "terminal_bench" in pending:
            try:
                tr = await run_terminal_bench(
                    qcfg.terminal_bench, base_url, model, fingerprint,
                )
            except Exception as e:  # noqa: BLE001
                logger.exception("quality_finalize: terminal-bench crashed")
                tr = None
                await client.upsert_quality_run(
                    payload("terminal_bench", status="failed", error=f"runner crashed: {e}")
                )
            if tr is not None:
                await client.upsert_quality_run(payload(
                    "terminal_bench", status=tr.status, score=tr.score,
                    suite_version=tr.suite_version, data=tr.data, error=tr.error,
                ))
    finally:
        await stop_container(container_name)


def make_quality_finalize_node(client: ExperimentApiClient) -> QualityFinalizeNode:
    """Build the quality_finalize node closing over the API client."""

    async def quality_finalize_node(state: AgentState) -> dict:
        config: AgentConfig = state["config"]
        qcfg = config.quality
        if not qcfg.enabled:
            return {}
        if not (qcfg.so_testing.enabled or qcfg.terminal_bench.enabled):
            return {}

        refs = select_finalist_refs(state, qcfg.finalists)
        if not refs:
            logger.info("quality_finalize: no finalists to validate")
            return {}

        # Fetch full results for the distinct finalist experiment ids.
        results: dict[str, ExperimentResult] = {}
        for eid in distinct_experiment_ids(refs):
            try:
                results[eid] = await client.get_experiment(eid)
            except APIClientError as e:
                logger.warning("quality_finalize: cannot fetch finalist %s: %s", eid, e)
        if not results:
            logger.warning("quality_finalize: no finalist results could be fetched")
            return {}

        # Group finalists. With dedup, the group key is the quality fingerprint
        # (one expensive run per quality-equivalent group). Without it, each
        # finalist is its own group (keyed by experiment id) so nothing merges.
        groups: dict[str, dict[str, Any]] = {}
        for ref in refs:
            result = results.get(ref.experiment_id)
            if result is None:
                continue
            fingerprint = quality_fingerprint(result.config, result.hardware, result.model)
            group_key = fingerprint if qcfg.fingerprint_dedup else ref.experiment_id
            group = groups.setdefault(
                group_key,
                {"fingerprint": fingerprint, "result": result,
                 "experiment_ids": [], "categories": []},
            )
            if ref.experiment_id not in group["experiment_ids"]:
                group["experiment_ids"].append(ref.experiment_id)
            group["categories"].append(ref.category)

        logger.info(
            "quality_finalize: validating %d finalist group(s) "
            "(so_testing=%s, terminal_bench=%s)",
            len(groups), qcfg.so_testing.enabled, qcfg.terminal_bench.enabled,
        )
        for group_key, group in groups.items():
            try:
                await _validate_group(
                    client, config, qcfg, group_key, group["fingerprint"],
                    group["result"], group["experiment_ids"], group["categories"],
                )
            except Exception:  # noqa: BLE001 — one bad finalist must not abort the rest
                logger.exception(
                    "quality_finalize: group %s validation failed", group_key
                )
        return {}

    return quality_finalize_node
