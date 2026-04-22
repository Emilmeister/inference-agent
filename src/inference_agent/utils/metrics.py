"""Prometheus /metrics endpoint parser."""

from __future__ import annotations

import logging
import re

import aiohttp

logger = logging.getLogger(__name__)


async def fetch_prometheus_metrics(url: str) -> dict[str, float]:
    """Fetch and parse Prometheus metrics from a /metrics endpoint.

    Returns a flat dict of metric_name -> value (for gauges/counters).
    """
    metrics: dict[str, float] = {}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    logger.warning("Metrics endpoint returned %d", resp.status)
                    return metrics
                text = await resp.text()
    except (aiohttp.ClientError, Exception) as e:
        logger.warning("Failed to fetch metrics from %s: %s", url, e)
        return metrics

    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Parse: metric_name{labels} value
        match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\b(?:\{[^}]*\})?\s+([\d.eE+-]+|NaN|Inf|-Inf)$', line)
        if match:
            name = match.group(1)
            try:
                value = float(match.group(2))
                metrics[name] = value
            except ValueError:
                pass

    return metrics


def extract_kv_cache_metrics(metrics: dict[str, float], engine: str) -> dict:
    """Extract KV cache related metrics based on engine type."""
    result = {"kv_cache_usage_percent": 0.0, "prefix_cache_hit_rate": 0.0}

    if engine == "vllm":
        result["kv_cache_usage_percent"] = metrics.get(
            "vllm:kv_cache_usage_perc", 0.0
        ) * 100
        hits = metrics.get("vllm:prefix_cache_hits", 0)
        queries = metrics.get("vllm:prefix_cache_queries", 0)
        if queries > 0:
            result["prefix_cache_hit_rate"] = hits / queries
    elif engine == "sglang":
        result["kv_cache_usage_percent"] = metrics.get(
            "sglang:token_usage", 0.0
        ) * 100
        result["prefix_cache_hit_rate"] = metrics.get(
            "sglang:cache_hit_rate", 0.0
        )

    return result
