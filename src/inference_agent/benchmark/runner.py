"""Async HTTP load generator for LLM benchmarking."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time

import aiohttp

from inference_agent.models import ConcurrencyResult, PercentileStats

logger = logging.getLogger(__name__)


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute percentile with linear interpolation (numpy-compatible)."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    k = (n - 1) * p
    f = int(math.floor(k))
    c = min(f + 1, n - 1)
    d = k - f
    return sorted_values[f] + d * (sorted_values[c] - sorted_values[f])


def _compute_percentiles(values: list[float]) -> PercentileStats:
    """Compute percentile statistics from a list of values."""
    if not values:
        return PercentileStats()
    s = sorted(values)
    n = len(s)
    return PercentileStats(
        mean=sum(s) / n,
        median=_percentile(s, 0.50),
        p75=_percentile(s, 0.75),
        p90=_percentile(s, 0.90),
        p95=_percentile(s, 0.95),
        p99=_percentile(s, 0.99),
        min=s[0],
        max=s[-1],
    )


_WORD_POOL = [
    "The", "system", "processes", "data", "through", "multiple", "layers",
    "of", "transformation", "and", "analysis", "to", "produce", "accurate",
    "results", "that", "can", "be", "used", "for", "decision", "making",
    "in", "complex", "environments", "where", "performance", "matters",
    "a", "model", "generates", "tokens", "using", "attention", "mechanism",
    "the", "input", "sequence", "is", "encoded", "into", "hidden", "states",
    "each", "layer", "applies", "normalization", "before", "computing",
    "output", "logits", "are", "projected", "from", "final", "representation",
    "batch", "size", "affects", "throughput", "while", "context", "length",
    "determines", "memory", "requirements", "on", "GPU", "hardware",
    "optimization", "techniques", "include", "quantization", "pruning",
    "speculative", "decoding", "prefix", "caching", "continuous", "batching",
    "server", "handles", "concurrent", "requests", "with", "scheduling",
    "policy", "controls", "request", "priority", "queue", "management",
]

_TASK_PREFIXES = [
    "Please analyze the following text and provide a detailed summary:\n\n",
    "Read the text below carefully and explain the key points:\n\n",
    "Summarize the main ideas from this passage:\n\n",
    "What are the important concepts described in this text?\n\n",
    "Provide a comprehensive analysis of the following:\n\n",
    "Extract and explain the core arguments from this text:\n\n",
    "Review the following content and highlight the main themes:\n\n",
    "Describe what this text is about in detail:\n\n",
]


def _generate_prompt(length_tokens: int, rng: random.Random) -> str:
    """Generate a unique synthetic prompt of approximately `length_tokens` tokens.

    Uses the provided RNG for reproducibility. Each call produces a different
    prompt by shuffling the word pool and picking a random task prefix.
    """
    chars_needed = length_tokens * 4
    words = _WORD_POOL.copy()
    rng.shuffle(words)
    text_parts: list[str] = []
    while len(" ".join(text_parts)) < chars_needed:
        rng.shuffle(words)
        text_parts.extend(words)
    text = " ".join(text_parts)[:chars_needed]
    prefix = rng.choice(_TASK_PREFIXES)
    return f"{prefix}{text}"


async def _send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt_length: int,
    max_tokens: int,
    model: str,
    rng: random.Random,
) -> dict:
    """Send a single chat completion request and measure timing."""
    prompt = _generate_prompt(prompt_length, rng)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": round(rng.random(), 2),
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    result = {
        "ttft_ms": 0.0,
        "tpot_ms": 0.0,
        "itl_ms_list": [],
        "e2e_latency_ms": 0.0,
        "output_tokens": 0,
        "input_tokens": 0,
        "error": None,
        "token_count_source": "sse_delta",  # or "usage_api"
    }

    start_time = time.perf_counter()
    first_token_time = None
    last_token_time = None
    token_count = 0
    usage_completion_tokens = 0

    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                result["error"] = f"HTTP {resp.status}: {body[:200]}"
                return result

            # Read SSE stream line by line (handles buffering correctly)
            buffer = ""
            async for chunk in resp.content.iter_any():
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line_str, buffer = buffer.split("\n", 1)
                    line_str = line_str.strip()
                    # Handle both "data: {...}" and "data:{...}"
                    if not line_str.startswith("data:"):
                        continue
                    data_str = line_str[5:].lstrip()
                    if data_str == "[DONE]":
                        break

                    now = time.perf_counter()
                    try:
                        data = json.loads(data_str)

                        # Capture usage (usually in the last event)
                        usage = data.get("usage")
                        if usage and isinstance(usage, dict):
                            ct = usage.get("completion_tokens")
                            if ct and ct > 0:
                                usage_completion_tokens = ct

                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            # Check all text fields in delta for token content
                            # (different engines use different field names:
                            #  content, reasoning_content, reasoning, etc.)
                            has_token = any(
                                isinstance(v, str) and len(v) > 0
                                for k, v in delta.items()
                                if k not in ("role", "tool_calls", "function_call", "refusal")
                            )
                            if has_token:
                                token_count += 1
                                if first_token_time is None:
                                    first_token_time = now
                                else:
                                    result["itl_ms_list"].append(
                                        (now - last_token_time) * 1000
                                    )
                                last_token_time = now
                    except json.JSONDecodeError:
                        pass
                    except (KeyError, IndexError):
                        pass

    except asyncio.TimeoutError:
        result["error"] = "Request timed out"
        return result
    except aiohttp.ClientError as e:
        result["error"] = str(e)
        return result

    end_time = time.perf_counter()
    result["e2e_latency_ms"] = (end_time - start_time) * 1000

    # Prefer usage-reported token count when available (more accurate)
    if usage_completion_tokens > 0:
        result["output_tokens"] = usage_completion_tokens
        result["token_count_source"] = "usage_api"
    else:
        result["output_tokens"] = token_count
        result["token_count_source"] = "sse_delta"

    if first_token_time is not None:
        result["ttft_ms"] = (first_token_time - start_time) * 1000

    if token_count > 1 and first_token_time and last_token_time:
        decode_time = (last_token_time - first_token_time) * 1000
        result["tpot_ms"] = decode_time / (token_count - 1)

    # Rough input token estimate (~4 chars per token)
    result["input_tokens"] = len(prompt) // 4

    return result


async def run_benchmark_phase(
    api_base_url: str,
    model_name: str,
    concurrency: int,
    prompt_length: int,
    max_output_tokens: int,
    duration_sec: int = 60,
    warmup: bool = False,
    seed: int | None = None,
) -> ConcurrencyResult:
    """Run a single benchmark phase with given concurrency and prompt length."""
    url = f"{api_base_url}/chat/completions"

    # Create per-phase RNG from seed for reproducibility
    rng = random.Random(seed)

    all_results: list[dict] = []
    start_time = time.perf_counter()

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:

        async def _worker():
            while time.perf_counter() - start_time < duration_sec:
                res = await _send_request(
                    session, url, prompt_length, max_output_tokens, model_name, rng,
                )
                all_results.append(res)

        tasks = [asyncio.create_task(_worker()) for _ in range(concurrency)]
        await asyncio.gather(*tasks)

    wall_time = time.perf_counter() - start_time

    # Aggregate results
    ttft_list = []
    tpot_list = []
    itl_list = []
    e2e_list = []
    total_output_tokens = 0
    total_input_tokens = 0
    errors = 0
    error_details: list[str] = []

    for r in all_results:
        if r["error"]:
            errors += 1
            error_details.append(r["error"])
            continue
        if r["ttft_ms"] > 0:
            ttft_list.append(r["ttft_ms"])
        if r["tpot_ms"] > 0:
            tpot_list.append(r["tpot_ms"])
        itl_list.extend(r["itl_ms_list"])
        if r["e2e_latency_ms"] > 0:
            e2e_list.append(r["e2e_latency_ms"])
        total_output_tokens += r["output_tokens"]
        total_input_tokens += r["input_tokens"]

    successful = len(all_results) - errors

    result = ConcurrencyResult(
        concurrency=concurrency,
        prompt_length=prompt_length,
        max_output_tokens=max_output_tokens,
        num_requests=len(all_results),
        ttft_ms=_compute_percentiles(ttft_list),
        tpot_ms=_compute_percentiles(tpot_list),
        itl_ms=_compute_percentiles(itl_list),
        e2e_latency_ms=_compute_percentiles(e2e_list),
        requests_per_sec=successful / wall_time if wall_time > 0 else 0,
        input_tokens_per_sec=total_input_tokens / wall_time if wall_time > 0 else 0,
        output_tokens_per_sec=total_output_tokens / wall_time if wall_time > 0 else 0,
        total_tokens_per_sec=(total_input_tokens + total_output_tokens) / wall_time
        if wall_time > 0
        else 0,
        errors=errors,
        error_details=error_details[:10],  # cap at 10
    )

    if not warmup:
        logger.info(
            "Phase complete: concurrency=%d, prompt_len=%d, requests=%d, "
            "throughput=%.1f tok/s, ttft_p95=%.1f ms, errors=%d",
            concurrency,
            prompt_length,
            len(all_results),
            result.output_tokens_per_sec,
            result.ttft_ms.p95,
            errors,
        )

    return result


# ── Benchmark phase matrix ────────────────────────────────────────────────

# Default phases when not overridden by config
_DEFAULT_PHASES = [
    # (phase_name, concurrency_levels, prompt_lengths, max_output_tokens, is_long_context)
    ("warmup", [1], [512], 128, False),
    ("latency", [1], [128, 512, 2048, 4096], 256, False),
    ("mid_throughput", [4, 16, 64], [512, 2048], 256, False),
    ("high_throughput", [128, 256], [512], 256, False),
    ("stress", [512], [512], 256, False),
    ("long_context_16k", [1, 4], [16384], 8192, True),
    ("long_context_24k", [1, 4], [24576], 8192, True),
    ("long_context_32k", [1, 4], [32768], 8192, True),
    ("long_context_64k", [1, 4], [65536], 8192, True),
    ("long_context_100k", [1, 2], [100000], 8192, True),
]


def get_benchmark_phases(
    model_max_context: int,
    max_model_len: int | None = None,
    benchmark_config: object | None = None,
) -> list[tuple[str, int, int, int]]:
    """Return list of (phase_name, concurrency, prompt_length, max_output_tokens)
    filtered by model context limits.

    If benchmark_config is provided and has custom concurrency_levels/prompt_lengths,
    those are used to build the phase matrix. Otherwise the default matrix is used.
    """
    effective_max = max_model_len or model_max_context
    phases = []
    for name, concurrencies, prompt_lengths, max_out, is_long in _DEFAULT_PHASES:
        for conc in concurrencies:
            for plen in prompt_lengths:
                # Skip if prompt + output exceeds context
                if plen + max_out > effective_max:
                    logger.info(
                        "Skipping %s (c=%d, p=%d): exceeds context %d",
                        name, conc, plen, effective_max,
                    )
                    continue
                phases.append((name, conc, plen, max_out))
    return phases
