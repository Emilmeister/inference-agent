"""Async HTTP load generator for LLM benchmarking."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time

import aiohttp

from inference_agent.models import (
    AgenticTurnMetric,
    BenchmarkConfig,
    ConcurrencyResult,
    PercentileStats,
)

logger = logging.getLogger(__name__)


# Substrings indicating the upstream HTTP server is no longer reachable —
# socket-level failures we get back as request errors when the engine has
# crashed or its container exited mid-benchmark. Used by phase watchdog to
# abort fire-into-the-void loops instead of burning the configured duration
# stuffing thousands of <5ms ECONNREFUSED bounces into error_details.
_CONNECTION_FAILURE_SUBSTRINGS: tuple[str, ...] = (
    "cannot connect to host",
    "connect call failed",
    "connection refused",
    "connection reset",
    "server disconnected",
    "econnrefused",
    "[errno 111]",
)


def _is_connection_failure(error: str | None) -> bool:
    """Return True when the request error looks like the server is down.

    Conservative: only matches socket-level failure markers. Stream errors
    mid-response, JSON parse failures, HTTP 5xx — all kept out so a server
    that's responding but unhappy with a single bad request doesn't trigger
    the watchdog.
    """
    if not error:
        return False
    e = error.lower()
    return any(s in e for s in _CONNECTION_FAILURE_SUBSTRINGS)


# Phase-level watchdog: once at least this many requests have come back,
# AND they're all socket-level failures, abort the phase early. The minimum
# probe count keeps a single flaky request from killing an otherwise healthy
# phase, while the all-or-nothing check is safe because partial connectivity
# (some succeed, some fail) does NOT trip it.
_PHASE_WATCHDOG_MIN_PROBES = 10


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
    mean = sum(s) / n
    if n >= 2:
        var = sum((x - mean) ** 2 for x in s) / (n - 1)
        stdev = math.sqrt(var)
    else:
        stdev = 0.0
    cv = stdev / mean if mean > 0 else 0.0
    return PercentileStats(
        mean=mean,
        median=_percentile(s, 0.50),
        p75=_percentile(s, 0.75),
        p90=_percentile(s, 0.90),
        p95=_percentile(s, 0.95),
        p99=_percentile(s, 0.99),
        min=s[0],
        max=s[-1],
        stdev=stdev,
        cv=cv,
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


async def _stream_chat_completion(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    timeout_sec: int,
) -> dict:
    """Stream a chat-completion request and collect timing + accumulated content.

    Single source of truth for SSE parsing — used by both single-shot
    `_send_request` и multi-turn `_send_agent_turn`. Returns:
      ttft_ms, tpot_ms, itl_ms_list, e2e_latency_ms, output_tokens,
      input_tokens (из usage.prompt_tokens или 0), error, token_count_source,
      content (накопленный assistant-text для multi-turn)
    """
    result = {
        "ttft_ms": 0.0,
        "tpot_ms": 0.0,
        "itl_ms_list": [],
        "e2e_latency_ms": 0.0,
        "output_tokens": 0,
        "input_tokens": 0,
        "error": None,
        "token_count_source": "sse_delta",  # or "usage_api"
        "content": "",
    }

    start_time = time.perf_counter()
    first_token_time = None
    last_token_time = None
    token_count = 0
    usage_completion_tokens = 0
    usage_prompt_tokens = 0
    content_parts: list[str] = []

    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_sec),
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
                            pt = usage.get("prompt_tokens")
                            if pt and pt > 0:
                                usage_prompt_tokens = pt

                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            # Accumulate visible text content for multi-turn.
                            # Only standard `content` field — reasoning/tool
                            # calls не предназначены для скармливания обратно
                            # в messages как assistant-content.
                            content_field = delta.get("content")
                            if isinstance(content_field, str) and content_field:
                                content_parts.append(content_field)
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

    if usage_prompt_tokens > 0:
        result["input_tokens"] = usage_prompt_tokens

    result["content"] = "".join(content_parts)

    # Some engines return HTTP 200 with an error JSON or an empty stream when
    # the request is invalid (e.g. context length exceeded for a 100K prompt
    # against a model that doesn't support it). Without this check the runner
    # would count those as successful zero-token responses, producing bogus
    # phases like "121 requests, throughput=2 tok/s, ttft_p95=0.0 ms,
    # errors=0" — and the phase error_rate gate wouldn't trigger.
    if result["output_tokens"] == 0:
        result["error"] = "Empty response: no tokens streamed (HTTP 200)"

    return result


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

    result = await _stream_chat_completion(session, url, payload, timeout_sec=300)

    # Fall back to rough input estimate if the engine didn't return prompt_tokens.
    if result["input_tokens"] == 0:
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
    workload_id: str = "",
    phase_id: str = "",
) -> ConcurrencyResult:
    """Run a single benchmark phase with given concurrency and prompt length."""
    url = f"{api_base_url}/chat/completions"

    # Create per-phase RNG from seed for reproducibility
    rng = random.Random(seed)

    all_results: list[dict] = []
    start_time = time.perf_counter()
    # Watchdog signal — once flipped, every worker drops out of its loop on
    # the next iteration. Asyncio is single-threaded so we don't need a Lock.
    server_down = False

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:

        async def _worker():
            nonlocal server_down
            while not server_down and time.perf_counter() - start_time < duration_sec:
                res = await _send_request(
                    session, url, prompt_length, max_output_tokens, model_name, rng,
                )
                all_results.append(res)
                # Phase watchdog: every accumulated request gives us a fresh
                # chance to notice the engine has died. We check the LAST
                # N results (so a death mid-phase trips it, not only one at
                # t=0), and require all of them to look socket-broken before
                # we abort.
                if (
                    not server_down
                    and len(all_results) >= _PHASE_WATCHDOG_MIN_PROBES
                    and all(
                        _is_connection_failure(r.get("error"))
                        for r in all_results[-_PHASE_WATCHDOG_MIN_PROBES:]
                    )
                ):
                    server_down = True
                    elapsed = time.perf_counter() - start_time
                    logger.warning(
                        "Phase aborted after %.1fs: last %d requests all "
                        "failed with connection errors — engine appears dead.",
                        elapsed, _PHASE_WATCHDOG_MIN_PROBES,
                    )

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
    total_requests = len(all_results)

    result = ConcurrencyResult(
        concurrency=concurrency,
        prompt_length=prompt_length,
        max_output_tokens=max_output_tokens,
        num_requests=total_requests,
        workload_id=workload_id,
        phase_id=phase_id,
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
        error_rate=errors / total_requests if total_requests > 0 else 0.0,
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


# ── Agentic long-context (multi-turn code-agent simulation) ──────────────


async def _send_agent_turn(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    messages: list[dict],
    max_output_tokens: int,
    rng: random.Random,
    timeout_sec: int,
) -> dict:
    """Send one turn of a multi-turn agentic conversation.

    Same return shape as _stream_chat_completion (timing + content), но шлёт
    готовый messages-список (не synthetic single prompt). Used by
    `_run_agent_session`.
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_output_tokens,
        "temperature": round(rng.random(), 2),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    return await _stream_chat_completion(session, url, payload, timeout_sec=timeout_sec)


async def _run_agent_session(
    session_idx: int,
    http_session: aiohttp.ClientSession,
    url: str,
    model: str,
    prefix_tokens: int,
    turns: int,
    max_output_tokens: int,
    tool_result_min: int,
    tool_result_max: int,
    rng: random.Random,
    per_turn_timeout_sec: int,
) -> list[dict]:
    """Run one full agentic session of N turns, return per-turn metric dicts.

    Сессия:
      1. Префикс ~prefix_tokens (как в обычном _generate_prompt).
      2. На каждом ходе:
         - Отправляем messages → получаем assistant content + метрики
         - Дописываем assistant message в history
         - Дописываем synthetic tool-result (rng.randint(min,max) токенов) как user
      3. Если ход вернул error — break (нет смысла продолжать сломанную сессию).

    Returns list[dict] per turn:
      session_idx, turn_idx, ttft_ms, tpot_ms, e2e_latency_ms, output_tokens,
      input_tokens, error.
    `input_tokens` берём из usage.prompt_tokens; если нет — оцениваем как
    len(joined messages) // 4 (та же rough формула что и в _send_request).
    """
    prefix = _generate_prompt(prefix_tokens, rng)
    messages: list[dict] = [{"role": "user", "content": prefix}]
    per_turn: list[dict] = []

    for turn_idx in range(turns):
        result = await _send_agent_turn(
            http_session, url, model, messages, max_output_tokens,
            rng, per_turn_timeout_sec,
        )

        # Fallback rough input estimate if engine не вернул usage.prompt_tokens.
        if result["input_tokens"] == 0:
            joined = sum(len(m.get("content", "")) for m in messages)
            result["input_tokens"] = joined // 4

        per_turn.append({
            "session_idx": session_idx,
            "turn_idx": turn_idx,
            "ttft_ms": result["ttft_ms"],
            "tpot_ms": result["tpot_ms"],
            "itl_ms_list": result["itl_ms_list"],
            "e2e_latency_ms": result["e2e_latency_ms"],
            "output_tokens": result["output_tokens"],
            "input_tokens": result["input_tokens"],
            "error": result["error"],
        })

        if result["error"]:
            break

        # Append assistant response and a simulated tool-call result for the next turn.
        assistant_content = result["content"]
        if not assistant_content:
            # No visible content but no error — defensively stop the session
            # (we have nothing to feed into the next turn's history).
            break
        messages.append({"role": "assistant", "content": assistant_content})

        tool_len = rng.randint(tool_result_min, tool_result_max)
        tool_text = _generate_prompt(tool_len, rng)
        messages.append({
            "role": "user",
            "content": f"[tool_result]\n{tool_text}",
        })

    return per_turn


async def run_agentic_long_context_phase(
    api_base_url: str,
    model_name: str,
    concurrency: int,
    prefix_tokens: int,
    max_output_tokens: int,
    turns: int,
    tool_result_min: int,
    tool_result_max: int,
    session_timeout_sec: int,
    per_turn_timeout_sec: int,
    seed: int | None = None,
    workload_id: str = "agentic_long_context",
    phase_id: str = "",
) -> ConcurrencyResult:
    """Run one agentic long-context phase: N parallel sessions × `turns` turns each.

    Sample size = concurrency × turns (e.g. 16 × 4 = 64 turn-requests).
    Phase ends когда все сессии завершились или истёк session_timeout_sec
    per worker. Per-turn metrics складываются в `agentic_turn_metrics` поле
    результата — это ключевой данные для дашборд-аналитики (TTFT vs turn_idx).
    """
    url = f"{api_base_url}/chat/completions"
    master_rng = random.Random(seed)

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    start_time = time.perf_counter()

    async with aiohttp.ClientSession(connector=connector) as http_session:

        async def _worker(idx: int) -> list[dict]:
            # Per-session RNG, derived from master so the whole phase is reproducible.
            session_rng = random.Random(master_rng.randint(0, 2**31 - 1))
            try:
                return await asyncio.wait_for(
                    _run_agent_session(
                        idx, http_session, url, model_name,
                        prefix_tokens, turns, max_output_tokens,
                        tool_result_min, tool_result_max,
                        session_rng, per_turn_timeout_sec,
                    ),
                    timeout=session_timeout_sec,
                )
            except asyncio.TimeoutError:
                return [{
                    "session_idx": idx,
                    "turn_idx": 0,
                    "ttft_ms": 0.0,
                    "tpot_ms": 0.0,
                    "itl_ms_list": [],
                    "e2e_latency_ms": session_timeout_sec * 1000.0,
                    "output_tokens": 0,
                    "input_tokens": 0,
                    "error": f"Session timeout after {session_timeout_sec}s",
                }]

        per_session_results = await asyncio.gather(
            *[_worker(i) for i in range(concurrency)]
        )

    wall_time = time.perf_counter() - start_time

    # Flatten per-turn results from all sessions.
    flat: list[dict] = []
    for session_results in per_session_results:
        flat.extend(session_results)

    ttft_list: list[float] = []
    tpot_list: list[float] = []
    itl_list: list[float] = []
    e2e_list: list[float] = []
    total_output_tokens = 0
    total_input_tokens = 0
    errors = 0
    error_details: list[str] = []
    turn_metrics: list[AgenticTurnMetric] = []

    for r in flat:
        turn_metrics.append(AgenticTurnMetric(
            session_idx=r["session_idx"],
            turn_idx=r["turn_idx"],
            ttft_ms=r["ttft_ms"],
            tpot_ms=r["tpot_ms"],
            e2e_latency_ms=r["e2e_latency_ms"],
            output_tokens=r["output_tokens"],
            input_tokens=r["input_tokens"],
            error=r["error"],
        ))
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

    total_requests = len(flat)
    successful = total_requests - errors

    result = ConcurrencyResult(
        concurrency=concurrency,
        prompt_length=prefix_tokens,
        max_output_tokens=max_output_tokens,
        num_requests=total_requests,
        workload_id=workload_id,
        phase_id=phase_id,
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
        error_rate=errors / total_requests if total_requests > 0 else 0.0,
        error_details=error_details[:10],
        agentic_turn_metrics=turn_metrics,
    )

    logger.info(
        "Agentic phase complete: c=%d, sessions=%d, turns/session=%d, "
        "total_turns=%d, throughput=%.1f tok/s, ttft_p95=%.1f ms, errors=%d",
        concurrency, concurrency, turns, total_requests,
        result.output_tokens_per_sec, result.ttft_ms.p95, errors,
    )

    return result


# ── Benchmark phase matrix ────────────────────────────────────────────────

# Workload classification thresholds
_LONG_PROMPT_THRESHOLD = 8192     # prompt_length >= this → long_context workload
_STRESS_CONCURRENCY = 512         # concurrency >= this → stress workload
_THROUGHPUT_CONCURRENCY = 64      # concurrency >= this (but < stress) → throughput

# Explicit concurrency sweep for long_context phases (decoupled from
# global concurrency_levels — RAG-style workloads care about a different
# slice than peak-throughput sweeps).
_LONG_CONTEXT_CONCURRENCIES: list[int] = [2, 8, 16]


def _classify_workload(concurrency: int, prompt_length: int) -> str:
    """Classify a phase into a workload based on concurrency and prompt length."""
    if prompt_length >= _LONG_PROMPT_THRESHOLD:
        return "long_context"
    if concurrency >= _STRESS_CONCURRENCY:
        return "stress"
    if concurrency >= _THROUGHPUT_CONCURRENCY:
        return "throughput"
    return "agent_short"


def get_benchmark_phases(
    model_max_context: int,
    max_model_len: int | None = None,
    benchmark_config: BenchmarkConfig | None = None,
) -> list[tuple[str, str, int, int, int]]:
    """Return list of (phase_id, workload_id, concurrency, prompt_length, max_output_tokens)
    built from BenchmarkConfig and filtered by model context limits.

    Phases are generated from config.concurrency_levels × config.prompt_lengths,
    classified into workloads, and filtered by effective context window.
    """
    cfg = benchmark_config or BenchmarkConfig()
    effective_max = max_model_len or model_max_context

    phases: list[tuple[str, str, int, int, int]] = []

    # Warmup phase (always first, fixed params)
    phases.append(("warmup", "warmup", 1, min(512, effective_max - 128), 128))

    short_prompts = sorted(p for p in cfg.prompt_lengths if p < _LONG_PROMPT_THRESHOLD)
    long_prompts = sorted(p for p in cfg.prompt_lengths if p >= _LONG_PROMPT_THRESHOLD)

    # Short / throughput / stress phases — driven by global concurrency_levels
    for conc in sorted(cfg.concurrency_levels):
        for plen in short_prompts:
            max_out = cfg.max_output_tokens
            if plen + max_out > effective_max:
                logger.info(
                    "Skipping c=%d p=%d: prompt+output (%d) exceeds context %d",
                    conc, plen, plen + max_out, effective_max,
                )
                continue
            workload = _classify_workload(conc, plen)
            phases.append((f"c{conc}_p{plen}", workload, conc, plen, max_out))

    # Long-context phases — explicit concurrency sweep, independent of global levels
    for conc in _LONG_CONTEXT_CONCURRENCIES:
        for plen in long_prompts:
            max_out = cfg.long_context_max_output_tokens
            if plen + max_out > effective_max:
                logger.info(
                    "Skipping long-context c=%d p=%d: prompt+output (%d) exceeds context %d",
                    conc, plen, plen + max_out, effective_max,
                )
                continue
            phases.append((f"c{conc}_p{plen}", "long_context", conc, plen, max_out))

    # Agentic long-context phases — multi-turn code-agent simulation. Контекст
    # растёт от хода к ходу, поэтому проверяем что max_model_len держит
    # prefix + turns × (max_output + tool_result_max).
    if cfg.enable_agentic_long_context:
        agentic_required = (
            cfg.agentic_prefix_tokens
            + cfg.agentic_turns_per_session
            * (cfg.agentic_max_output_tokens + cfg.agentic_tool_result_max_tokens)
        )
        if agentic_required > effective_max:
            logger.info(
                "Skipping agentic_long_context: required %d (prefix %d + %d turns × "
                "(out %d + tool %d)) exceeds effective context %d",
                agentic_required,
                cfg.agentic_prefix_tokens,
                cfg.agentic_turns_per_session,
                cfg.agentic_max_output_tokens,
                cfg.agentic_tool_result_max_tokens,
                effective_max,
            )
        else:
            for conc in cfg.agentic_concurrency_levels:
                phases.append((
                    f"agentic_c{conc}_p{cfg.agentic_prefix_tokens}",
                    "agentic_long_context",
                    conc,
                    cfg.agentic_prefix_tokens,
                    cfg.agentic_max_output_tokens,
                ))

    # Sort by concurrency for stable, predictable ordering (warmup stays first)
    head, tail = phases[:1], phases[1:]
    tail.sort(key=lambda p: (p[2], p[3]))
    return head + tail
