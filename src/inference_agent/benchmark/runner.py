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


# ── Agentic SLO gate ──────────────────────────────────────────────────────


class AgenticSLO:
    """Per-phase SLO contract for agentic_long_context.

    Plain immutable bundle the runner uses to decide if a finished phase is
    "viable" — i.e. could serve real production agent traffic at this
    concurrency. ALL gates must hold for the phase to be marked viable;
    otherwise we record reasons in `slo_violations` so the executor can
    route the phase to ceiling probes with a meaningful explanation.
    """

    __slots__ = (
        "ttft_p95_ms", "tpot_p95_ms", "e2e_p95_ms", "session_error_rate",
    )

    def __init__(
        self,
        *,
        ttft_p95_ms: float,
        tpot_p95_ms: float,
        e2e_p95_ms: float,
        session_error_rate: float,
    ):
        self.ttft_p95_ms = ttft_p95_ms
        self.tpot_p95_ms = tpot_p95_ms
        self.e2e_p95_ms = e2e_p95_ms
        self.session_error_rate = session_error_rate

    @classmethod
    def from_config(cls, cfg: BenchmarkConfig) -> "AgenticSLO":
        e2e_ms = cfg.agentic_e2e_p95_slo_ms
        if e2e_ms <= 0:
            # Auto: 80% of session timeout — leaves headroom for tail.
            e2e_ms = cfg.agentic_session_timeout_sec * 1000.0 * 0.8
        return cls(
            ttft_p95_ms=cfg.agentic_ttft_p95_slo_ms,
            tpot_p95_ms=cfg.agentic_tpot_p95_slo_ms,
            e2e_p95_ms=e2e_ms,
            session_error_rate=cfg.agentic_session_error_rate_slo,
        )


def _evaluate_agentic_slo(
    *,
    ttft_p95: float,
    tpot_p95: float,
    e2e_p95: float,
    error_rate: float,
    slo: AgenticSLO,
) -> list[str]:
    """Return the list of SLO violations (empty list = viable).

    Each violation is a short string suitable for surfacing in
    CeilingProbeInfo.reason or analyzer logs. Zero/missing percentile values
    don't trip the gate (a phase with no successful samples gets caught by
    the error_rate gate instead).
    """
    violations: list[str] = []
    if error_rate > slo.session_error_rate:
        violations.append(
            f"error_rate={error_rate:.1%} > slo={slo.session_error_rate:.0%}"
        )
    if ttft_p95 > 0 and ttft_p95 > slo.ttft_p95_ms:
        violations.append(
            f"ttft_p95={ttft_p95:.0f}ms > slo={slo.ttft_p95_ms:.0f}ms"
        )
    if tpot_p95 > 0 and tpot_p95 > slo.tpot_p95_ms:
        violations.append(
            f"tpot_p95={tpot_p95:.1f}ms > slo={slo.tpot_p95_ms:.1f}ms"
        )
    if e2e_p95 > 0 and e2e_p95 > slo.e2e_p95_ms:
        violations.append(
            f"e2e_p95={e2e_p95 / 1000:.0f}s > slo={slo.e2e_p95_ms / 1000:.0f}s"
        )
    return violations

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
        "cached_tokens": 0,
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
    usage_cached_tokens = 0
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
                            # Prefix-cache hits: prompt tokens served from KV
                            # cache (~0 prefill compute). OpenAI-compatible
                            # engines (vLLM, SGLang) report this under
                            # usage.prompt_tokens_details.cached_tokens. Used to
                            # separate logical input from real prefill work.
                            details = usage.get("prompt_tokens_details")
                            if isinstance(details, dict):
                                cached = details.get("cached_tokens")
                                if cached and cached > 0:
                                    usage_cached_tokens = cached

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

    # Cached tokens are only meaningful when the engine actually reported the
    # prompt-token usage; clamp to input so a malformed payload can't push
    # real_prefill negative downstream.
    if usage_cached_tokens > 0 and result["input_tokens"] > 0:
        result["cached_tokens"] = min(usage_cached_tokens, result["input_tokens"])

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
    total_cached_tokens = 0
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
        total_cached_tokens += r.get("cached_tokens", 0)

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
        cached_tokens_per_sec=total_cached_tokens / wall_time if wall_time > 0 else 0,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cached_tokens=total_cached_tokens,
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
    shared_prefix: str,
    unique_prompt_tokens: int,
    turns: int,
    max_output_tokens: int,
    tool_result_min: int,
    tool_result_max: int,
    rng: random.Random,
    per_turn_timeout_sec: int,
) -> list[dict]:
    """Run one full agentic session of N turns, return per-turn metric dicts.

    The first user message is `shared_prefix + unique_tail`. The shared
    prefix is the SAME string for every session in the phase (built once by
    the caller from a deterministic seed) so the engine's prefix cache can
    reuse the KV — that is the whole point of agentic-first sizing.

    Sequence:
      1. messages[0] = shared_prefix + unique_tail (unique per session).
      2. For each turn:
         - Send messages, capture assistant content + metrics.
         - Append assistant message + synthetic tool_result user-turn.
      3. Stop the session on the first error turn (no point pumping more
         turns into a broken conversation).

    Returns one dict per turn: session_idx, turn_idx, ttft_ms, tpot_ms,
    e2e_latency_ms, output_tokens, input_tokens, error.
    """
    unique_tail = _generate_prompt(unique_prompt_tokens, rng)
    first_user = f"{shared_prefix}\n\n{unique_tail}" if shared_prefix else unique_tail
    messages: list[dict] = [{"role": "user", "content": first_user}]
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
            "cached_tokens": result.get("cached_tokens", 0),
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


def build_shared_prefix(shared_prefix_tokens: int, seed: int | None) -> str:
    """Build the shared per-phase agentic prefix (deterministic from seed).

    Used by `run_agentic_long_context_phase` and exposed for tests so they
    can verify all sessions in a phase see literally the same tokens (→ KV
    cache reuse). With seed=None the prefix is random per phase but still
    shared within the phase.
    """
    if shared_prefix_tokens <= 0:
        return ""
    rng = random.Random(seed if seed is not None else 0)
    return _generate_prompt(shared_prefix_tokens, rng)


def _aggregate_session_throughputs(
    per_session_results: list[list[dict]],
) -> tuple[float, float, float, float, float]:
    """Straggler-robust agentic throughput = Σ over sessions of (tokensᵢ / tᵢ).

    Returns ``(output_tps, input_tps, total_tps, requests_tps, cached_tps)``.
    ``tᵢ`` is the session's active generation time — the sum of its turns' e2e
    latencies (turns run back-to-back, so this is the real wall span spent
    generating, excluding any simulated tool gaps). ``cached_tps`` is the rate
    of prefix-cache-hit prompt tokens — a subset of ``input_tps`` that cost ~0
    prefill compute, so ``input_tps - cached_tps`` is the real prefill rate.

    Why not ``Σtokens / wall_time``: wall_time is the SLOWEST session's
    completion, so one tail straggler (invisible in p95 latencies) deflates the
    whole phase and produces spurious throughput dips. Crediting each session at
    its OWN rate makes a straggler one small term instead of a global divisor.
    In the uniform (equal-tᵢ) case this is algebraically identical to
    ``Σtokens / wall_time``, so the aggregate scale is preserved — only the
    straggler artifact is removed. Sessions with ``tᵢ <= 0`` contribute nothing.
    """
    out_tps = in_tps = total_tps = req_tps = cached_tps = 0.0
    for session_results in per_session_results:
        o_i = sum(r["output_tokens"] for r in session_results)
        in_i = sum(r["input_tokens"] for r in session_results)
        cached_i = sum(r.get("cached_tokens", 0) for r in session_results)
        succ_i = sum(1 for r in session_results if not r.get("error"))
        t_i = sum(r["e2e_latency_ms"] for r in session_results) / 1000.0
        if t_i <= 0:
            continue
        out_tps += o_i / t_i
        in_tps += in_i / t_i
        total_tps += (o_i + in_i) / t_i
        req_tps += succ_i / t_i
        cached_tps += cached_i / t_i
    return out_tps, in_tps, total_tps, req_tps, cached_tps


async def run_agentic_long_context_phase(
    api_base_url: str,
    model_name: str,
    concurrency: int,
    shared_prefix_tokens: int,
    unique_prompt_tokens: int,
    max_output_tokens: int,
    turns: int,
    tool_result_min: int,
    tool_result_max: int,
    session_timeout_sec: int,
    per_turn_timeout_sec: int,
    seed: int | None = None,
    workload_id: str = "agentic_long_context",
    phase_id: str = "",
    slo: AgenticSLO | None = None,
) -> ConcurrencyResult:
    """Run one agentic long-context phase: N parallel sessions × `turns` turns.

    Workload shape per session:
      messages[0].content = <shared_prefix> + <unique_tail>     # first user turn
      Then `turns - 1` additional rounds where the assistant replies and a
      synthetic tool_result user message is appended.

    `shared_prefix` is built ONCE per phase (`build_shared_prefix`) so every
    parallel session sees the same tokens — engines with prefix caching
    reuse the KV across sessions instead of paying the prefill cost per
    session. This is the single biggest lever for agentic concurrency.

    If `slo` is provided, the result is marked `viable=False` and
    `slo_violations` is populated when any SLO is missed. Non-agentic
    callers leave `slo=None` and the phase is always viable.
    """
    url = f"{api_base_url}/chat/completions"
    master_rng = random.Random(seed)

    # Build shared prefix once, deterministic from seed — every session in
    # this phase sees these exact tokens.
    shared_prefix = build_shared_prefix(shared_prefix_tokens, seed)

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
                        shared_prefix, unique_prompt_tokens,
                        turns, max_output_tokens,
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

    total_requests = len(flat)

    # Raw token totals across all turns (errored turns contribute ~0). Kept
    # alongside the straggler-robust rates so the benchmark-level prefix hit
    # rate can be computed token-weighted (Σcached / Σinput) rather than from
    # per-session rates with differing denominators.
    total_input_tokens = sum(r["input_tokens"] for r in flat)
    total_output_tokens = sum(r["output_tokens"] for r in flat)
    total_cached_tokens = sum(r.get("cached_tokens", 0) for r in flat)

    ttft_stats = _compute_percentiles(ttft_list)
    tpot_stats = _compute_percentiles(tpot_list)
    itl_stats = _compute_percentiles(itl_list)
    e2e_stats = _compute_percentiles(e2e_list)
    error_rate = errors / total_requests if total_requests > 0 else 0.0

    # Straggler-robust throughput: Σ over sessions of (tokensᵢ / active_timeᵢ),
    # instead of Σtokens / wall_time which a single tail straggler deflates.
    # See _aggregate_session_throughputs for the rationale.
    out_tps, in_tps, total_tps, req_tps, cached_tps = _aggregate_session_throughputs(
        per_session_results
    )

    # SLO gate. None = no SLO requested (legacy callers); otherwise we mark
    # the phase non-viable on the first failed condition. Viable phases
    # become the basis for max_viable_agentic_concurrency.
    slo_violations: list[str] = []
    if slo is not None:
        slo_violations = _evaluate_agentic_slo(
            ttft_p95=ttft_stats.p95,
            tpot_p95=tpot_stats.p95,
            e2e_p95=e2e_stats.p95,
            error_rate=error_rate,
            slo=slo,
        )

    result = ConcurrencyResult(
        concurrency=concurrency,
        prompt_length=shared_prefix_tokens + unique_prompt_tokens,
        max_output_tokens=max_output_tokens,
        num_requests=total_requests,
        workload_id=workload_id,
        phase_id=phase_id,
        ttft_ms=ttft_stats,
        tpot_ms=tpot_stats,
        itl_ms=itl_stats,
        e2e_latency_ms=e2e_stats,
        requests_per_sec=req_tps,
        input_tokens_per_sec=in_tps,
        output_tokens_per_sec=out_tps,
        total_tokens_per_sec=total_tps,
        cached_tokens_per_sec=cached_tps,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cached_tokens=total_cached_tokens,
        errors=errors,
        error_rate=error_rate,
        error_details=error_details[:10],
        agentic_turn_metrics=turn_metrics,
        viable=(slo is None) or (not slo_violations),
        slo_violations=slo_violations,
    )

    viable_tag = "VIABLE" if result.viable else f"NOT-VIABLE({','.join(slo_violations)})"
    logger.info(
        "Agentic phase complete: c=%d, sessions=%d, turns/session=%d, "
        "total_turns=%d, wall=%.1fs, throughput=%.1f tok/s (Σ per-session, "
        "straggler-robust), ttft_p95=%.0f ms, tpot_p95=%.1f ms, errors=%d → %s",
        concurrency, concurrency, turns, total_requests, wall_time,
        result.output_tokens_per_sec, result.ttft_ms.p95, result.tpot_ms.p95,
        errors, viable_tag,
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
    # растёт от хода к ходу: считаем shared + unique + turns × (output + tool).
    if cfg.enable_agentic_long_context:
        total_prompt = cfg.agentic_shared_prefix_tokens + cfg.agentic_unique_prompt_tokens
        agentic_required = (
            total_prompt
            + cfg.agentic_turns_per_session
            * (cfg.agentic_max_output_tokens + cfg.agentic_tool_result_max_tokens)
        )
        if agentic_required > effective_max:
            logger.info(
                "Skipping agentic_long_context: required %d (shared %d + unique %d "
                "+ %d turns × (out %d + tool %d)) exceeds effective context %d",
                agentic_required,
                cfg.agentic_shared_prefix_tokens,
                cfg.agentic_unique_prompt_tokens,
                cfg.agentic_turns_per_session,
                cfg.agentic_max_output_tokens,
                cfg.agentic_tool_result_max_tokens,
                effective_max,
            )
        else:
            for conc in cfg.agentic_concurrency_levels:
                phases.append((
                    f"agentic_c{conc}_p{total_prompt}",
                    "agentic_long_context",
                    conc,
                    total_prompt,
                    cfg.agentic_max_output_tokens,
                ))

    # Sort by concurrency for stable, predictable ordering (warmup stays first)
    head, tail = phases[:1], phases[1:]
    tail.sort(key=lambda p: (p[2], p[3]))
    return head + tail
