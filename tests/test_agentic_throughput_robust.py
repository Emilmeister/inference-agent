"""Straggler-robust agentic throughput: Σ(tokensᵢ / tᵢ) over sessions.

The agentic phase wall_time = the slowest session's completion, so dividing
total tokens by it lets one tail straggler (invisible in p95 latencies) deflate
the whole phase and create spurious throughput dips. Crediting each session at
its own rate fixes that while preserving the aggregate scale in the uniform
case. These tests pin both properties.
"""

from __future__ import annotations

from inference_agent.benchmark.runner import _aggregate_session_throughputs


def _turn(out: int, inp: int, e2e_ms: float, error: str | None = None) -> dict:
    return {"output_tokens": out, "input_tokens": inp, "e2e_latency_ms": e2e_ms, "error": error}


def test_uniform_case_matches_naive_aggregate():
    # 4 identical sessions, 1 turn each: 100 out tok in 1s. Naive aggregate
    # Σout/wall = 400/1 = 400. Robust Σ(o/t) = 4 * (100/1) = 400 — identical.
    sessions = [[_turn(100, 1000, 1000.0)] for _ in range(4)]
    out_tps, in_tps, total_tps, req_tps = _aggregate_session_throughputs(sessions)
    assert out_tps == 400.0          # == naive Σout/wall_time
    assert in_tps == 4000.0
    assert total_tps == 4400.0
    assert req_tps == 4.0


def test_straggler_does_not_deflate_other_sessions():
    # 3 fast sessions (100 tok in 1s → 100 tok/s each) + 1 straggler that took
    # 10s for the same 100 tok (10 tok/s). Naive aggregate would divide ALL 400
    # tokens by wall_time≈10s = 40 tok/s (the dip). Robust = 100+100+100+10 = 310.
    sessions = [[_turn(100, 1000, 1000.0)] for _ in range(3)]
    sessions.append([_turn(100, 1000, 10_000.0)])  # straggler
    out_tps, *_ = _aggregate_session_throughputs(sessions)
    naive = 400 / 10.0  # Σtokens / wall_time(slowest)
    assert out_tps == 310.0
    assert out_tps > naive  # robust is NOT dragged down to 40 by the straggler


def test_multi_turn_session_active_time_sums_turn_latencies():
    # one session, 2 turns: (50 tok, 0.5s) + (150 tok, 1.5s) → 200 tok / 2.0s = 100.
    sessions = [[_turn(50, 500, 500.0), _turn(150, 1500, 1500.0)]]
    out_tps, _, _, req_tps = _aggregate_session_throughputs(sessions)
    assert out_tps == 100.0
    assert req_tps == 1.0  # 2 successful turns / 2.0s


def test_errored_turn_excluded_from_requests_but_time_counts():
    # 2 turns, second errored (0 tokens, but took 1s before failing).
    sessions = [[_turn(100, 1000, 1000.0), _turn(0, 0, 1000.0, error="timeout")]]
    out_tps, _, _, req_tps = _aggregate_session_throughputs(sessions)
    assert out_tps == 100 / 2.0      # 100 tok over the 2s the session was active
    assert req_tps == 1 / 2.0        # only 1 successful turn counted


def test_zero_active_time_session_skipped():
    sessions = [[_turn(0, 0, 0.0, error="connection refused")], [_turn(100, 1000, 1000.0)]]
    out_tps, *_ = _aggregate_session_throughputs(sessions)
    assert out_tps == 100.0  # the t=0 session contributes nothing, no ZeroDivision


def test_empty_input_is_zero():
    assert _aggregate_session_throughputs([]) == (0.0, 0.0, 0.0, 0.0)
