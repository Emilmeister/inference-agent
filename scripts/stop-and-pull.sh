#!/usr/bin/env bash
# stop-and-pull.sh — gracefully stop a running inference-agent + its benchmark
# containers, then pull the latest code while preserving local changes.
#
# Usage:
#   ./scripts/stop-and-pull.sh           # stop agent + containers, git pull
#   ./scripts/stop-and-pull.sh --restart # also relaunch via nohup after pull
#
# Behavior:
#   1. SIGTERM all `inference-agent` processes; SIGKILL stragglers after 10s.
#   2. `nerdctl kill` any `bench-vllm-*` / `bench-sglang-*` containers.
#   3. If git working tree has local changes, stash them.
#   4. `git pull --ff-only` (refuses to merge — we want fast-forward only).
#   5. `git stash pop` (only if we actually stashed in step 3).
#   6. Optional: relaunch agent under nohup with the project's config.yaml.
#
# Run from the repo root (or anywhere — the script cd's to its own repo root).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RESTART=0
for arg in "$@"; do
    case "$arg" in
        --restart) RESTART=1 ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown arg: $arg" >&2
            exit 2
            ;;
    esac
done

log() { printf '[stop-and-pull] %s\n' "$*"; }

# ── 1. Stop inference-agent processes ────────────────────────────────────
# pgrep -f matches the full command line, catching `python ... inference-agent`
# regardless of how the user launched it (entrypoint, module, nohup, etc.).
mapfile -t AGENT_PIDS < <(pgrep -f 'inference-agent' || true)
# Drop our own PID and any direct ancestor (this script's own grep won't match
# pgrep -f, but be defensive in case the user names their shell session weirdly).
SELF=$$
AGENT_PIDS=("${AGENT_PIDS[@]/$SELF}")
AGENT_PIDS=("${AGENT_PIDS[@]/$PPID}")

if [[ ${#AGENT_PIDS[@]} -eq 0 ]]; then
    log "no inference-agent process running"
else
    log "sending SIGTERM to inference-agent PIDs: ${AGENT_PIDS[*]}"
    kill -TERM "${AGENT_PIDS[@]}" 2>/dev/null || true

    # Give the agent up to 10s to finalize the current experiment, persist DB
    # state, and shut down cleanly.
    for _ in $(seq 1 10); do
        sleep 1
        STILL_ALIVE=()
        for pid in "${AGENT_PIDS[@]}"; do
            [[ -z "$pid" ]] && continue
            if kill -0 "$pid" 2>/dev/null; then
                STILL_ALIVE+=("$pid")
            fi
        done
        AGENT_PIDS=("${STILL_ALIVE[@]}")
        [[ ${#AGENT_PIDS[@]} -eq 0 ]] && break
    done

    if [[ ${#AGENT_PIDS[@]} -gt 0 ]]; then
        log "SIGKILL stragglers: ${AGENT_PIDS[*]}"
        kill -KILL "${AGENT_PIDS[@]}" 2>/dev/null || true
    fi
fi

# ── 2. Kill benchmark containers ─────────────────────────────────────────
# Names follow the `bench-<engine>-<experiment_id>` convention from
# engines/base.py:container_name. Anything else (user's own containers,
# unrelated services) is left alone.
mapfile -t BENCH_CONTAINERS < <(
    nerdctl ps --filter 'name=bench-vllm-' --filter 'name=bench-sglang-' \
        --format '{{.ID}} {{.Names}}' 2>/dev/null || true
)
if [[ ${#BENCH_CONTAINERS[@]} -eq 0 ]]; then
    log "no bench-* containers running"
else
    for line in "${BENCH_CONTAINERS[@]}"; do
        log "nerdctl kill $line"
        cid="${line%% *}"
        nerdctl kill "$cid" >/dev/null 2>&1 || true
    done
fi

# ── 3. Stash local changes (if any) ──────────────────────────────────────
STASHED=0
if ! git diff --quiet || ! git diff --cached --quiet; then
    log "stashing local changes"
    git stash push --include-untracked --message "stop-and-pull-$(date +%s)" >/dev/null
    STASHED=1
else
    log "working tree clean"
fi

# ── 4. Pull latest ───────────────────────────────────────────────────────
log "git pull --ff-only"
git pull --ff-only

# ── 5. Restore stash ─────────────────────────────────────────────────────
if [[ $STASHED -eq 1 ]]; then
    log "restoring stashed changes"
    if ! git stash pop; then
        log "ERROR: stash pop hit a conflict — resolve manually with 'git status'"
        log "your changes are preserved in the stash; run 'git stash list' to find them"
        exit 1
    fi
fi

# ── 6. Optional restart ──────────────────────────────────────────────────
if [[ $RESTART -eq 1 ]]; then
    if [[ ! -f config.yaml ]]; then
        log "ERROR: --restart requested but config.yaml not found in $REPO_ROOT"
        exit 1
    fi
    rm -f nohup.out
    log "relaunching: nohup inference-agent -c config.yaml -v"
    nohup inference-agent -c config.yaml -v >nohup.out 2>&1 &
    NEW_PID=$!
    disown "$NEW_PID" 2>/dev/null || true
    sleep 1
    if kill -0 "$NEW_PID" 2>/dev/null; then
        log "agent started, PID=$NEW_PID, logs: tail -f $REPO_ROOT/nohup.out"
    else
        log "ERROR: agent died immediately — check nohup.out"
        exit 1
    fi
fi

log "done"
