#!/usr/bin/env bash
# stop-and-pull.sh — stop the running inference-agent + its benchmark
# containers, then pull the latest code while preserving local changes.
#
# Usage:
#   ./scripts/stop-and-pull.sh           # stop agent + containers, git pull
#   ./scripts/stop-and-pull.sh --restart # also relaunch via scripts/start.sh
#
# Behavior:
#   1. Source .env so HTTP_PROXY and the API token are present for `git pull`
#      (locked-down VMs need the proxy) and for the optional restart.
#   2. SIGTERM all inference-agent processes via sudo (the agent runs as
#      root under nerdctl); SIGKILL stragglers after 10s.
#   3. `nerdctl kill` any bench-vllm-* / bench-sglang-* containers (also
#      under sudo for the same reason).
#   4. Stash local changes (if any).
#   5. `git pull --ff-only` as the invoking user — git refuses to use ssh
#      keys it can't read, and sudo would strip HTTP_PROXY anyway.
#   6. Pop the stash.
#   7. Optional: relaunch the agent via scripts/start.sh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RESTART=0
for arg in "$@"; do
    case "$arg" in
        --restart) RESTART=1 ;;
        -h|--help)
            sed -n '2,22p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown arg: $arg" >&2
            exit 2
            ;;
    esac
done

log() { printf '[stop-and-pull] %s\n' "$*"; }

# sudo prefix — empty if we're already root, otherwise `sudo`. The agent
# runs under root and we mustn't fail killing it from a non-root user.
if [[ $EUID -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
fi

# Load .env so the rest of the script has HTTP_PROXY (for git pull) and
# the agent's secrets (for optional --restart). Missing .env is non-fatal
# for stop+pull on a fully open network, but warn so it's not silent.
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    . .env
    set +a
else
    log "WARNING: no .env in $REPO_ROOT — git pull may fail behind a proxy"
fi

# ── 1. Stop inference-agent processes ────────────────────────────────────
# Match against the absolute path of the entrypoint binary, NOT the bare
# string "inference-agent". Bare-string pgrep matches the script's own
# command line and ends up killing the shell that's running this script.
AGENT_PATTERN="/inference-agent\\b"

mapfile -t AGENT_PIDS < <($SUDO pgrep -f "$AGENT_PATTERN" 2>/dev/null || true)
SELF=$$
# Defensive: drop our own PID + parent in case pgrep ever races against
# our own bash process (it shouldn't with the path-anchored pattern, but
# this costs nothing).
FILTERED=()
for pid in "${AGENT_PIDS[@]}"; do
    [[ -z "$pid" || "$pid" == "$SELF" || "$pid" == "$PPID" ]] && continue
    FILTERED+=("$pid")
done
AGENT_PIDS=("${FILTERED[@]}")

if [[ ${#AGENT_PIDS[@]} -eq 0 ]]; then
    log "no inference-agent process running"
else
    log "sending SIGTERM to inference-agent PIDs: ${AGENT_PIDS[*]}"
    $SUDO kill -TERM "${AGENT_PIDS[@]}" 2>/dev/null || true

    # Give the agent up to 10s to finalize the current experiment and shut
    # down cleanly (closes the API session, flushes logs).
    for _ in $(seq 1 10); do
        sleep 1
        STILL_ALIVE=()
        for pid in "${AGENT_PIDS[@]}"; do
            if $SUDO kill -0 "$pid" 2>/dev/null; then
                STILL_ALIVE+=("$pid")
            fi
        done
        AGENT_PIDS=("${STILL_ALIVE[@]}")
        [[ ${#AGENT_PIDS[@]} -eq 0 ]] && break
    done

    if [[ ${#AGENT_PIDS[@]} -gt 0 ]]; then
        log "SIGKILL stragglers: ${AGENT_PIDS[*]}"
        $SUDO kill -KILL "${AGENT_PIDS[@]}" 2>/dev/null || true
    fi
fi

# ── 2. Kill benchmark containers ─────────────────────────────────────────
mapfile -t BENCH_CONTAINERS < <(
    $SUDO nerdctl ps --filter 'name=bench-vllm-' --filter 'name=bench-sglang-' \
        --format '{{.ID}} {{.Names}}' 2>/dev/null || true
)
if [[ ${#BENCH_CONTAINERS[@]} -eq 0 ]]; then
    log "no bench-* containers running"
else
    for line in "${BENCH_CONTAINERS[@]}"; do
        log "nerdctl kill $line"
        cid="${line%% *}"
        $SUDO nerdctl kill "$cid" >/dev/null 2>&1 || true
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
# Run as the invoking user (not via $SUDO) so HTTP_PROXY from .env is
# honored and the user's git credentials/ssh config are used. If you ran
# `sudo ./scripts/stop-and-pull.sh`, $SUDO is empty above and this just
# stays as root — make sure root has access to whatever git remote you use.
log "git pull --ff-only"
if ! git pull --ff-only; then
    log "ERROR: git pull failed; check HTTP_PROXY in .env and ssh/credentials"
    [[ $STASHED -eq 1 ]] && log "your changes are preserved in stash@{0}"
    exit 1
fi

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
    log "relaunching via scripts/start.sh"
    bash "$REPO_ROOT/scripts/start.sh"
fi

log "done"
