#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# P5 - unattended pi batch against an isolated bench server.
#
# Runs N concurrent `pi -p` sessions, each fed a large, distinct file set, and
# counts how many finish without a human. Run it twice (or MODE=both): once
# with pi-omlx-unattended loaded ("with") and once without it ("without").
#
# pi's omlx provider targets port 8001, so the BENCH SERVER must listen on
# 8001 while this runs. Start it (after the oMLX Fork app is stopped) with:
#   .venv/bin/python benchmarks/unattended/bench_premises.py --variant fork \
#       --serve --port 8001 --allow-port 8001 --max-concurrent 3
# That server uses an isolated base path and cache dir, never ~/.omlx. Stop it
# with Ctrl-C / SIGTERM to the bench_premises.py process (it only signals the
# server PIDs it started). This script only talks to PORT and kills only the
# pi PIDs it spawned itself.
#
# Env / flags (defaults in brackets):
#   PORT [8001]          server port (informational check + log pointer)
#   SESSIONS [3]         concurrent pi sessions
#   PROVIDER [omlx]      pi provider name;  MODEL [Qwen3.8-27B-oQ4e-mtp]
#   MODE [with]          with | without | both
#   VARIANT [fork]       fork | upstream; output goes to $OUT/p5/<VARIANT>-<MODE>
#                        so the two servers never overwrite each other
#   PROVIDER_EXT []      path of the pi-omlx provider extension; REQUIRED for
#                        "without": runs `pi -ne -e $PROVIDER_EXT` so only the
#                        provider is loaded (no pi-omlx-unattended).
#   TIMEOUT_S [5400]     per-session wall limit (session counted as needing a human)
#   SERVER_LOG []        bench server log; abort lines are counted from it
#   OUT [/tmp/omlx-unattended-bench]
#   REPO [omlx-fork root]  where the attached files come from
set -u

PORT="${PORT:-8001}"
SESSIONS="${SESSIONS:-3}"
PROVIDER="${PROVIDER:-omlx}"
MODEL="${MODEL:-Qwen3.8-27B-oQ4e-mtp}"
MODE="${MODE:-with}"
VARIANT="${VARIANT:-fork}"
TIMEOUT_S="${TIMEOUT_S:-5400}"
SERVER_LOG="${SERVER_LOG:-}"
OUT="${OUT:-/tmp/omlx-unattended-bench}/p5"
REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
PROVIDER_EXT="${PROVIDER_EXT:-}"

# Distinct large file sets (about 100k+ tokens each), one per session slot.
file_set() {
  case "$1" in
    0) echo "omlx/scheduler.py" ;;
    1) echo "omlx/engine_core.py omlx/process_memory_enforcer.py omlx/server.py" ;;
    2) echo "omlx/cache/paged_ssd_cache.py omlx/cache/prefix_cache.py omlx/settings.py" ;;
    *) echo "omlx/scheduler.py omlx/server.py" ;;
  esac
}

curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 \
  || { echo "no healthy server on port ${PORT}" >&2; exit 2; }

run_mode() {
  local mode="$1" dir="$OUT/${VARIANT}-$1" pids=() start end i
  mkdir -p "$dir"
  local ext_args=()
  if [ "$mode" = "without" ]; then
    [ -n "$PROVIDER_EXT" ] || { echo "MODE=without needs PROVIDER_EXT" >&2; return 2; }
    ext_args=(-ne -e "$PROVIDER_EXT")
  fi
  local log_off=0
  [ -n "$SERVER_LOG" ] && [ -f "$SERVER_LOG" ] && log_off=$(wc -c < "$SERVER_LOG")
  start=$(date +%s)
  for ((i = 0; i < SESSIONS; i++)); do
    local files=() f
    for f in $(file_set "$i"); do files+=("@$REPO/$f"); done
    (
      SECONDS=0
      cd "$REPO" || exit 99
      pi -p --no-session --provider "$PROVIDER" --model "$MODEL" ${ext_args[@]+"${ext_args[@]}"} \
        "${files[@]}" \
        "Read the attached files and write about 200 words per file on its responsibilities. Do not call tools." \
        >"$dir/session$i.out" 2>"$dir/session$i.err"
      echo $? >"$dir/session$i.exit"
      echo "$SECONDS" >"$dir/session$i.wall"
    ) &
    pids+=($!)
    sleep 2
  done
  # Watchdog: kill only the sessions this script spawned.
  (
    sleep "$TIMEOUT_S"
    for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done
  ) &
  local wd=$!
  for p in "${pids[@]}"; do wait "$p" 2>/dev/null; done
  kill "$wd" 2>/dev/null
  end=$(date +%s)

  python3 "$(dirname "$0")/p5_summarize.py" "$dir" --variant "$VARIANT" --mode "$mode" \
    --sessions "$SESSIONS" --wall-s "$((end - start))" \
    --server-log "$SERVER_LOG" --log-offset "$log_off"
}

case "$MODE" in
  both) run_mode with; run_mode without ;;
  with|without) run_mode "$MODE" ;;
  *) echo "MODE must be with|without|both" >&2; exit 2 ;;
esac
