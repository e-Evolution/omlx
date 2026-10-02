# Unattended-scheduling premise benchmark

Compares the official oMLX v0.7.0 (`upstream`) with this fork (`fork`, optionally
`fork-off` = fork code with the four features disabled) on the same machine,
model and inputs, following the "Performance PRs" section of
`docs/CONTRIBUTING.md`.

Premises: P1 no lost work after a mid-prefill cancel, P2 no aborts with 2 x ~130k
concurrent, P3 greedy batch correctness (B=3 == B=1), P4 concurrency pays,
P6 no single-request regression (110k prefill/decode within -5%), P5 unattended
pi batch (separate script).

## Isolation and safety

* Each (variant, premise) starts its **own** server on an isolated base path
  `~/.omlx-bench/<variant>-<premise>/` (a copy of `~/.omlx/settings.json` and
  `model_settings.json`; model dir stays the real, read-only `~/.omlx/models`).
  The real `~/.omlx` is never written; the harness prints a before/after hash
  check of `settings.json`.
* Overrides in the copy: `server.port`, `cache.ssd_cache_dir` (fresh per start),
  `memory_guard_tier=aggressive`, `max_concurrent_requests` per premise.
* Only PIDs of servers started by this process are ever signalled (taken from
  its own subprocess handles, plus recorded descendants checked by start time).
  There is no pkill/killall/pgrep-by-name anywhere. Never run anything that
  matches processes by name next to this harness.
* Port 8001 (the live app) is refused unless `--allow-port 8001`; a port that
  already accepts connections is refused.
* Upstream = `/Applications/oMLX.app/Contents/MacOS/omlx-cli serve` (read-only
  use of the bundle); the GUI binary is never executed.
* The SSD cache must initialise: the harness checks the log for
  `PagedSSDCacheManager initialized` and aborts (exit 3) on "cache disabled" or
  "Operation not permitted". Launch from a terminal that has access to the
  external volume. The volume root is not writable from a terminal session, so
  the bench caches live in `<volume>/.omlx/oLMXCache-bench/<variant>-<premise>`
  (override: `--cache-root`). Never point it at the live `oLMXCache`.
* Results, samples and server logs go to `--out` (default
  `/tmp/omlx-unattended-bench`), outside git.

## Reproduce

Use the fork venv (`httpx`, `transformers` are installed there).

```bash
cd <fork repo>
PY=.venv/bin/python

# Unit tests of the pure logic (no servers):
$PY -m pytest tests/test_bench_unattended.py -q

# Smoke (tiny prompts, port 8002, results flagged invalid as evidence):
$PY benchmarks/unattended/bench_premises.py --variant fork --premise P6 --runs 1 --smoke --out /tmp/bench-smoke

# Full run, one variant after the other (premises P6 P1 P2 P3 P4 in that order):
$PY benchmarks/unattended/bench_premises.py --variant upstream --premise all --runs 3
$PY benchmarks/unattended/bench_premises.py --variant fork     --premise all --runs 3
# optional attribution:
$PY benchmarks/unattended/bench_premises.py --variant fork-off --premise all --runs 3

# Report (markdown, also written to <out>/report.md):
$PY benchmarks/unattended/report.py --out /tmp/omlx-unattended-bench
```

`--premise {P1,P2,P3,P4,P6,all}`, `--runs N`, `--port` (default 8002),
`--p1-cancel-fraction` (0.6), `--p3-max-tokens` (512), `--keep-cache`.

Method notes:

* Prompts come from a deterministic corpus (the repo's own `omlx/*.py`), tokenised
  with the model tokenizer, with a unique nonce prefix per prompt, so runs do not
  share prefix cache unless a scenario intends it (P1 re-issue, P3 cached arm).
  Seeds do not include the variant: upstream and fork get identical prompts.
  Real `prompt_tokens` come from the stream `usage` frame.
* Streaming with `stream_options.include_usage`. An SSE error frame, a missing
  usage frame, a missing `[DONE]` or no generated text is a FAILURE.
* TTFT = first generated text (content or reasoning); prefill tok/s =
  prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (end - first token).
* Every second: `footprint` (phys_footprint of the server process tree) and
  `/health`. Server log windows (byte offsets) are parsed for aborts, pressure
  levels, `pool=`, parks and skips, hold/release, prefix-cache restores,
  `MLX buffer cache limit`, `TurboQuant batched attention`.
* Warm-up: one ~2k-token request (also loads the model) before the first
  measured run. Cache state: cold (nonce-prefixed, fresh SSD cache dir).
* The server is restarted per (variant, premise), not per run.
* P1: calibration request (same length, `max_tokens=1`) measures full TTFT; the
  real request is cancelled at `fraction x TTFT`; estimated progress =
  `fraction x prompt_tokens` (prefill slows with context, so this is
  conservative); the identical prompt is re-issued and the cached tokens are read
  from `Prefix cache restore ... cached=N`. Pass: cached >= 50% of progress.
* P3: three prompts (20k/45k/135k), temperature 0, top_k 1, like-for-like:
  cold B=1 (only populates the prefix cache), B=1 restored (reference), B=3
  restored (started together so all rows decode in one batch). Fork variants run
  with `OMLX_MAX_BATCHED_DECODE_CONTEXT_TOKENS=0`. PASS: B=3 equals the
  B=1-restored text for the first 32 tokens' worth of chunks on every row, no
  degeneration (distinct word-4gram ratio >= 0.5x the B=1 value and >= 0.3), and
  the log shows a 3-row batch (`Lightning MTP batch activated for 3 sequences`
  or `per-row fused path (B=3)`). First divergence and the metric are reported
  per row for every variant; upstream is reported, not judged.
* P4: serial = client sends one at a time, concurrent = all three at once, on
  distinct prompts of the same lengths (both cold), server
  `max_concurrent_requests=3`. PASS: 0 aborts, concurrent wall <= serial wall,
  and the aggregate decode rate over the window where all three rows decode
  (per-chunk timestamps) >= the median single-row serial rate. Per-row ratios
  are informational. `--p4-max-tokens` widens the all-three window.
* P6: `--runs 3` reports median and min-max. Variants for attribution:
  `upstream`, `fork-off`, `fork`, plus `fork-d1only` and `fork-cachelimit-only`.
* `fork-off` cannot disable the T5d per-row TurboQuant fix (it is a bug fix),
  only the four env features.

## P5 (pi-driven)

`p5_pi_unattended.sh` runs N concurrent `pi -p` sessions and counts those that
finish without a human. pi's omlx provider targets **port 8001**, so the bench
server has to listen there while P5 runs, after the oMLX Fork app is stopped:

```bash
# terminal 1: isolated fork server on 8001 (own base path and cache dir)
$PY benchmarks/unattended/bench_premises.py --variant fork --serve --port 8001 \
    --allow-port 8001 --max-concurrent 3
# wait for the READY line, then terminal 2:
SERVER_LOG=~/.omlx-bench/fork-P6/logs/server.log MODE=with   benchmarks/unattended/p5_pi_unattended.sh
SERVER_LOG=~/.omlx-bench/fork-P6/logs/server.log MODE=without VARIANT=fork \
  PROVIDER_EXT=<path of the pi-omlx provider extension> benchmarks/unattended/p5_pi_unattended.sh
# stop terminal 1 with Ctrl-C when done.
```

The user's pi key is the one copied from `~/.omlx/settings.json` into the bench
base path, so pi authenticates without extra setup. Summaries land in
`<out>/p5/p5-<variant>-<mode>-summary.json` (session files in
`<out>/p5/<variant>-<mode>/`, `VARIANT` defaults to `fork`; use `VARIANT=upstream`
against the upstream server so runs never overwrite each other). A session is a
memory failure only when an oMLX error signature appears in stderr or on an
`Error:` stdout line (`bench_metrics.classify_p5_session`); a signature followed
by exit 0 with a full answer counts as recovered, i.e. success. The P5 verdict
passes when fork-with has 0 manual interventions and upstream-without shows the
failure mode. `report.py` includes it.
