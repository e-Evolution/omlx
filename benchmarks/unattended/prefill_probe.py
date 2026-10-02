"""Fast single-request prefill probe (prompt_tokens / TTFT) for bisecting.

Starts one isolated fork-off server on port 8002 from whichever ``omlx``
package ``PYTHONPATH`` selects, warms it up, then measures ``--runs`` unique
cold prompts of ``--tokens`` tokens with max_tokens=1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_premises as bp  # noqa: E402


async def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(bp.BENCH_BASE_ROOT) / "probe-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    corpus = bp.Corpus(bp.REAL_BASE / "models" / "Jundot" / bp.MODEL, cache_dir)
    srv = bp.Server("fork-off", "PROBE", args.port, 1, out, False)
    srv.prepare()
    srv.start()
    res = []
    try:
        srv.wait_ready()
        async with bp.make_client() as client:
            ctx = bp.Ctx(srv, corpus, client, argparse.Namespace(smoke=False), run=0)
            w = await ctx.chat(corpus.prompt("probe-warm", 3000), 1, name="warm")
            print("warm ok", w.get("ok"), w.get("ttft_s"), flush=True)
            for i in range(args.runs):
                p = corpus.prompt(f"probe-{args.tag}-{i}-{time.time()}", args.tokens)
                r = await ctx.chat(p, 1, name=f"p{i}")
                pt, ttft = r.get("prompt_tokens"), r.get("ttft_s")
                tps = pt / ttft if pt and ttft else None
                res.append(tps)
                print(f"[probe] {args.tag} run{i} tokens={pt} ttft={ttft} tps={tps}", flush=True)
    finally:
        srv.stop()
        srv.archive_and_clean()
    good = [x for x in res if x]
    summary = {"tag": args.tag, "tps": res, "median": statistics.median(good) if good else None}
    print("[probe-summary]", json.dumps(summary), flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--tokens", type=int, default=40000)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--out", default="/tmp/omlx-unattended-bench/probe-out")
    asyncio.run(run(ap.parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
