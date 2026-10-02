#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write the P5 summary JSON for one variant/mode session directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import bench_metrics as bm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--sessions", type=int, required=True)
    ap.add_argument("--wall-s", type=int, default=None)
    ap.add_argument("--server-log", default="")
    ap.add_argument("--log-offset", type=int, default=0)
    ap.add_argument("--log-end", type=int, default=None, help="byte end of the P5 window")
    a = ap.parse_args()
    log_text = None
    if a.server_log and Path(a.server_log).exists():
        raw = Path(a.server_log).read_bytes()[a.log_offset : a.log_end]
        log_text = raw.decode("utf-8", "replace")
    s = bm.summarize_p5(a.dir, a.variant, a.mode, a.sessions, a.wall_s, log_text)
    dest = a.dir.parent / f"p5-{a.variant}-{a.mode}-summary.json"
    dest.write_text(json.dumps(s, indent=2))
    print(
        json.dumps({k: v for k, v in s.items() if k != "rows"}),
    )


if __name__ == "__main__":
    main()
