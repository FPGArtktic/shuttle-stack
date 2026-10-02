# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mateusz Okulanis
"""Whether starting on the small server pays, on this machine.

A cascade asks the cheap profile first and climbs only when the answer
does not verify. That is a saving when the cheap server is cheap. The
small server here is CPU-only while the large one has layers on the
GPU and a draft model in front of it, so "cheap" has to be measured
rather than assumed.

    python -m evals.cascade
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from shuttle_delegate import cascade, tasks
from shuttle_delegate.server import backend

ROOT = Path(__file__).resolve().parents[2]
CASES = Path(__file__).resolve().parent / "cases.json"
LADDER = ("fast", "extract")
TOOLS = ("extract", "ask")


def _run(case: dict, profile: str) -> dict:
    text = tasks.read_text(str(ROOT / case["path"]))
    call = getattr(tasks, case["tool"])
    return call(backend(profile), text, **case.get("args", {}))


def main() -> int:
    cases = [c for c in json.loads(CASES.read_text()) if c["tool"] in TOOLS]
    print(f"{'case':<30} {'direct':>8} {'cascade':>9} {'climbed':>8}")
    totals = {"direct": [0.0, 0], "cascade": [0.0, 0]}
    escalated = climbed = 0
    for case in cases:
        start = time.time()
        straight = _run(case, "extract")
        direct = time.time() - start
        totals["direct"][0] += direct
        totals["direct"][1] += straight.get("local_tokens", 0)

        start = time.time()
        result = cascade.climb(
            cascade.ladder(",".join(LADDER)),
            lambda profile, this=case: _run(this, profile),
        )
        took = time.time() - start
        totals["cascade"][0] += took
        totals["cascade"][1] += result.get("local_tokens", 0)
        steps = len(result["ladder"])
        climbed += steps > 1
        escalated += bool(result.get("escalate"))
        print(f"{case['id']:<30} {direct:>7.1f}s {took:>8.1f}s {steps:>8}")

    print()
    for name, (seconds, tokens) in totals.items():
        print(f"{name:<10} {seconds:>7.0f}s  {tokens:>7} local tokens")
    print(
        f"\n{len(cases)} cases, {climbed} climbed past the first profile, "
        f"{escalated} came back unverified"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
