# SPDX-License-Identifier: GPL-3.0-only
"""Run the evaluation cases against a live stack.

Five hand-picked examples tell you what a model can do on a good day.
A fixed set with known answers tells you how often it does it, which is
the number worth arguing from. Nothing here runs in CI: it needs the
servers up, and it reports rather than gates.

    python -m evals.run --server long
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from shuttle_delegate import tasks
from shuttle_delegate.server import backend

ROOT = Path(__file__).resolve().parents[2]
CASES = Path(__file__).resolve().parent / "cases.json"


@dataclass
class Outcome:
    """One case on one server."""

    case: str
    tool: str
    passed: bool
    seconds: float
    tokens: int
    why: str = ""
    misses: list[str] = field(default_factory=list)


def _text(result: dict) -> str:
    for key in ("answer", "summary", "label"):
        if key in result:
            return str(result[key])
    return json.dumps(result.get("fields", result))


def check(expect: dict, result: dict) -> list[str]:
    """Every way the result failed to meet the expectation."""
    misses: list[str] = []
    body = _text(result).lower()
    for key, want in expect.get("fields", {}).items():
        got = result.get("fields", {}).get(key)
        if got != want:
            misses.append(f"{key}={got!r} want {want!r}")
    for key, want in expect.get("field_contains", {}).items():
        got = str(result.get("fields", {}).get(key, ""))
        if want.lower() not in got.lower():
            misses.append(f"{key}={got!r} lacks {want!r}")
    for want in expect.get("contains", []):
        if want.lower() not in body:
            misses.append(f"lacks {want!r}")
    any_of = expect.get("contains_any", [])
    if any_of and not any(want.lower() in body for want in any_of):
        misses.append(f"lacks any of {any_of}")
    for unwanted in expect.get("absent", []):
        if unwanted.lower() in body:
            misses.append(f"holds {unwanted!r}")
    if "found" in expect and result.get("found") != expect["found"]:
        misses.append(f"found={result.get('found')} want {expect['found']}")
    if "label" in expect and result.get("label") != expect["label"]:
        misses.append(f"label={result.get('label')!r}")
    return misses


def run_case(case: dict, role: str) -> Outcome:
    server = backend(role)
    text = tasks.read_text(str(ROOT / case["path"]))
    call = getattr(tasks, case["tool"])
    started = time.time()
    try:
        result = call(server, text, **case.get("args", {}))
    except Exception as error:  # noqa: BLE001 - reported, not raised
        return Outcome(
            case["id"],
            case["tool"],
            False,
            time.time() - started,
            0,
            why=f"{type(error).__name__}: {error}",
        )
    misses = check(case["expect"], result)
    return Outcome(
        case["id"],
        case["tool"],
        not misses,
        time.time() - started,
        int(result.get("local_tokens", 0)),
        why=_text(result)[:70].replace("\n", " "),
        misses=misses,
    )


def report(role: str, outcomes: list[Outcome]) -> None:
    print(f"\n=== shuttle-{role} ===")
    print(f"{'case':<30} {'tool':<10} {'ok':<4} {'s':>5} {'tokens':>7}")
    for out in outcomes:
        mark = "pass" if out.passed else "FAIL"
        print(
            f"{out.case:<30} {out.tool:<10} {mark:<4} "
            f"{out.seconds:>5.1f} {out.tokens:>7}"
        )
        if not out.passed:
            print(f"{'':<30} -> {'; '.join(out.misses) or out.why}")
    by_tool: dict[str, list[Outcome]] = {}
    for out in outcomes:
        by_tool.setdefault(out.tool, []).append(out)
    print(f"\n{'tool':<12} {'passed':>10} {'tokens':>9} {'seconds':>9}")
    for tool, group in sorted(by_tool.items()):
        good = sum(o.passed for o in group)
        print(
            f"{tool:<12} {good:>4}/{len(group):<5} "
            f"{sum(o.tokens for o in group):>9} "
            f"{sum(o.seconds for o in group):>9.0f}"
        )
    good = sum(o.passed for o in outcomes)
    print(
        f"{'total':<12} {good:>4}/{len(outcomes):<5} "
        f"{sum(o.tokens for o in outcomes):>9} "
        f"{sum(o.seconds for o in outcomes):>9.0f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="evals.run")
    parser.add_argument(
        "--server", default="long", choices=("long", "fast", "both")
    )
    parser.add_argument("--only", default="", help="substring of a case id")
    args = parser.parse_args()
    cases = [
        case
        for case in json.loads(CASES.read_text())
        if args.only in case["id"]
    ]
    if not cases:
        print(f"no case matches {args.only!r}", file=sys.stderr)
        return 2
    roles = ("long", "fast") if args.server == "both" else (args.server,)
    failed = 0
    for role in roles:
        outcomes = [run_case(case, role) for case in cases]
        report(role, outcomes)
        failed += sum(not out.passed for out in outcomes)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
