# SPDX-License-Identifier: GPL-3.0-only
"""The golden set with an adapter and without it, on one server.

M7 asks whether a LoRA beats the base by a measurable margin, which
is a number and not an impression.  This runs the twenty cases twice
against two endpoints and prints the difference, scored with the same
`check` the ordinary evaluation uses, so neither side is judged by a
rule invented for it.

Three ways the comparison could lie, refused rather than noted.

A server given `--lora` does not say so, and a path it could not read
is not an error it reports.  Both endpoints are asked what adapters
they hold: the base must hold none and the other must hold one, or
this stops.  Without that check, measuring the base twice looks
exactly like an adapter that changes nothing.

Two servers with different parameters make the difference a
difference of servers.  Measured here: the same base scores 19 of 20
at a context of 16384 and 18 at 8192, because a 24000-token document
is split into more parts and the vote over them comes out otherwise.
Both ends are asked for their context and their model, and a mismatch
stops it.

And a margin of one case in twenty is five points, which at a small
training set is variance and not a result.  The number is printed as
cases and the reader is told what the set can resolve.

    python -m evals.ab http://127.0.0.1:8099 http://127.0.0.1:8098
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from unittest import mock

from shuttle_delegate.backend import Backend
from shuttle_delegate.config import Endpoint

from . import run as runner

# What one case out of twenty is worth, so the reader is not left to
# divide. Below this a difference is the set's resolution rather than
# a finding about the adapter.
ONE_CASE = 100.0 / 20


@dataclass(frozen=True)
class Side:
    """One endpoint, and what it says about itself."""

    url: str
    label: str
    model: str
    context: int
    adapters: list[str]


def _get(url: str, path: str) -> Any:
    try:
        with urllib.request.urlopen(  # noqa: S310 - a loopback URL
            f"{url.rstrip('/')}{path}", timeout=30
        ) as answer:
            return json.loads(answer.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"ab: {url}{path}: {error}") from error


def asked(url: str, label: str) -> Side:
    """What the server is serving, and which adapters it holds."""
    props = _get(url, "/props")
    held = _get(url, "/lora-adapters")
    names = [str(one.get("path", "")) for one in held if isinstance(one, dict)]
    return Side(
        url=url,
        label=label,
        model=str(props.get("model_path", "")),
        context=int(
            props.get("default_generation_settings", {}).get("n_ctx", 0)
        ),
        adapters=names,
    )


def comparable(base: Side, lora: Side) -> None:
    """Stop unless the two differ in the adapter and nothing else."""
    if base.adapters:
        raise SystemExit(
            f"ab: {base.url} already holds {base.adapters}, so it is not "
            "a base"
        )
    if not lora.adapters:
        raise SystemExit(
            f"ab: {lora.url} holds no adapter. `--lora` reports nothing "
            "when it loads and nothing when it does not, so this is the "
            "only way to tell, and measuring the base twice looks just "
            "like an adapter that changes nothing"
        )
    if base.model != lora.model:
        raise SystemExit(
            f"ab: {base.model} against {lora.model} is a comparison of models"
        )
    if base.context != lora.context:
        raise SystemExit(
            f"ab: context {base.context} against {lora.context} is a "
            "comparison of contexts; the same base scores 19 of 20 at "
            "16384 and 18 at 8192, because a long document is split "
            "into more parts and the vote comes out otherwise"
        )


def scored(side: Side, cases: list[dict[str, Any]]) -> list[Any]:
    """The golden set against one endpoint, scored by evals.run."""
    server = Backend(Endpoint(role="ab", url=side.url))
    with mock.patch.object(runner, "backend", lambda _role: server):
        return [runner.run_case(one, "ab") for one in cases]


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m evals.ab BASE_URL LORA_URL", file=sys.stderr)
        return 2
    base = asked(sys.argv[1], "base")
    lora = asked(sys.argv[2], "lora")
    comparable(base, lora)
    print(f"model    {base.model}")
    print(f"context  {base.context}")
    print(f"adapter  {lora.adapters[0]}")
    cases = json.loads(runner.CASES.read_text())
    out = {}
    for side in (base, lora):
        held = scored(side, cases)
        runner.report(side.label, held)
        out[side.label] = held
    counts = {
        name: sum(1 for one in held if one.passed)
        for name, held in out.items()
    }
    margin = counts["lora"] - counts["base"]
    print(
        f"\nbase {counts['base']} of {len(cases)}, "
        f"lora {counts['lora']} of {len(cases)}, "
        f"margin {margin:+d} cases"
    )
    if margin == 0:
        print("no margin: the adapter changed no outcome")
    elif abs(margin) == 1:
        print(
            f"one case is {ONE_CASE:.0f} points, which this set cannot "
            "tell from variance; run it again before believing it"
        )
    for name, held in out.items():
        failed = [one.case for one in held if not one.passed]
        if failed:
            print(f"{name} failed: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
