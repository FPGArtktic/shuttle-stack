# SPDX-License-Identifier: GPL-3.0-only
"""Trying one server, then another, and then giving up out loud.

A verifier that can say an answer failed can also say when to try
somewhere better. The ladder is a list of profiles: each is asked in
turn until one of them verifies its own answer, and if none does the
last answer comes back with `escalate` set and the whole ladder
recorded, so the caller knows what was tried and can do the work
itself rather than act on something that did not check out.

Which order pays is a question about the machine, not about the
design. On the reference machine the small server is the slower one,
so a ladder starting with it costs time rather than saving it; the
caller names the order and the README gives the numbers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .profiles import get


class CascadeError(RuntimeError):
    """The ladder cannot be climbed as written."""


def ladder(spec: str) -> list[str]:
    """The profiles named in a cascade, checked before anything runs."""
    names = [name.strip() for name in spec.split(",") if name.strip()]
    if not names:
        raise CascadeError(
            "a cascade is a comma-separated list of profiles, as in "
            "'fast,extract'"
        )
    for name in names:
        get(name)
    return names


def _step(profile: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile": profile,
        "verified": bool(result.get("verified")),
        "attempts": result.get("attempts"),
        "local_tokens": result.get("local_tokens"),
    }


def climb(
    names: Iterable[str], run: Callable[[str], dict[str, Any]]
) -> dict[str, Any]:
    """Ask each profile until one verifies its answer.

    The tokens reported are the whole climb rather than its last step,
    because that is what the machine spent.
    """
    history: list[dict[str, Any]] = []
    result: dict[str, Any] = {}
    for profile in names:
        result = run(profile)
        history.append(_step(profile, result))
        if result.get("verified"):
            break
    spent = sum(step["local_tokens"] or 0 for step in history)
    # The profile that answered is named, so whatever caps and records
    # the result afterwards reaches the server that did the work.
    climbed = result | {
        "ladder": history,
        "local_tokens": spent,
        "profile": history[-1]["profile"] if history else "",
    }
    if not result.get("verified"):
        climbed["escalate"] = True
        climbed["note"] = (
            "no profile verified its own answer; what came back is the "
            "last attempt and the question is better done by you"
        )
    return climbed
