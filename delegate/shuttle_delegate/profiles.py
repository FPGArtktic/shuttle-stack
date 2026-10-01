# SPDX-License-Identifier: GPL-3.0-only
"""Named settings for a delegation: where it goes and how it is read.

A profile names one way of asking: which server answers, how the
sampling is set, and how much of the answer comes back. The built-in
three cover what the tools need; a file alongside stack.env overrides
them or adds more, so changing how extraction is sampled is an edit in
one place rather than an argument at every call site.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .config import ROLES, config_home

NAME = "profiles.toml"

BUILT_IN: dict[str, dict[str, Any]] = {
    "long": {"server": "long", "temperature": 0.2},
    "fast": {"server": "fast", "temperature": 0.2},
    # Schema-bound work wants no sampling: there is one right shape and
    # temperature can only move the answer away from it.
    "extract": {"server": "long", "temperature": 0.0},
    # Asking for several different ideas is the one case where sampling
    # is the point rather than a hazard.
    "brainstorm": {"server": "long", "temperature": 0.8},
}


class ProfileError(RuntimeError):
    """The profiles file asks for something that cannot be honoured."""


@dataclass(frozen=True)
class Profile:
    """One named way of asking."""

    name: str
    server: str = "long"
    temperature: float = 0.2
    report_tokens: int = 1000

    def __post_init__(self) -> None:
        if self.server not in ROLES:
            raise ProfileError(
                f"profile {self.name!r} names server {self.server!r}; "
                f"use one of {ROLES}"
            )
        if not 0.0 <= self.temperature <= 2.0:
            raise ProfileError(
                f"profile {self.name!r} has temperature "
                f"{self.temperature}, outside 0.0 to 2.0"
            )
        if self.report_tokens < 1:
            raise ProfileError(
                f"profile {self.name!r} reports {self.report_tokens} "
                "tokens, which is not a report"
            )


def path() -> Path:
    return config_home() / "shuttle" / NAME


def _build(name: str, values: dict[str, Any]) -> Profile:
    known = {field.name for field in fields(Profile)} - {"name"}
    unknown = set(values) - known
    if unknown:
        raise ProfileError(
            f"profile {name!r} sets {sorted(unknown)}, which no profile "
            f"has; the keys are {sorted(known)}"
        )
    return Profile(name=name, **values)


def load(file: Path | None = None) -> dict[str, Profile]:
    """The built-in profiles, with the file's on top of them."""
    file = file or path()
    merged = {name: dict(values) for name, values in BUILT_IN.items()}
    if file.is_file():
        try:
            written = tomllib.loads(file.read_text())
        except tomllib.TOMLDecodeError as error:
            raise ProfileError(f"{file}: {error}") from error
        for name, values in written.items():
            if not isinstance(values, dict):
                raise ProfileError(
                    f"{file}: {name!r} is not a table; a profile is "
                    f"written as [{name}]"
                )
            merged.setdefault(name, {}).update(values)
    return {name: _build(name, values) for name, values in merged.items()}


def get(name: str, file: Path | None = None) -> Profile:
    known = load(file)
    if name not in known:
        raise ProfileError(f"no profile {name!r}; there is {sorted(known)}")
    return known[name]
