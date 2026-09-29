# SPDX-License-Identifier: GPL-3.0-only
"""Finding the part of a file worth asking about.

Locating a passage is work for a regular expression rather than for a
model: it is exact, it costs no tokens, and it cannot report a line
that is not there. Measured against the running stack, the same
question over a 35 KB script answered wrongly with a fabricated quote,
and answered correctly from the matching function alone for a
thirtieth of the tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CONTEXT_LINES = 6
MAX_REGIONS = 40
MARKER = "--- line {line}"


class PatternError(ValueError):
    """The pattern is not a usable regular expression."""


@dataclass(frozen=True)
class Region:
    """A run of lines around one or more matches."""

    line: int
    text: str


def compile_pattern(pattern: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as error:
        raise PatternError(f"{pattern!r} is not a regexp: {error}") from error


def regions(
    text: str, pattern: str, context: int = CONTEXT_LINES
) -> list[Region]:
    """Every matching line with its neighbours, overlaps merged."""
    if context < 0:
        raise ValueError(f"context must not be negative, got {context}")
    matcher = compile_pattern(pattern)
    lines = text.splitlines()
    spans: list[list[int]] = []
    for number, line in enumerate(lines):
        if not matcher.search(line):
            continue
        low = max(0, number - context)
        high = min(len(lines), number + context + 1)
        if spans and low <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], high)
        else:
            spans.append([low, high])
    return [Region(low + 1, "\n".join(lines[low:high])) for low, high in spans]


def narrow(
    text: str, pattern: str, context: int = CONTEXT_LINES
) -> tuple[str, int]:
    """The matching regions as one text, and how many there were.

    Each region keeps the line number it starts at, so an answer drawn
    from it can be checked against the file it came from.
    """
    found = regions(text, pattern, context)
    kept = found[:MAX_REGIONS]
    joined = "\n\n".join(
        f"{MARKER.format(line=region.line)}\n{region.text}" for region in kept
    )
    return joined, len(found)
