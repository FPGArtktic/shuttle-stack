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
# A boundary that never matches must not pull in the rest of the file.
MAX_REGION_LINES = 400
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


def _ends_at(lines: list[str], start: int, boundary: re.Pattern[str]) -> int:
    """The line after the region: the next boundary, or a cap."""
    cap = min(len(lines), start + MAX_REGION_LINES)
    for number in range(start + 1, cap):
        if boundary.search(lines[number]):
            return number
    return cap


def regions(
    text: str,
    pattern: str,
    context: int = CONTEXT_LINES,
    until: str = "",
) -> list[Region]:
    """Every matching line with its surroundings, overlaps merged.

    Without `until`, a region is the match and `context` lines either
    side of it. With it, the region runs from `context` lines before
    the match to the line before the next one matching `until`, which
    is how a whole definition or section is asked for without this
    code knowing what language it is written in.

    The difference is not cosmetic. A six-line definition followed by
    thirty lines of context hands the model its neighbours, and the
    answer comes back about them: measured on the installer, four of
    ten phases were described as the phase defined after them.
    """
    if context < 0:
        raise ValueError(f"context must not be negative, got {context}")
    matcher = compile_pattern(pattern)
    boundary = compile_pattern(until) if until else None
    lines = text.splitlines()
    spans: list[list[int]] = []
    for number, line in enumerate(lines):
        if not matcher.search(line):
            continue
        low = max(0, number - context)
        if boundary is None:
            high = min(len(lines), number + context + 1)
        else:
            high = _ends_at(lines, number, boundary)
        if spans and low <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], high)
        else:
            spans.append([low, high])
    # The cap is per match, and merging composed it away: each later
    # match re-anchored it at its own line, so a pattern matching more
    # often than once every MAX_REGION_LINES walked a single region to
    # the end of the file. It is applied again to the merged span.
    return [
        Region(
            low + 1, "\n".join(lines[low : min(high, low + MAX_REGION_LINES)])
        )
        for low, high in spans
    ]


def narrow(
    text: str,
    pattern: str,
    context: int = CONTEXT_LINES,
    until: str = "",
) -> tuple[str, int]:
    """The matching regions as one text, and how many there were.

    Each region keeps the line number it starts at, so an answer drawn
    from it can be checked against the file it came from.
    """
    found = regions(text, pattern, context, until)
    kept = found[:MAX_REGIONS]
    joined = "\n\n".join(
        f"{MARKER.format(line=region.line)}\n{region.text}" for region in kept
    )
    return joined, len(found)
