# SPDX-License-Identifier: GPL-3.0-only
"""Checking that a quotation is really in the source.

A model that cites a line it invented is worse than one that refuses,
because the citation is the thing that makes an answer checkable and a
fabricated one reads exactly like a real one. Measured here, asked
about one function of the installer, the 8B server answered wrongly
and supported it with a sentence that appears nowhere in the file.

The check is cheap, exact and local: every quoted span in the answer
has to occur in the text the model was shown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Short spans are words rather than citations, and matching them proves
# nothing either way.
MIN_QUOTE = 12
# Only double quotes mark a citation. Backticks were tried and pair
# ambiguously: in `shellcheck` ... `checkpatch.pl` the regexp joins the
# closing backtick of one identifier to the opening one of the next and
# reports the prose between them as an invented quotation. Markup is
# dropped in flatten instead, so a backticked identifier inside a
# quoted span still matches.
QUOTED = re.compile(rf'"([^"\n]{{{MIN_QUOTE},}})"')
# A model ends a quotation where its own sentence ends, which moves the
# final comma to a full stop. That is not a different claim.
EDGE = ".,;:!? "
SPACE = re.compile(r"\s+")
# A backslash before a newline is how a line wraps, not content, and
# a model quoting across one writes the two halves as a single line.
CONTINUED = re.compile(r"\\\s*\n\s*")
# Backticks and emphasis are how Markdown writes a thing down, not part
# of the thing. A model quoting `subsystem: description` writes it
# without them and is right to; two of three quotations in a session
# over CONTRIBUTING.md were reported missing for this alone.
MARKUP = re.compile(r"[`*]+")


def flatten(text: str) -> str:
    """Wrapping, markup and case dropped, for comparison only."""
    plain = MARKUP.sub("", CONTINUED.sub(" ", text))
    return SPACE.sub(" ", plain).strip().lower()


def quotes(answer: str) -> list[str]:
    """The spans the answer presents as quotations."""
    found = []
    for match in QUOTED.finditer(answer):
        span = match.group(1).strip().strip(EDGE)
        if span and span not in found:
            found.append(span)
    return found


@dataclass
class Grounding:
    """How many of an answer's quotations are in its source."""

    total: int = 0
    missing: list[str] = field(default_factory=list)

    @property
    def grounded(self) -> int:
        return self.total - len(self.missing)

    @property
    def ok(self) -> bool:
        """An answer with no quotations is not contradicted by one."""
        return not self.missing

    def report(self) -> dict:
        entry: dict = {"quotes": self.total, "quotes_grounded": self.grounded}
        if self.missing:
            entry["quotes_not_in_source"] = self.missing
        return entry


def check(answer: str, source: str) -> Grounding:
    """Every quotation in the answer, against the text it was given."""
    flat = flatten(source)
    found = quotes(answer)
    return Grounding(
        total=len(found),
        missing=[span for span in found if flatten(span) not in flat],
    )
