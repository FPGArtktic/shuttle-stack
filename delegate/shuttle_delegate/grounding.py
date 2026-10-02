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
from typing import Any

# Short spans are words rather than citations, and matching them proves
# nothing either way.
MIN_QUOTE = 12
# Quotation marks are paired by position rather than matched by a
# regular expression. A pattern with a length condition in it scans from
# the left, fails at the opening mark of a span too short to count, and
# then starts again at that span's CLOSING mark, so it reports the prose
# between two short quotations as a quotation itself: in
#     Pass "--no-ngram" to disable it, or "--no-draft" to free VRAM.
# it returned 'to disable it, or'. Splitting the line on the mark and
# taking the odd pieces cannot make that mistake, whatever the lengths.
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
    """Wrapping, markup, spacing and case dropped, for comparison only.

    The markup characters are removed from both the quotation and the
    source. In a shell script a backtick is syntax rather than markup,
    so in principle this could ground an invention that matches only
    once the backticks are gone. That is accepted: leaving them in was
    measured to report two of three genuine citations as missing, and
    a check distrusted that often is a check nobody reads.

    Space goes the same way. A datasheet column read by pdftotext
    says "1µA" and a model writing it down says "1 µA", which is the
    same reading of the same cell; asked for the ratings table on one
    page, every row failed on that alone and the answer came back
    twice and ungrounded. Removing space rather than collapsing it
    does not loosen what a quotation has to match: the words still
    have to be contiguous in the source, and now they may be
    contiguous across a line break, which in a PDF column they often
    are.
    """
    plain = MARKUP.sub("", CONTINUED.sub(" ", text))
    return SPACE.sub("", plain).lower()


def quotes(answer: str) -> list[str]:
    """The spans the answer presents as quotations.

    Marks are paired within a line, so an unclosed quotation ends at the
    end of its line rather than swallowing the paragraph after it.
    """
    found: list[str] = []
    for line in answer.splitlines():
        for span in line.split('"')[1::2]:
            cleaned = span.strip().strip(EDGE)
            if len(cleaned) >= MIN_QUOTE and cleaned not in found:
                found.append(cleaned)
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

    def report(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "quotes": self.total,
            "quotes_grounded": self.grounded,
        }
        if self.missing:
            entry["quotes_not_in_source"] = self.missing
        return entry


def present(span: str, flat: str) -> bool:
    """Whether the source holds this quotation, as a span or by lines.

    A span first, because contiguity is the stronger claim and prose
    is contiguous. A table is not: the meaningful quotation out of one
    is a column's name and its value, and between those two lines sits
    a rule of dashes nobody quotes. Measured on a timing report, every
    correct answer was refused for that alone --

        ; Fmax       ; Restricted Fmax ; Clock Name ; Note ;
        ; 154.23 MHz ; 154.23 MHz      ; clk        ;      ;

    is the table read right, and the source has `+---+---+` between
    the two.

    Line by line is still a real check: each line has to be a line the
    source holds, so nothing can be invented, only put side by side.
    What a reordering could do is pair a value with the wrong label,
    and that claim lives in the answer, which is prose and grounded
    nowhere.
    """
    if flatten(span) in flat:
        return True
    lines = [one for one in span.splitlines() if one.strip()]
    if len(lines) < 2:
        return False
    return all(flatten(one) in flat for one in lines)


def check(answer: str, source: str) -> Grounding:
    """Every quotation in the answer, against the text it was given."""
    flat = flatten(source)
    found = quotes(answer)
    return Grounding(
        total=len(found),
        missing=[span for span in found if not present(span, flat)],
    )


# A value of a word or two is as likely to be a shared word as a
# citation, and flagging it would say nothing either way.
MIN_FIELD = 6


def fields_in_source(fields: dict[str, Any], source: str) -> list[str]:
    """Extracted string values the source does not contain.

    A number or a boolean is the model's reading of the text rather
    than a span of it, so only strings long enough to be a quotation
    are checked. The answer is a list of what was not found, named by
    its field, which is what a caller needs to decide whether to
    believe the rest.
    """
    flat = flatten(source)
    missing: list[str] = []
    for key, value in fields.items():
        missing.extend(_absent(key, value, flat))
    return missing


def _absent(key: str, value: object, flat: str) -> list[str]:
    """Every string anywhere under this field that the source lacks.

    A schema may ask for a list or an object, and skipping those left
    extract reporting a verified answer it had not checked.
    """
    if isinstance(value, str):
        if len(value.strip()) < MIN_FIELD or present(value, flat):
            return []
        return [f"{key}={value!r}"]
    if isinstance(value, dict):
        return [
            item
            for name, inner in value.items()
            for item in _absent(f"{key}.{name}", inner, flat)
        ]
    if isinstance(value, list):
        return [
            item
            for index, inner in enumerate(value)
            for item in _absent(f"{key}[{index}]", inner, flat)
        ]
    return []
