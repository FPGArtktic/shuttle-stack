# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mateusz Okulanis
"""Where the whole output goes, and how little of it comes back.

The project's overriding rule: the caller sees a report under a hard
token limit, and everything the local model produced goes to a file it
can read if the report is not enough. A tool that returned its full
answer would spend the context the delegation exists to protect.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPORT_TOKENS = 1000
LONG_KEYS = ("answer", "summary")
CUT = "\n\n[cut: the whole output is in the run file]"


def home() -> Path:
    """The directory holding runs and the audit log.

    Relative to the working directory, because a run belongs to the
    project being worked on rather than to the delegate.
    """
    value = os.environ.get("SHUTTLE_HOME")
    return Path(value) if value else Path.cwd() / ".shuttle"


@dataclass(frozen=True)
class Run:
    """One call, as it was written down."""

    id: str
    path: Path

    def reference(self) -> str:
        try:
            return str(self.path.relative_to(Path.cwd()))
        except ValueError:
            return str(self.path)


def new_id(tool: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{tool}-{secrets.token_hex(3)}"


def write(tool: str, request: dict[str, Any], result: dict[str, Any]) -> Run:
    """Keep the request and the whole result, in that order."""
    run_id = new_id(tool)
    run = Run(run_id, home() / "runs" / f"{run_id}.md")
    run.path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        f"# {tool}",
        "",
        "## Request",
        "",
        "```json",
        json.dumps(request, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    for key in LONG_KEYS:
        if result.get(key):
            body += [f"## {key.capitalize()}", "", str(result[key]), ""]
    body += [
        "## Result",
        "",
        "```json",
        json.dumps(result, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    run.path.write_text("\n".join(body), encoding="utf-8")
    return run


def cap(
    count: Callable[[str], int], text: str, limit: int = REPORT_TOKENS
) -> tuple[str, bool]:
    """Cut the text to the limit, measured by the server's tokeniser.

    The cut is proportional, which lands under the limit in one or two
    counts for ordinary prose, and it repeats until the measurement
    agrees. Two rounds were not enough: a text denser at the front than
    at the back keeps its dense part when truncated, so the
    proportional estimate under-trims every time and the report came
    back over the limit the project calls its overriding rule. Each
    round removes at least a quarter of what is left, so the loop ends.
    """
    if limit < 1:
        raise ValueError(f"limit must be positive, got {limit}")
    used = count(text)
    if used <= limit:
        return text, False
    while used > limit and len(text) > 1:
        proportional = len(text) * limit // max(used, 1) * 85 // 100
        text = text[: max(1, min(proportional, len(text) * 3 // 4))]
        used = count(text)
    return text.rstrip() + CUT, True


def report(
    count: Callable[[str], int],
    tool: str,
    request: dict[str, Any],
    result: dict[str, Any],
    limit: int = REPORT_TOKENS,
) -> dict[str, Any]:
    """The capped result, with the run file that holds the rest."""
    run = write(tool, request, result)
    capped = dict(result)
    truncated = False
    for key in LONG_KEYS:
        if capped.get(key):
            capped[key], cut = cap(count, str(capped[key]), limit)
            truncated = truncated or cut
    return capped | {"run": run.reference(), "report_cut": truncated}
