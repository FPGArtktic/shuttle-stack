# SPDX-License-Identifier: GPL-3.0-only
"""One line per call, so that what was sent where is answerable.

The log is the record the project asks for before any isolation work:
a delegation that cannot be audited cannot be trusted with anything
that matters.
"""

from __future__ import annotations

import json
import time

from .runs import home

NAME = "audit.jsonl"


def log(entry: dict) -> None:
    """Append one record. A failure to log is not a failure to answer."""
    path = home() / NAME
    line = json.dumps(
        {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z")} | entry,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as sink:
            sink.write(line + "\n")
    except OSError:
        pass
