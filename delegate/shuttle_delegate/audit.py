# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mateusz Okulanis
"""One line per call, so that what was sent where is answerable.

The log is the record the project asks for before any isolation work:
a delegation that cannot be audited cannot be trusted with anything
that matters.

It is also the only file here that grows without an end. A line is a
few hundred bytes and an afternoon of jobs is thousands of them, so
the log is rotated at a size rather than left to fill the disk of the
machine it is auditing. The oldest generation is dropped: a record
that cannot be kept forever is better kept recently than not at all.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .runs import home

NAME = "audit.jsonl"
# Measured on this machine, a line of an ordinary call is 300 to 600
# bytes, so eight megabytes is on the order of twenty thousand calls.
MAX_BYTES = int(os.environ.get("SHUTTLE_AUDIT_BYTES", 8 * 1024 * 1024))
# Generations kept beside the live file: audit.1.jsonl is the one
# before this, audit.4.jsonl the oldest. Four of them at eight
# megabytes is forty in all, which is small beside one KV dump.
KEEP = int(os.environ.get("SHUTTLE_AUDIT_KEEP", 4))


def generation(path: Path, number: int) -> Path:
    """Where generation N of the log lives."""
    return path.with_name(f"{path.stem}.{number}{path.suffix}")


def rotate(path: Path, limit: int = MAX_BYTES, keep: int = KEEP) -> bool:
    """Move the log aside if it has grown past the limit.

    Checked before the write rather than after, so the live file never
    exceeds the limit by more than one line. Rotation that fails for
    any reason leaves the log where it is and the line is still
    written: an unrotated record is a disk problem, a lost record is
    an audit problem.
    """
    try:
        if limit <= 0 or path.stat().st_size < limit:
            return False
        if keep < 1:
            path.unlink()
            return True
        generation(path, keep).unlink(missing_ok=True)
        for number in range(keep - 1, 0, -1):
            older = generation(path, number)
            if older.exists():
                older.replace(generation(path, number + 1))
        path.replace(generation(path, 1))
    except OSError:
        return False
    return True


def log(entry: dict[str, Any]) -> None:
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
        # The limits are passed rather than left to the defaults: a
        # default is bound when the function is defined, so a caller
        # that changes MAX_BYTES afterwards would be ignored.
        rotate(path, MAX_BYTES, KEEP)
        with path.open("a", encoding="utf-8") as sink:
            sink.write(line + "\n")
    except OSError:
        pass


def held() -> list[Path]:
    """The log and every generation of it that is still on disk."""
    path = home() / NAME
    return [
        one
        for one in [path, *(generation(path, n) for n in range(1, KEEP + 1))]
        if one.exists()
    ]
