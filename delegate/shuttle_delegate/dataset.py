# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mateusz Okulanis
"""What Claude thought of a local answer, and the set that makes.

M7 trains an adapter on triples of input, local output and Claude's
grade and correction. The first two were already written down — the
audit log records every call and the run file holds the whole result —
and the third did not exist. Nothing asked Claude what it made of an
answer, so there was nothing to learn from.

`grade_run` is that missing half. Claude delegates, judges the answer
as it would anyway, and says so; the verdict lands beside the run it
is about. A correction is worth more than a verdict and is optional
because an answer that was right has none.

The set is exported from the three together and is a file, not a
service: training happens elsewhere, on a machine with room for it,
and the thing that crosses over is one JSONL.

    shuttle-dataset [FILE]
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import audit, indexing
from .indexing import IndexingError
from .runs import home

# The verdicts, narrow on purpose. A five-point scale invites a
# judgement nobody can reproduce; these three are the decisions a
# caller actually makes about a delegated answer.
VERDICTS = ("good", "wrong", "refused")
# A run id, as runs.new_id writes it. Checked rather than trusted: it
# becomes a path, and a grade for ../../etc is not a grade.
RUN_ID = re.compile(r"^\d{8}-\d{6}-[a-z_]+-[0-9a-f]{6}$")
# The run file's sections, as runs.write lays them out.
REQUEST = re.compile(r"## Request\n\n```json\n(.*?)\n```", re.S)
RESULT = re.compile(r"## Result\n\n```json\n(.*?)\n```", re.S)
# Fields that say what the delegation cost rather than what it
# answered. They are the wrong thing to teach a model to produce.
COST = frozenset(
    {
        "local_calls",
        "local_tokens",
        "truncated",
        "run",
        "report_cut",
        "attempts",
        "server",
        "parts",
        "parts_answering",
        "cached",
        "from_cache",
        "cached_age_seconds",
        "distance",
    }
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS grades (
    run        TEXT PRIMARY KEY,
    tool       TEXT NOT NULL,
    verdict    TEXT NOT NULL,
    correction TEXT NOT NULL,
    note       TEXT NOT NULL,
    graded_at  REAL NOT NULL
);
"""


class GradeError(RuntimeError):
    """The grade cannot be recorded, and why."""


@dataclass(frozen=True)
class Case:
    """One graded call, as a training example would see it."""

    run: str
    tool: str
    verdict: str
    request: dict[str, Any]
    answer: dict[str, Any]
    correction: str

    def as_json(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "tool": self.tool,
            "verdict": self.verdict,
            "input": self.request,
            "output": self.answer,
            "correction": self.correction,
        }


def prepare(db: sqlite3.Connection) -> None:
    db.executescript(SCHEMA)
    db.commit()


def connect(path: Path | None = None) -> sqlite3.Connection:
    db = indexing.connect(path)
    prepare(db)
    return db


def run_path(run: str) -> Path:
    """Where that run was written, having checked it is a run id."""
    name = Path(run).name.removesuffix(".md")
    if not RUN_ID.match(name):
        raise GradeError(
            f"{run!r} is not a run id; the reply of every tool carries "
            "one under 'run'"
        )
    return home() / "runs" / f"{name}.md"


def grade(
    db: sqlite3.Connection,
    run: str,
    verdict: str,
    correction: str = "",
    note: str = "",
) -> dict[str, Any]:
    """Record what the caller made of one delegated answer."""
    if verdict not in VERDICTS:
        raise GradeError(
            f"{verdict!r} is not a verdict; it is one of "
            + ", ".join(VERDICTS)
        )
    path = run_path(run)
    if not path.is_file():
        raise GradeError(f"{path.name}: no such run under {path.parent}")
    name = path.stem
    tool = name.split("-")[2] if len(name.split("-")) > 2 else ""
    with db:
        db.execute(
            "INSERT OR REPLACE INTO grades (run, tool, verdict, correction,"
            " note, graded_at) VALUES (?, ?, ?, ?, ?, ?)",
            (name, tool, verdict, correction, note, time.time()),
        )
    audit.log({"tool": "grade_run", "args": {"run": name, "verdict": verdict}})
    return {
        "run": name,
        "verdict": verdict,
        "graded": True,
        "corrected": bool(correction),
        "cases": counted(db),
    }


def counted(db: sqlite3.Connection) -> dict[str, int]:
    """How many of each verdict the set holds."""
    return {
        str(row["verdict"]): int(row["n"])
        for row in db.execute(
            "SELECT verdict, COUNT(*) AS n FROM grades GROUP BY verdict"
        )
    }


def _sections(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    asked = REQUEST.search(text)
    answered = RESULT.search(text)
    if not asked or not answered:
        raise GradeError("the run file has no request and result to read")
    try:
        one = json.loads(asked.group(1))
        two = json.loads(answered.group(1))
    except json.JSONDecodeError as error:
        raise GradeError(f"the run file does not parse: {error}") from error
    if not isinstance(one, dict) or not isinstance(two, dict):
        raise GradeError("the run file's sections are not objects")
    return one, two


def cases(db: sqlite3.Connection) -> list[Case]:
    """Every graded run whose file is still there.

    A run whose file has been cleared away is skipped rather than
    refused: the grade is still a record of what happened, and an
    export that fails because one old run was deleted is an export
    nobody runs.
    """
    out: list[Case] = []
    for row in db.execute(
        "SELECT run, tool, verdict, correction FROM grades ORDER BY run"
    ):
        path = home() / "runs" / f"{row['run']}.md"
        if not path.is_file():
            continue
        try:
            asked, answered = _sections(path.read_text(encoding="utf-8"))
        except (GradeError, OSError):
            continue
        out.append(
            Case(
                run=str(row["run"]),
                tool=str(row["tool"]),
                verdict=str(row["verdict"]),
                request=asked,
                answer={
                    key: value
                    for key, value in answered.items()
                    if key not in COST
                },
                correction=str(row["correction"]),
            )
        )
    return out


def export(into: Path, db: sqlite3.Connection | None = None) -> dict[str, Any]:
    """Write the set as JSONL and say what went into it."""
    own = db is None
    db = db or connect()
    try:
        found = cases(db)
        held = counted(db)
        into.parent.mkdir(parents=True, exist_ok=True)
        with into.open("w", encoding="utf-8") as sink:
            for case in found:
                sink.write(
                    json.dumps(case.as_json(), ensure_ascii=False) + "\n"
                )
        return {
            "file": str(into),
            "cases": len(found),
            "graded": sum(held.values()),
            "by_verdict": held,
            "corrections": sum(1 for one in found if one.correction),
        }
    finally:
        if own:
            db.close()


def main() -> int:
    """Export the set to the file named, or to the default path."""
    into = Path(sys.argv[1]) if len(sys.argv) > 1 else home() / "dataset.jsonl"
    try:
        answer = export(into)
    except (GradeError, IndexingError, OSError) as error:
        print(f"dataset: {error}", file=sys.stderr)
        return 1
    for key, value in answer.items():
        print(f"{key:<13} {value}")
    if not answer["cases"]:
        print(
            "\nnothing graded yet; the set fills as grade_run is called "
            "on delegated answers",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
