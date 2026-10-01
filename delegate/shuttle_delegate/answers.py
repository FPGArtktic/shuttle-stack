# SPDX-License-Identifier: GPL-3.0-only
"""A question asked again, answered without a model.

`ask_file` is the expensive tool: the server reads the file, answers,
and is drawn again when the answer does not check out, which on the
reference machine is tens of seconds to minutes. The same question
asked twice is the common case in a conversation, and asking it twice
of the model is waste.

The key is the question's embedding, not its wording, so a repeat
reworded or repunctuated still finds its answer. It is deliberately
near-exact, and the measurement in MAX_DISTANCE says why: the design
asks for a paraphrase to be served too, and on this evidence that
cannot be done safely.

What makes it safe is what it refuses. An entry belongs to one file
at one size and modification time, so editing the file throws its
answers away. An answer that failed its own grounding check is never
kept. A neighbour further than MAX_DISTANCE is not served. And every
hit says what it was originally asked and how near that was, so the
caller can see what it is being given.

Set SHUTTLE_ANSWER_TTL=0 to switch it off.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import indexing
from .indexing import IndexingError

# How near a stored question has to be, as a cosine distance on
# bge-m3. The design asks for a paraphrase to be served from here; it
# cannot be, and the number is what is left after measuring that.
#
# Over 22 questions written from one datasheet and 22 rewrites of them
# by shuttle-long, the rewrites sat between 0.018 and 0.173 and every
# other pairing of the 462 began at 0.087. That reads like a clean
# threshold anywhere near 0.2, and it is not, because a model asked to
# reword a question keeps most of its vocabulary. Six pairs written by
# hand to mean the same thing in different words say otherwise:
#
#   0.705  "which leg is ground" / "what pin is VSS on"
#   0.623  "how do I keep the chip from cooking in a TSSOP"
#            / "what is the thermal impedance of the PW package"
#   0.402  "how hot can it get before it breaks"
#            / "what is the maximum junction temperature"
#
# and against them, two questions with opposite answers:
#
#   0.213  "how much current can one output sink"
#            / "how much current can one output source"
#
# The same question reaches 0.705 and a question whose answer differs
# by a sign is 0.213 away. There is no threshold between them, so this
# one is set below the dangerous pair rather than above the useful
# ones: a reworded repeat is served, a paraphrase in a different
# vocabulary is not, and a datasheet is never answered about source
# when it was asked about sink.
MAX_DISTANCE = 0.10
# A cached answer older than this is thrown away. The fingerprint
# catches a changed file; this catches a model, a profile or a prompt
# that changed underneath the entry. Zero switches the cache off.
ANSWER_TTL = float(os.environ.get("SHUTTLE_ANSWER_TTL", 86400.0))

SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    source   TEXT NOT NULL,
    mark     TEXT NOT NULL,
    question TEXT NOT NULL,
    reply    TEXT NOT NULL,
    made_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS answers_source ON answers (source, mark);
"""

VECTORS = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS vec_answers USING vec0("
    f"id INTEGER PRIMARY KEY, embedding float[{indexing.DIMENSIONS}])"
)

# Spent on this call. A served answer cost one embedding and no
# generation, and reporting what the original cost would read as work
# this call did.
FREE = {"local_calls": 0, "local_tokens": 0}


@dataclass(frozen=True)
class Kept:
    """One answer as it was stored."""

    question: str
    reply: dict[str, Any]
    distance: float
    age: float

    def report(self) -> dict[str, Any]:
        """The stored answer, marked as one, with what matched it."""
        return (
            self.reply
            | FREE
            | {
                "from_cache": True,
                "asked_as": self.question,
                "distance": round(self.distance, 3),
                "cached_age_seconds": round(self.age, 1),
            }
        )


def prepare(db: sqlite3.Connection) -> None:
    db.executescript(SCHEMA)
    db.execute(VECTORS)
    db.commit()


def connect(path: Path | None = None) -> sqlite3.Connection:
    db = indexing.connect(path)
    prepare(db)
    return db


def fingerprint(path: str) -> str:
    """What the file was when it was read.

    Size and modification time rather than a hash: a datasheet is
    tens of megabytes and hashing it on every question would cost
    more than the answer. A file edited back to the same size within
    the same second would be missed, which for a document is not a
    case worth paying for.
    """
    try:
        stat = Path(path).expanduser().stat()
    except OSError as error:
        raise IndexingError(f"{path}: {error}") from error
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _candidates(
    db: sqlite3.Connection, source: str, mark: str
) -> list[tuple[int, str, str, float, list[float]]]:
    return [
        (
            int(row["id"]),
            str(row["question"]),
            str(row["reply"]),
            float(row["made_at"]),
            indexing._floats(row["embedding"]),
        )
        for row in db.execute(
            "SELECT a.id, a.question, a.reply, a.made_at, v.embedding"
            " FROM answers a JOIN vec_answers v ON v.id = a.id"
            " WHERE a.source = ? AND a.mark = ?",
            (source, mark),
        )
    ]


def _drop(db: sqlite3.Connection, query: str, args: tuple[Any, ...]) -> int:
    """Remove the rows the query names, and their vectors with them.

    A vec0 table takes no foreign key and no trigger, so deleting an
    answer and forgetting its vector leaves an orphan that nothing
    will ever read and nothing will ever remove. Answering the same
    question twice left one each time.
    """
    ids = [int(row[0]) for row in db.execute(query, args)]
    if not ids:
        return 0
    places = ",".join("?" * len(ids))
    db.execute(f"DELETE FROM vec_answers WHERE id IN ({places})", ids)
    db.execute(f"DELETE FROM answers WHERE id IN ({places})", ids)
    return len(ids)


def recall(
    db: sqlite3.Connection,
    source: str,
    question: str,
    distance: float = MAX_DISTANCE,
    ttl: float = ANSWER_TTL,
) -> Kept | None:
    """The nearest answer kept for this file, if it is near enough.

    The candidates are only this file's at this fingerprint, so the
    comparison is exact arithmetic over a few dozen vectors rather
    than a nearest-neighbour search over everything in the index.
    Stale entries are dropped where they are found: a cache that
    keeps answers to a file nobody has looked at since March grows
    without bound.
    """
    mark = fingerprint(source)
    _drop(
        db,
        "SELECT id FROM answers WHERE source = ? AND mark != ?",
        (source, mark),
    )
    held = _candidates(db, source, mark)
    if not held:
        db.commit()
        return None
    mine = indexing.embed([question])[0]
    now = time.time()
    best: Kept | None = None
    for row_id, asked, reply, made_at, vector in held:
        if now - made_at > ttl:
            _drop(db, "SELECT id FROM answers WHERE id = ?", (row_id,))
            continue
        apart = 1.0 - indexing._cosine(mine, vector)
        if apart > distance or (best is not None and apart >= best.distance):
            continue
        best = Kept(asked, json.loads(reply), apart, now - made_at)
    db.commit()
    return best


def remember(
    db: sqlite3.Connection,
    source: str,
    question: str,
    reply: dict[str, Any],
) -> None:
    """Keep an answer, if it is one worth keeping.

    An unverified answer is not kept. The whole argument for serving
    a stored answer is that it was checked once; storing one that
    failed its grounding would spread a bad answer to every
    paraphrase of the question that produced it.
    """
    if not reply.get("verified", False):
        return
    mark = fingerprint(source)
    vector = indexing._blob(indexing.embed([question])[0])
    with db:
        _drop(
            db,
            "SELECT id FROM answers WHERE source = ? AND question = ?",
            (source, question),
        )
        row = db.execute(
            "INSERT INTO answers (source, mark, question, reply, made_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                source,
                mark,
                question,
                json.dumps(reply, ensure_ascii=False),
                time.time(),
            ),
        )
        db.execute(
            "INSERT INTO vec_answers (id, embedding) VALUES (?, ?)",
            (row.lastrowid, vector),
        )


def held(path: str, question: str) -> dict[str, Any] | None:
    """A kept answer to this question about this file, ready to return.

    Nothing here is allowed to turn a cache miss into a failed call.
    The index may be absent, the embedding model may be down, the file
    may have been replaced between the two statements: in every case
    the answer is that there is no cached answer, and the question
    goes to the server as it would have anyway.
    """
    if ANSWER_TTL <= 0:
        return None
    try:
        with closing(connect()) as db:
            found = recall(db, path, question)
    except (IndexingError, sqlite3.Error):
        return None
    return found.report() if found is not None else None


def keep(path: str, question: str, reply: dict[str, Any]) -> None:
    """Store an answer, or quietly do not.

    Same rule in the other direction: an answer that was produced is
    not thrown away because the cache could not be written, and the
    caller is not told about a cache it did not ask for.
    """
    if ANSWER_TTL <= 0:
        return
    try:
        with closing(connect()) as db:
            remember(db, path, question, reply)
    except (IndexingError, sqlite3.Error):
        return


def kept(db: sqlite3.Connection | None = None) -> dict[str, Any]:
    """What the cache holds, per file."""
    own = db is None
    db = db or connect()
    try:
        rows = db.execute(
            "SELECT source, COUNT(*) AS held, MAX(made_at) AS newest"
            " FROM answers GROUP BY source ORDER BY source"
        ).fetchall()
        return {
            "answers": [
                {
                    "file": str(row["source"]),
                    "questions": int(row["held"]),
                    "newest": time.strftime(
                        "%Y-%m-%d %H:%M", time.localtime(row["newest"])
                    ),
                }
                for row in rows
            ]
        }
    finally:
        if own:
            db.close()
