# SPDX-License-Identifier: GPL-3.0-only
"""Source indexed by the units the language defines.

A file is not a useful answer and a fixed window is worse: half a
function tells you it exists and not what it does. `shuttle-docs`
parses the file with Tree-sitter and returns the units — a module, an
entity, a `do_install`, a device tree node — each with the lines it
occupies, and a hit is then a place in a file rather than a file.

The tables live beside the document tables in the same index file.
They are separate because the citation is: a document hit names a
page, a source hit names a line range, and squeezing both through one
column would lose the difference. Everything else — the two channels,
the fusion, the thinning, the cache — is the same code and is
imported from `indexing`.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import indexing
from .documents import IMAGE, TIMEOUT, DocumentError, in_container
from .indexing import Count, Hit, IndexingError

# What the chunker can be asked for. A file outside this list is not
# refused, only cut on blank lines, because an answer from a README is
# better than no answer; the reply says which happened.
SUFFIXES = (
    ".bash",
    ".bb",
    ".bbappend",
    ".bbclass",
    ".c",
    ".cc",
    ".cmake",
    ".cpp",
    ".cxx",
    ".dts",
    ".dtsi",
    ".h",
    ".hh",
    ".hpp",
    ".inc",
    ".mk",
    ".py",
    ".sdc",
    ".sh",
    ".sv",
    ".svh",
    ".tcl",
    ".v",
    ".vh",
    ".vhd",
    ".vhdl",
    ".xdc",
)
NAMES = ("Makefile", "makefile", "GNUmakefile", "CMakeLists.txt")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    path       TEXT PRIMARY KEY,
    language   TEXT NOT NULL,
    how        TEXT NOT NULL,
    units      INTEGER NOT NULL,
    lines      INTEGER NOT NULL,
    indexed_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS units (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source     TEXT NOT NULL,
    ordinal    INTEGER NOT NULL,
    kind       TEXT NOT NULL,
    name       TEXT,
    first_line INTEGER NOT NULL,
    last_line  INTEGER NOT NULL,
    text       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS units_source ON units (source, ordinal);
CREATE INDEX IF NOT EXISTS units_name ON units (name);
"""

# The name and the kind are indexed beside the body, so that "who
# instantiates counter" and "do_install" are both lexical questions.
FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS units_fts USING fts5(
    text, name, kind, content='units', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS units_fts_insert
AFTER INSERT ON units BEGIN
    INSERT INTO units_fts(rowid, text, name, kind)
    VALUES (new.id, new.text, new.name, new.kind);
END;
CREATE TRIGGER IF NOT EXISTS units_fts_delete
AFTER DELETE ON units BEGIN
    INSERT INTO units_fts(units_fts, rowid, text, name, kind)
    VALUES ('delete', old.id, old.text, old.name, old.kind);
END;
CREATE TRIGGER IF NOT EXISTS units_fts_update
AFTER UPDATE ON units BEGIN
    INSERT INTO units_fts(units_fts, rowid, text, name, kind)
    VALUES ('delete', old.id, old.text, old.name, old.kind);
    INSERT INTO units_fts(rowid, text, name, kind)
    VALUES (new.id, new.text, new.name, new.kind);
END;
"""

VECTORS = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS vec_units USING vec0("
    f"id INTEGER PRIMARY KEY, embedding float[{indexing.DIMENSIONS}])"
)


@dataclass(frozen=True)
class Unit:
    """One citable piece of a file."""

    kind: str
    name: str
    first_line: int
    last_line: int
    text: str

    @property
    def lines(self) -> str:
        if self.first_line == self.last_line:
            return f"line {self.first_line}"
        return f"lines {self.first_line}-{self.last_line}"

    @property
    def label(self) -> str:
        return f"{self.kind} {self.name}".strip()


@dataclass(frozen=True)
class Cut:
    """What the chunker made of one file."""

    file: str
    language: str
    how: str
    why: str
    units: list[Unit]


def is_source(path: str | Path) -> bool:
    where = Path(path)
    return (
        where.suffix.lower() in SUFFIXES
        or where.name in NAMES
        or where.name.startswith(("Kconfig", "Makefile"))
    )


def prepare(db: sqlite3.Connection) -> None:
    """The code tables, made once beside the document tables."""
    db.executescript(SCHEMA)
    db.execute(VECTORS)
    db.executescript(FTS)
    db.commit()


def units(path: str) -> Cut:
    """Ask the container to cut the file into its units."""
    file = Path(path).expanduser().resolve()
    if not file.is_file():
        raise DocumentError(f"{file}: not a readable file")
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell
            in_container(file, "chunk"),
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise DocumentError(
            f"{file.name}: still parsing after {TIMEOUT:.0f}s"
        ) from error
    if done.returncode != 0:
        reason = (done.stderr or done.stdout).strip()[:300]
        raise DocumentError(
            f"{file.name}: {reason or f'{IMAGE} would not parse it'}"
        )
    try:
        answer = json.loads(done.stdout)
    except json.JSONDecodeError as error:
        raise DocumentError(
            f"{file.name}: the chunker did not answer with JSON: "
            f"{done.stdout[:200]}"
        ) from error
    return Cut(
        file=str(answer.get("file", file.name)),
        language=str(answer.get("language", "")),
        how=str(answer.get("how", "")),
        why=str(answer.get("why", "")),
        units=[
            Unit(
                kind=str(one["kind"]),
                name=str(one.get("name", "")),
                first_line=int(one["first_line"]),
                last_line=int(one["last_line"]),
                text=str(one["text"]),
            )
            for one in answer.get("units", [])
        ],
    )


def forget(db: sqlite3.Connection, file: str) -> int:
    """Drop everything a file contributed. The triggers see it."""
    ids = [
        int(row["id"])
        for row in db.execute("SELECT id FROM units WHERE source = ?", (file,))
    ]
    db.executemany("DELETE FROM vec_units WHERE id = ?", [(i,) for i in ids])
    db.execute("DELETE FROM units WHERE source = ?", (file,))
    db.execute("DELETE FROM sources WHERE path = ?", (file,))
    db.execute("DELETE FROM searches")
    return len(ids)


def store(db: sqlite3.Connection, file: str, cut: Cut, lines: int) -> int:
    """Embed the units and put them in, replacing an earlier pass.

    One transaction: a file half indexed would answer about the half
    that made it and say nothing about the rest.
    """
    if not cut.units:
        raise IndexingError(f"{file}: no units to index")
    # The name and the kind go into the embedded text. A question is
    # usually about a named thing, and the body of a `do_install` does
    # not contain the words "do_install" anywhere.
    vectors = indexing.embed(
        [f"{unit.label}\n{unit.text}" for unit in cut.units]
    )
    blobs = [indexing._blob(vector) for vector in vectors]
    with db:
        forget(db, file)
        for order, unit in enumerate(cut.units):
            row = db.execute(
                "INSERT INTO units"
                " (source, ordinal, kind, name, first_line, last_line,"
                " text) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    file,
                    order,
                    unit.kind,
                    unit.name,
                    unit.first_line,
                    unit.last_line,
                    unit.text,
                ),
            )
            db.execute(
                "INSERT INTO vec_units (id, embedding) VALUES (?, ?)",
                (row.lastrowid, blobs[order]),
            )
        db.execute(
            "INSERT INTO sources (path, language, how, units, lines,"
            " indexed_at) VALUES (?, ?, ?, ?, ?, ?)",
            (file, cut.language, cut.how, len(cut.units), lines, time.time()),
        )
    return len(cut.units)


def _lexical(db: sqlite3.Connection, question: str) -> list[int]:
    query = indexing.terms(question)
    if not query:
        return []
    try:
        rows = db.execute(
            "SELECT rowid FROM units_fts WHERE units_fts MATCH ?"
            " ORDER BY rank LIMIT ?",
            (query, indexing.CANDIDATES),
        ).fetchall()
    except sqlite3.OperationalError as error:
        raise IndexingError(
            f"the lexical index refused {query!r}: {error}"
        ) from error
    return [int(row["rowid"]) for row in rows]


def _semantic(db: sqlite3.Connection, vector: list[float]) -> list[int]:
    rows = db.execute(
        "SELECT id FROM vec_units WHERE embedding MATCH ?"
        " AND k = ? ORDER BY distance",
        (indexing._blob(vector), indexing.CANDIDATES),
    ).fetchall()
    return [int(row["id"]) for row in rows]


def _vectors(db: sqlite3.Connection, ids: list[int]) -> dict[int, list[float]]:
    if not ids:
        return {}
    places = ",".join("?" * len(ids))
    return {
        int(row["id"]): indexing._floats(row["embedding"])
        for row in db.execute(
            f"SELECT id, embedding FROM vec_units WHERE id IN ({places})",
            ids,
        )
    }


def _rows(
    db: sqlite3.Connection, chosen: list[tuple[int, float, str]]
) -> list[Hit]:
    if not chosen:
        return []
    places = ",".join("?" * len(chosen))
    found = {
        int(row["id"]): row
        for row in db.execute(
            "SELECT id, source, kind, name, first_line, last_line, text"
            f" FROM units WHERE id IN ({places})",
            [entry[0] for entry in chosen],
        )
    }
    hits = []
    for row_id, score, how in chosen:
        row = found.get(row_id)
        if row is None:
            continue
        unit = Unit(
            str(row["kind"]),
            str(row["name"] or ""),
            int(row["first_line"]),
            int(row["last_line"]),
            str(row["text"]),
        )
        hits.append(
            Hit(
                file=str(row["source"]),
                page=unit.first_line,
                clause=unit.lines,
                heading=unit.label,
                text=unit.text,
                score=score,
                how=how,
            )
        )
    return hits


def _look(
    db: sqlite3.Connection, question: str, limit: int
) -> tuple[list[Hit], dict[str, Any]]:
    ranked = {"lexical": _lexical(db, question)}
    note = ""
    try:
        ranked["semantic"] = _semantic(db, indexing.embed([question])[0])
    except IndexingError as error:
        note = f"the semantic half is missing: {error}"
    fused = indexing.fuse(ranked)
    vectors = _vectors(db, [entry[0] for entry in fused])
    hits = _rows(db, indexing.diversify(fused, vectors, limit))
    about: dict[str, Any] = {
        "searched": sorted(name for name, ids in ranked.items() if ids),
        "cached": False,
    }
    if note:
        about["degraded"] = note
    return hits, about


def search(
    question: str,
    limit: int = 3,
    count: Count | None = None,
    budget: int = indexing.SEARCH_TOKENS,
    db: sqlite3.Connection | None = None,
    ttl: float = indexing.SEARCH_TTL,
) -> dict[str, Any]:
    """The units nearest the question, each with its lines.

    The cache is keyed on the question with a marker in front of it,
    so that the same words asked of the documents and of the source
    are two questions rather than one answer served to both.
    """
    if not question.strip():
        raise IndexingError("a search needs something to search for")
    own = db is None
    db = db or connect()
    try:
        if not db.execute("SELECT 1 FROM units LIMIT 1").fetchone():
            raise IndexingError(
                f"{indexing.index_file()} holds no source; index_path "
                "adds a file"
            )
        key = f"code:{question}"
        found = indexing.remembered(db, key, limit, ttl) if ttl > 0 else None
        if found is not None:
            hits, about = found
        else:
            hits, about = _look(db, question, limit)
            if ttl > 0:
                indexing.remember(db, key, limit, hits, about)
        answer: dict[str, Any] = {"index": str(indexing.index_file())} | about
        if count is None:
            answer["trimmed"] = False
            return indexing._trim(answer, hits, lambda _text: 0, budget)
        answer["trimmed"] = True
        answer["budget_tokens"] = budget
        return indexing._trim(answer, hits, count, budget)
    finally:
        if own:
            db.close()


def connect(path: Path | None = None) -> sqlite3.Connection:
    """The index, with the code tables ready."""
    db = indexing.connect(path)
    prepare(db)
    return db


def indexed(db: sqlite3.Connection | None = None) -> dict[str, Any]:
    """Which files are in the index, and how each was cut."""
    own = db is None
    db = db or connect()
    try:
        rows = db.execute(
            "SELECT path, language, how, units, lines, indexed_at"
            " FROM sources ORDER BY path"
        ).fetchall()
        return {
            "index": str(indexing.index_file()),
            "sources": [
                {
                    "file": str(row["path"]),
                    "language": str(row["language"]),
                    "cut_by": str(row["how"]),
                    "units": int(row["units"]),
                    "lines": int(row["lines"]),
                    "indexed": time.strftime(
                        "%Y-%m-%d %H:%M", time.localtime(row["indexed_at"])
                    ),
                }
                for row in rows
            ],
        }
    finally:
        if own:
            db.close()
