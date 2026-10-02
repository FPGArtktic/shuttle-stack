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
from dataclasses import dataclass, field
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
    ".go",
    ".h",
    ".hh",
    ".hpp",
    ".inc",
    ".mk",
    ".py",
    ".rs",
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
    indexed_at REAL NOT NULL,
    location   TEXT NOT NULL DEFAULT ''
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
CREATE TABLE IF NOT EXISTS edges (
    source TEXT NOT NULL,
    kind   TEXT NOT NULL,
    name   TEXT NOT NULL,
    line   INTEGER NOT NULL,
    PRIMARY KEY (source, kind, name, line)
);
CREATE INDEX IF NOT EXISTS edges_name ON edges (name, kind);
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
class Edge:
    """One reference out of a file: a name and the line naming it."""

    kind: str
    name: str
    line: int


@dataclass(frozen=True)
class Cut:
    """What the chunker made of one file."""

    file: str
    language: str
    how: str
    why: str
    units: list[Unit]
    edges: list[Edge] = field(default_factory=list)


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
        edges=[
            Edge(
                kind=str(one["kind"]),
                name=str(one["name"]),
                line=int(one["line"]),
            )
            for one in answer.get("edges", [])
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
    db.execute("DELETE FROM edges WHERE source = ?", (file,))
    db.execute("DELETE FROM sources WHERE path = ?", (file,))
    db.execute("DELETE FROM searches")
    return len(ids)


def store(
    db: sqlite3.Connection,
    file: str,
    cut: Cut,
    lines: int,
    location: str = "",
) -> int:
    """Embed the units and put them in, replacing an earlier pass.

    One transaction: a file half indexed would answer about the half
    that made it and say nothing about the rest.

    The location is where the file was read from. A citation needs
    only the name, but anything that wants to open the file again
    needs the path, and storing the name alone meant the index knew
    `counter_top.sv` was indexed and not where it is.
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
        db.executemany(
            "INSERT OR REPLACE INTO edges (source, kind, name, line)"
            " VALUES (?, ?, ?, ?)",
            [(file, one.kind, one.name, one.line) for one in cut.edges],
        )
        db.execute(
            "INSERT INTO sources (path, language, how, units, lines,"
            " indexed_at, location) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                file,
                cut.language,
                cut.how,
                len(cut.units),
                lines,
                time.time(),
                location,
            ),
        )
    return len(cut.units)


# What a question about a name is asking for, by the edge kinds that
# answer it. Named rather than inferred: "who instantiates counter"
# and "what does this recipe depend on" are different questions over
# the same table, and guessing from the wording is how a graph query
# starts returning the wrong direction.
RELATIONS: dict[str, tuple[str, ...]] = {
    "instantiates": ("instantiate",),
    "includes": ("include", "require", "source"),
    "calls": ("call",),
    "uses": ("use", "inherit"),
    "depends": ("depends", "rdepends"),
    "provides": ("provides",),
    "assigns": ("assign",),
}
# How many places a name may be referred to from before the answer
# starts being a file listing rather than an answer.
MAX_REFERENCES = 40
# What may follow a name and still be the same thing: a bitbake
# override, a Rust path, a Go selector, a VHDL library. The separator
# has to be present, so that a question about `counter` is not
# answered about `counterweight`.
SEPARATORS = (":", "::", ".")


def located(db: sqlite3.Connection, name: str) -> Path | None:
    """Where an indexed file was read from, if it is still there.

    The index is the allowlist: a name it does not hold does not
    resolve, so nothing can read a path nobody indexed.
    """
    row = db.execute(
        "SELECT location FROM sources WHERE path = ?", (Path(name).name,)
    ).fetchone()
    if row is None or not row["location"]:
        return None
    where = Path(str(row["location"]))
    return where if where.is_file() else None


def references(
    db: sqlite3.Connection,
    name: str,
    relation: str = "",
    limit: int = MAX_REFERENCES,
) -> dict[str, Any]:
    """Every place that names this thing, as file and line.

    The exact question M5 asks of a scout: who instantiates module X,
    and which file overrides variable Y. It is a lookup and not a
    search — no model, no embedding, no ranking — because the answer
    is a fact the parse already established and a ranked guess at it
    would be worse.

    A name is matched with whatever follows it as well as on its own,
    so asking about RDEPENDS finds the file setting RDEPENDS:${PN},
    asking about `fmt` finds the call to `fmt.Println`, and asking
    about `std` finds `use std::io::Write`. The separator has to be
    there: asking about `counter` does not match `counterweight`.
    """
    if not name.strip():
        raise IndexingError("a reference needs a name to look for")
    kinds = RELATIONS.get(relation, ()) if relation else ()
    if relation and not kinds:
        raise IndexingError(
            f"{relation!r} is not a relation; the ones there are: "
            + ", ".join(sorted(RELATIONS))
        )
    sql, args = _matching(name)
    query = f"SELECT source, kind, name, line FROM edges WHERE {sql}"
    if kinds:
        query += " AND kind IN (" + ",".join("?" * len(kinds)) + ")"
        args.extend(kinds)
    query += " ORDER BY source, line LIMIT ?"
    args.append(limit + 1)
    rows = db.execute(query, args).fetchall()
    found = [
        {
            "at": f"{row['source']}:{row['line']}",
            "relation": str(row["kind"]),
            "names": str(row["name"]),
        }
        for row in rows[:limit]
    ]
    answer: dict[str, Any] = {"name": name, "references": found}
    if relation:
        answer["relation"] = relation
    if len(rows) > limit:
        answer["more_than"] = limit
    if not found:
        answer["note"] = (
            "nothing in the index names it; index_path reads a file in, "
            "and list_indexed says what is already there"
        )
    return answer


# --- two-hop expansion ------------------------------------------------
#
# A reference is one hop: this file names that thing, at this line.
# The second hop is what the place doing the naming reaches in turn,
# and that is where both of M5's scout questions actually live. "Who
# instantiates counter" is answered by the first hop. "Which layer
# overrides RDEPENDS" is not: the answer is the recipe assigning it
# and what that recipe inherits, and the inherit is only reachable
# from the assignment.
#
# It is a join, not a search. Every edge carries the line it sits on
# and every unit the span it occupies, so the unit enclosing an edge
# is a comparison. No model, no embedding, no ranking.

# How many definitions, and how many naming sites, to walk out of. A
# name defined in six places is a name the question was wrong about.
MAX_DEFINITIONS = 4
# How many units one name may cover. A module's span is the lowest and
# highest line of its whole family, so the cap is on inner units and
# not on definitions.
MAX_FAMILY = 200
# How many reached names to report. Without a cap the second hop is a
# transitive closure of the repository.
MAX_REACHED = 12
# Which relation answers "what is this place" first. The second hop
# is capped, so this order decides what makes the budget. It is a
# fixed precedence over relation kinds and not a score: what a file
# pulls in says what it is, an instantiation says what it is made of,
# and a variable assignment beside another variable assignment is the
# recipe's header rather than an answer.
#
# Measured over meta-virtualization. By line order, expanding
# RDEPENDS filled all twelve slots with SUMMARY, DESCRIPTION,
# HOMEPAGE, SECTION, LICENSE and DEPENDS from lines 1 to 14 of
# openvswitch.inc, and the `inherit autotools` on line 75 — the one
# structural fact in the file — did not make the budget.
PRECEDENCE = (
    "inherit",
    "require",
    "include",
    "source",
    "use",
    "instantiate",
    "provides",
    "rprovides",
    "depends",
    "rdepends",
    "declare",
    "call",
    "assign",
)
# How many edges to rank before cutting. The whole span is read, so
# that a late `inherit` is not lost to an early assignment, and a
# kind nobody listed sorts after every kind that is.
MAX_CANDIDATES = 400
# What a file pulls in says what the file is, wherever in the file it
# says so. A bbappend's `inherit` is at the top and the variable it
# overrides at the bottom, so the unit around the override holds
# neither; the rest of the file is read for these kinds alone, which
# keeps the structure without the expansion becoming a file listing.
STRUCTURAL = ("inherit", "require", "include", "source", "use")
# How much of a definition to show. The chunker emits a container's
# head as its own unit, so for HDL these lines are the ports and the
# parameters and for C they are the signature.
HEAD_LINES = 12


@dataclass(frozen=True)
class Span:
    """A stretch of one file, and why the expansion is looking at it."""

    source: str
    first_line: int
    last_line: int
    why: str
    only: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, int, int, tuple[str, ...]]:
        return (self.source, self.first_line, self.last_line, self.only)


def same_name(name: str, other: str) -> bool:
    """Whether the second is the first, or the first with a tail.

    `RDEPENDS` and `RDEPENDS:${PN}` are the same variable, so an
    expansion of one must not report the other as something it
    reaches: a name reaching itself is not a hop.
    """
    return other == name or any(
        other.startswith(f"{name}{one}") for one in SEPARATORS
    )


def _matching(name: str) -> tuple[str, list[Any]]:
    """SQL for "this name, or this name with a separator after it"."""
    quoted = name.replace("\\", "\\\\").replace("%", "\\%")
    quoted = quoted.replace("_", "\\_")
    sql = (
        "(name = ?"
        + "".join(" OR name LIKE ? ESCAPE '\\'" for _ in SEPARATORS)
        + ")"
    )
    return sql, [name, *(f"{quoted}{one}%" for one in SEPARATORS)]


def _family(
    db: sqlite3.Connection, name: str, limit: int
) -> list[sqlite3.Row]:
    """Every unit the name covers: the thing, and what is inside it.

    The chunker names a container's inner units `container.inner`, so
    the separator match that finds `RDEPENDS:${PN}` from `RDEPENDS`
    also finds a module's body from the module. That is what makes the
    span below the whole module rather than only its port list, and
    without it nothing a module instantiates is one hop away.
    """
    sql, args = _matching(name)
    return list(
        db.execute(
            "SELECT source, kind, name, first_line, last_line, text"
            f" FROM units WHERE {sql} ORDER BY source, first_line LIMIT ?",
            [*args, limit],
        )
    )


def _as_definition(row: sqlite3.Row) -> dict[str, Any]:
    head = str(row["text"]).splitlines()[:HEAD_LINES]
    return {
        "at": f"{row['source']}:{row['first_line']}",
        "kind": str(row["kind"]),
        "name": str(row["name"] or ""),
        "lines": f"{row['first_line']}-{row['last_line']}",
        "head": "\n".join(head).rstrip(),
    }


def defines(
    db: sqlite3.Connection, name: str
) -> tuple[list[dict[str, Any]], list[Span]]:
    """Where a name is defined, and the spans to walk out of.

    What is shown is the unit named exactly that, because a module's
    header is the definition and its sixteen always blocks are not.
    What is walked is the family's whole extent.
    """
    rows = _family(db, name, MAX_FAMILY)
    exact = [row for row in rows if str(row["name"] or "") == name]
    shown = (exact or rows)[:MAX_DEFINITIONS]
    spans: list[Span] = []
    for source in dict.fromkeys(str(row["source"]) for row in rows):
        here = [row for row in rows if str(row["source"]) == source]
        spans.append(
            Span(
                source,
                min(int(row["first_line"]) for row in here),
                max(int(row["last_line"]) for row in here),
                f"definition of {name}",
            )
        )
    return [_as_definition(row) for row in shown], spans[:MAX_DEFINITIONS]


def _enclosing(
    db: sqlite3.Connection, source: str, line: int, why: str
) -> Span | None:
    """The smallest indexed unit holding that line."""
    row = db.execute(
        "SELECT first_line, last_line FROM units WHERE source = ?"
        " AND first_line <= ? AND last_line >= ?"
        " ORDER BY last_line - first_line LIMIT 1",
        (source, line, line),
    ).fetchone()
    if row is None:
        return None
    return Span(source, int(row["first_line"]), int(row["last_line"]), why)


def _also_from(
    db: sqlite3.Connection,
    name: str,
    naming: list[dict[str, Any]],
    spans: list[Span],
) -> list[Span]:
    """Add the units that name it to the ones that define it."""
    out = list(spans)
    seen = {one.key for one in out}
    for one in naming[:MAX_DEFINITIONS]:
        source, _, line = str(one["at"]).rpartition(":")
        if not line.isdigit():
            continue
        held = _enclosing(db, source, int(line), f"names {name}")
        if held is None or held.key in seen:
            continue
        seen.add(held.key)
        out.append(held)
        whole = Span(source, 0, 1 << 30, f"what {source} pulls in", STRUCTURAL)
        if whole.key not in seen:
            seen.add(whole.key)
            out.append(whole)
    return out


def _candidates(
    db: sqlite3.Connection, span: Span, name: str
) -> list[sqlite3.Row]:
    """Every edge one span holds, in the order worth reporting."""
    query = (
        "SELECT kind, name, line FROM edges WHERE source = ?"
        " AND line BETWEEN ? AND ?"
    )
    args: list[Any] = [span.source, span.first_line, span.last_line]
    if span.only:
        query += " AND kind IN (" + ",".join("?" * len(span.only)) + ")"
        args.extend(span.only)
    query += " ORDER BY line LIMIT ?"
    args.append(MAX_CANDIDATES)
    rows = [
        row
        for row in db.execute(query, args)
        if not same_name(name, str(row["name"]))
    ]
    place = {kind: at for at, kind in enumerate(PRECEDENCE)}
    rows.sort(
        key=lambda row: (
            place.get(str(row["kind"]), len(PRECEDENCE)),
            int(row["line"]),
        )
    )
    return rows


def _next_unseen(
    rows: list[sqlite3.Row], seen: set[tuple[str, str]]
) -> sqlite3.Row | None:
    """The next edge nobody has reported yet, taken off the queue."""
    while rows:
        row = rows.pop(0)
        key = (str(row["kind"]), str(row["name"]))
        if key not in seen:
            seen.add(key)
            return row
    return None


def _as_reached(
    db: sqlite3.Connection, span: Span, row: sqlite3.Row
) -> dict[str, Any]:
    where = _family(db, str(row["name"]), 1)
    return {
        "name": str(row["name"]),
        "relation": str(row["kind"]),
        "named_at": f"{span.source}:{row['line']}",
        "through": span.why,
        "defined_at": (
            f"{where[0]['source']}:{where[0]['first_line']}" if where else None
        ),
    }


def _by_turn(
    held: list[tuple[Span, list[sqlite3.Row]]],
) -> list[tuple[Span, list[sqlite3.Row]]]:
    """Order the queues so the rotation visits each file once a round.

    A file contributes two spans — the unit around the name and what
    the file pulls in — and taking them one after the other gives it
    two slots before another file gets one. The sort is stable, so
    within a round the files keep the order they came in.
    """
    turn: dict[str, int] = {}
    numbered = []
    for one in held:
        at = turn.get(one[0].source, 0)
        turn[one[0].source] = at + 1
        numbered.append((at, one))
    numbered.sort(key=lambda one: one[0])
    return [one for _, one in numbered]


def _reached(
    db: sqlite3.Connection, spans: list[Span], name: str, limit: int
) -> list[dict[str, Any]]:
    """What those spans name in turn, each span taking a turn.

    Round-robin rather than one ranked list. Expanding EXTRA_OECONF
    over meta-virtualization, where four layers assign it, a single
    order spent all twelve slots on openvswitch.inc — five inherits
    and six DEPENDS — and said nothing about the other three. The
    question is what each of those layers is, so each one answers
    before any of them answers twice.
    """
    held = [(span, _candidates(db, span, name)) for span in spans]
    held = _by_turn(held)
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    while len(out) <= limit:
        took = False
        for span, rows in held:
            row = _next_unseen(rows, seen)
            if row is None:
                continue
            took = True
            out.append(_as_reached(db, span, row))
            if len(out) > limit:
                return out
        if not took:
            return out
    return out


def expand(
    db: sqlite3.Connection,
    name: str,
    hops: int = 2,
    limit: int = MAX_REACHED,
) -> dict[str, Any]:
    """A name: where it is defined, who names it, what that reaches.

    One hop is the lookup `find_references` already does. Two hops
    adds what the defining and the naming units refer to themselves,
    with the location of each of those, so a scout gets the next place
    to look without a second question and without grepping for it.

    Everything here is a fact the parse established. A `defined_at` of
    null means the index holds no definition of that name, which is
    what a call into a library looks like; it is not a failure and
    guessing at it would be worse.
    """
    if not name.strip():
        raise IndexingError("an expansion needs a name to start from")
    if hops not in (1, 2):
        raise IndexingError(f"an expansion is one hop or two, not {hops}")
    found, spans = defines(db, name)
    naming = references(db, name)["references"]
    answer: dict[str, Any] = {
        "name": name,
        "hops": hops,
        "defined": found,
        "named_from": naming,
    }
    if hops == 2:
        walk = _also_from(db, name, naming, spans)
        reached = _reached(db, walk, name, limit)
        answer["reaches"] = reached[:limit]
        if len(reached) > limit:
            answer["more_than"] = limit
    if not found and not naming:
        answer["note"] = (
            "nothing in the index defines it or names it; index_path "
            "reads a file in, and list_indexed says what is there"
        )
    return answer


def _lexical(
    db: sqlite3.Connection, question: str, file: str = ""
) -> list[int]:
    query = indexing.terms(question)
    if not query:
        return []
    sql = "SELECT rowid FROM units_fts WHERE units_fts MATCH ?"
    args: list[Any] = [query]
    if file:
        sql += " AND rowid IN (SELECT id FROM units WHERE source = ?)"
        args.append(file)
    sql += " ORDER BY rank LIMIT ?"
    args.append(indexing.CANDIDATES)
    try:
        rows = db.execute(sql, args).fetchall()
    except sqlite3.OperationalError as error:
        raise IndexingError(
            f"the lexical index refused {query!r}: {error}"
        ) from error
    return [int(row["rowid"]) for row in rows]


def _semantic(
    db: sqlite3.Connection, vector: list[float], file: str = ""
) -> list[int]:
    """The nearest units, from the whole index or from one file.

    The same shape as the document half, and for the same reason:
    vec0 answers a KNN query over the table and takes no condition on
    the file, so narrowing by taking the top k of everything and then
    dropping the other files loses a unit that is the nearest within
    its own file and far down the index.
    """
    if not file:
        rows = db.execute(
            "SELECT id FROM vec_units WHERE embedding MATCH ?"
            " AND k = ? ORDER BY distance",
            (indexing._blob(vector), indexing.CANDIDATES),
        ).fetchall()
        return [int(row["id"]) for row in rows]
    mine = [
        int(row["id"])
        for row in db.execute(
            "SELECT id FROM units WHERE source = ? ORDER BY ordinal LIMIT ?",
            (file, indexing.MAX_IN_DOCUMENT),
        )
    ]
    held = _vectors(db, mine)
    ranked = sorted(
        held.items(), key=lambda one: -indexing._cosine(vector, one[1])
    )
    return [one[0] for one in ranked[: indexing.CANDIDATES]]


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
    db: sqlite3.Connection, question: str, limit: int, file: str = ""
) -> tuple[list[Hit], dict[str, Any]]:
    ranked = {"lexical": _lexical(db, question, file)}
    note = ""
    try:
        ranked["semantic"] = _semantic(db, indexing.embed([question])[0], file)
    except IndexingError as error:
        note = f"the semantic half is missing: {error}"
    fused = indexing.fuse(ranked)
    vectors = _vectors(db, [entry[0] for entry in fused])
    hits = _rows(db, indexing.diversify(fused, vectors, limit))
    about: dict[str, Any] = {
        "searched": sorted(name for name, ids in ranked.items() if ids),
        "cached": False,
    }
    if file:
        about["within"] = file
        # What the file holds, beside what matched, for the reason the
        # document half carries its sections: a ranking answers which
        # unit fits the question and not what is in the file, and the
        # question a loop sends is its own whole request.
        about["units"] = _unit_names(db, file)
    if note:
        about["degraded"] = note
    return hits, about


def _unit_names(db: sqlite3.Connection, file: str) -> list[str]:
    """The file's unit names, for a narrowed search to carry."""
    rows = db.execute(
        "SELECT kind, name, first_line FROM units WHERE source = ?"
        " ORDER BY ordinal LIMIT ?",
        (file, indexing.IN_SEARCH + 1),
    ).fetchall()
    held = [
        f"{row['name'] or row['kind']}:{row['first_line']}"
        for row in rows[: indexing.IN_SEARCH]
    ]
    if len(rows) > indexing.IN_SEARCH:
        held.append(f"... and more than {indexing.IN_SEARCH}")
    return held


def search(
    question: str,
    limit: int = 3,
    count: Count | None = None,
    budget: int = indexing.SEARCH_TOKENS,
    db: sqlite3.Connection | None = None,
    ttl: float = indexing.SEARCH_TTL,
    file: str = "",
) -> dict[str, Any]:
    """The units nearest the question, each with its lines.

    The cache is keyed on the question with a marker in front of it,
    so that the same words asked of the documents and of the source
    are two questions rather than one answer served to both, and the
    file a search was narrowed to is part of that marker for the same
    reason.
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
        file = Path(file).name if file else ""
        if (
            file
            and not db.execute(
                "SELECT 1 FROM units WHERE source = ? LIMIT 1", (file,)
            ).fetchone()
        ):
            raise IndexingError(
                f"{file} is not an indexed source file; list_indexed "
                "says what is there, and index_path reads one in"
            )
        key = f"code:{file}:{question}"
        found = indexing.remembered(db, key, limit, ttl) if ttl > 0 else None
        if found is not None:
            hits, about = found
        else:
            hits, about = _look(db, question, limit, file)
            if ttl > 0:
                indexing.remember(db, key, limit, hits, about)
        answer: dict[str, Any] = dict(about)
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
