# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mateusz Okulanis
"""One SQLite file holding the index: BM25 beside the vectors.

The point is a citation. A section is stored with the file, the page
and the clause it came from, so an answer can name where it was read
rather than asking anyone to take its word. The caller gets a few
hundred tokens and three facts about each of them.

Two ways of asking, because they fail differently. A register name or
a clause number is a lexical question: FTS5 finds `TIMER0` or `4.2.1`
and a vector search is as likely to return the paragraph next to it.
"what resets the peripheral" is the opposite, and BM25 cannot see the
word the document used instead. The two result lists are fused by
reciprocal rank and then thinned by MMR, which drops the second copy
of a section the document repeats.

The file is the whole index: the tables are the ones the WEFT OCR
module writes, so a document indexed there is searchable here and a
copy of the index to an air-gapped machine is one `cp`. The FTS5 table
is kept by triggers on `chunks`, so either side's writes stay
searchable by the other.

The embeddings come from bge-m3 through Ollama, on the CPU. Measured
here, a batch of thirty-two sections costs 62 ms each on the CPU
against 43 ms on the GPU, and the GPU version takes 289 MiB of a 4 GiB
card that shuttle-long is already using: a saving of 19 ms per section
is not worth taking memory from the server answering the questions.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import struct
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunking import chunk
from .documents import MARKER
from .runs import home

OLLAMA = os.environ.get("SHUTTLE_EMBED_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("SHUTTLE_EMBED_MODEL", "bge-m3")

DIMENSIONS = 1024
BATCH = 32
# bge-m3 holds 8192 tokens; a section longer than this is split, so a
# standard's forty-page clause is still searchable.
SECTION_CHARS = 6000
# What comes back per hit before trimming, and the budget the whole
# answer is held to. The budget is in tokens because the promise is in
# tokens: a character budget does not keep it. Measured on the
# datasheet, 1066 characters of its dot leaders and numbers came to 549
# tokens — half a token a character, against a third for prose — so the
# same character count means two very different answers.
EXCERPT_CHARS = 700
SEARCH_TOKENS = 300
MIN_EXCERPT = 60
# How much of one section `section()` hands back. Named apart from
# SECTION_CHARS above, which is how much of a section is stored: the
# two are different questions and giving them one name quietly
# changed the stored size, which the tests did not catch because they
# check a section against whatever the constant says.
READ_CHARS = 4000
# How many sections to name when a heading matches several.
MORE_SECTIONS = 5
# How many sections of one document to rank when a search is narrowed
# to it. A report of five thousand lines came out in 483 sections, so
# this is a ceiling rather than a limit anyone meets.
MAX_IN_DOCUMENT = 1000
# How many section names to list when a document is asked for without
# one. A report of five thousand lines has 483 of them, which is a
# listing nobody reads; forty is a table of contents.
MAX_CONTENTS = 40
# How many of them a narrowed search carries. Fewer, because they are
# paid for out of the same budget as the excerpts.
IN_SEARCH = 12
# Below this a section is a line of a contents page rather than a
# place to read, so it is left out of the listing.
WORTH_NAMING = 80
TIMEOUT = 300.0
# How many each side contributes before fusion. Wider than the answer
# because the whole point of two channels is that a section ranked
# eighth by one may be ranked first by the other.
CANDIDATES = 20
# The constant in 1/(k + rank). 60 is the value the RRF paper settled
# on; it flattens the top of each list enough that a confident wrong
# first place cannot carry the fusion on its own.
RRF_K = 60
# How much of MMR's score is relevance against novelty. At 1.0 it is
# not MMR at all; at 0.5 a datasheet's repeated boilerplate starts
# outranking a second genuinely relevant clause.
MMR_LAMBDA = 0.7
# How long a search result is served from the cache. The number is
# arbitrary and the staleness it guards against is not: an index that
# has changed invalidates every cached search regardless of the clock,
# which is what the stamp below is for. The TTL is only there so that
# a cache nobody cleared does not answer next month's question with
# last month's ranking.
SEARCH_TTL = float(os.environ.get("SHUTTLE_SEARCH_TTL", 3600.0))
# A numbered clause or a Markdown heading. The word after the number
# has to look like a word: the first version asked only for a number
# followed by whitespace, and a datasheet is mostly tables of numbers,
# so every row of a timing table became a heading and the citations
# read "8" and "15 5". A clause is "4 Scope" or "3.2.1 Registers"; a
# table row is not, and neither is "3.3 V".
HEADING = r"^\s*(\d+(\.\d+)*\.?\s+[A-Z][a-z]{2,}|#{1,6}\s+\S)"
# A document organised by headings has several. One match is a numbered
# footnote: in the datasheet measured here, the single line "1. The
# package thermal impedance is calculated..." put the whole file into
# heading mode and then every page of it was cited under that footnote.
MIN_HEADINGS = 3
# The other way a report titles a section: a one-cell row of an ASCII
# table, under a rule with no interior join.
#
#     +--------------------------------------------------+
#     ; Slow 1200mV 125C Model Fmax Summary              ;
#     +------------+-----------------+------------+------+
#     ; Fmax       ; Restricted Fmax ; Clock Name ; Note ;
#
# The one cell is what tells a title from a header row, and the
# unbroken rule above it says the same thing twice. This is a text
# convention and not vendor knowledge — no more so than `## ` — and
# the reason it is needed is measured: a 631-line timing report put
# its table of contents in as 39 headings of forty characters each,
# because those lines are numbered and the real titles are not, and
# then every table in the file, Fmax included, landed in 23 sections
# all called "40. Timing Analyzer Messages".
RULE = re.compile(r"^\+-+\+$")
TITLE = re.compile(r"^;\s*([^;]*?)\s*;$")
PAGE = re.compile(re.escape(MARKER).replace(r"\{page\}", r"(\d+)"))
# The number at the front of a heading, kept apart from its words so
# that "4.2.1" is a term a lexical search can hit.
CLAUSE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+(.*)$")
# What FTS5 is given. Dots and hyphens stay inside a term so that
# "4.2.1" and "GPIO-A" survive as phrases; unicode61 splits them into
# their parts, and a quoted phrase puts them back in order.
TERM = re.compile(r"[\w.\-/]+")
# Every term is asked for, common ones included. Dropping the terms
# the index is full of looked obviously right and was measured to be
# worth nothing: on 40 exact-term questions and 22 written by
# shuttle-long from the sections they answer, thresholds from 0.25 to
# 1.0 of the index gave identical top-three recall, and 0.1 lost one.
# A 44-section index is too small for document frequency to tell
# "before" from "absolute"; both appear once. BM25 weighs the terms,
# and the fusion below is what keeps a noisy channel in its place.
Count = Callable[[str], int]

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    path        TEXT PRIMARY KEY,
    title       TEXT,
    designation TEXT,
    doc_type    TEXT,
    pages       INTEGER NOT NULL,
    ocr_pages   INTEGER NOT NULL,
    chunks      INTEGER NOT NULL,
    dimensions  INTEGER,
    indexed_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    document TEXT NOT NULL,
    ordinal  INTEGER NOT NULL,
    page     INTEGER NOT NULL,
    clause   TEXT,
    heading  TEXT,
    text     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_document ON chunks (document, ordinal);
CREATE INDEX IF NOT EXISTS chunks_clause ON chunks (clause);
CREATE TABLE IF NOT EXISTS digests (
    chunk_id INTEGER PRIMARY KEY,
    summary  TEXT NOT NULL,
    made_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS document_digests (
    path     TEXT PRIMARY KEY,
    summary  TEXT NOT NULL,
    sections INTEGER NOT NULL,
    made_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS searches (
    question TEXT NOT NULL,
    wanted   INTEGER NOT NULL,
    stamp    TEXT NOT NULL,
    hits     TEXT NOT NULL,
    made_at  REAL NOT NULL,
    PRIMARY KEY (question, wanted)
);
"""

# External content, so the text is stored once. The triggers live in
# the file rather than in this module on purpose: WEFT writes the same
# tables, and its inserts have to reach the lexical index too.
FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, clause, heading, content='chunks', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS chunks_fts_insert
AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, clause, heading)
    VALUES (new.id, new.text, new.clause, new.heading);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_delete
AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, clause, heading)
    VALUES ('delete', old.id, old.text, old.clause, old.heading);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_update
AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, clause, heading)
    VALUES ('delete', old.id, old.text, old.clause, old.heading);
    INSERT INTO chunks_fts(rowid, text, clause, heading)
    VALUES (new.id, new.text, new.clause, new.heading);
END;
"""

VECTORS = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
    f"id INTEGER PRIMARY KEY, embedding float[{DIMENSIONS}])"
)


class IndexingError(RuntimeError):
    """The index cannot be read or written, and why."""


@dataclass(frozen=True)
class Section:
    """One citable piece of a document."""

    page: int
    heading: str
    text: str

    @property
    def clause(self) -> str:
        """The numbering at the front of the heading, if it has any."""
        found = CLAUSE.match(self.heading)
        return found.group(1) if found else ""


@dataclass(frozen=True)
class Hit:
    """One section, and how it was found."""

    file: str
    page: int
    clause: str
    heading: str
    text: str
    score: float
    how: str
    # A line written overnight saying what the section is about, where
    # one has been written. A page of a datasheet that is all numbers
    # has an excerpt nobody can read and a summary anyone can.
    summary: str = ""

    def report(self, room: int) -> dict[str, Any]:
        """The hit, with as much of its text as there is room for.

        Below the floor the excerpt stops being worth its tokens:
        sixty characters of a datasheet's numbers say nothing, and a
        line saying what the section is about says something. The
        summary stands in only there, and the reply says the text was
        left out so that nobody quotes a label. Carrying both at
        every size was measured to cost two of three hits.
        """
        entry: dict[str, Any] = {
            "file": self.file,
            "page": self.page,
            "section": self.heading,
            "score": round(self.score, 4),
            "found_by": self.how,
        }
        if self.clause:
            entry["clause"] = self.clause
        if room <= MIN_EXCERPT and self.summary:
            entry["summary"] = self.summary
        else:
            entry["text"] = self.text[:room]
        return entry


def index_file() -> Path:
    """The one file. SHUTTLE_INDEX points at a shared one."""
    value = os.environ.get("SHUTTLE_INDEX")
    return Path(value) if value else home() / "documents.sqlite"


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open the index, making the tables and the triggers if absent.

    Loading the vector extension is not optional and not something to
    work around: without it a search would answer lexically and look
    as though it had answered, which is the one failure this would
    rather be loud about.
    """
    where = path or index_file()
    where.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(where)
    db.row_factory = sqlite3.Row
    try:
        import sqlite_vec

        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
    except (ImportError, AttributeError, sqlite3.Error) as error:
        db.close()
        raise IndexingError(
            f"sqlite-vec will not load ({error}); the index needs it for "
            "the vector half of every search"
        ) from error
    db.executescript(SCHEMA)
    db.execute(VECTORS)
    _ensure_fts(db)
    db.commit()
    return db


def _ensure_fts(db: sqlite3.Connection) -> None:
    """Add the lexical index, and fill it in if rows came first.

    A file written by WEFT has chunks and no FTS5 table. Creating one
    over an existing content table leaves it empty until it is told to
    rebuild, and an empty lexical index is exactly the silent half
    answer this module is trying not to give.
    """
    found = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        ("chunks_fts",),
    ).fetchone()
    if found:
        return
    db.executescript(FTS)
    db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")


def _post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as answer:
            parsed = json.loads(answer.read())
    except urllib.error.HTTPError as error:
        raise IndexingError(
            f"{url}: HTTP {error.code}: "
            f"{error.read().decode(errors='replace')[:200]}"
        ) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise IndexingError(f"{url}: {error}") from error
    if not isinstance(parsed, dict):
        raise IndexingError(
            f"{url}: expected a JSON object, got {type(parsed).__name__}"
        )
    return parsed


def embed(texts: list[str]) -> list[list[float]]:
    """Embed on the CPU, in batches, because the GPU is in use."""
    out: list[list[float]] = []
    for start in range(0, len(texts), BATCH):
        batch = texts[start : start + BATCH]
        answer = _post(
            f"{OLLAMA}/api/embed",
            {"model": MODEL, "input": batch, "options": {"num_gpu": 0}},
        )
        vectors = answer.get("embeddings") or []
        if len(vectors) != len(batch):
            raise IndexingError(
                f"{MODEL} returned {len(vectors)} embeddings for "
                f"{len(batch)} inputs"
            )
        out.extend(vectors)
    return out


def _blob(vector: Iterable[float]) -> bytes:
    values = [float(v) for v in vector]
    if len(values) != DIMENSIONS:
        raise IndexingError(
            f"the index holds {DIMENSIONS}-dimension vectors, "
            f"{MODEL} returned {len(values)}"
        )
    return struct.pack(f"{DIMENSIONS}f", *values)


def _floats(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{DIMENSIONS}f", blob))


def split(text: str, heading: str = HEADING) -> tuple[list[Section], str]:
    """The document's sections, and how they were found.

    A heading pattern that matches nothing falls back to one section
    per page, and the answer says which happened: a caller told it
    indexed sections should be able to find out it indexed pages.
    """
    lines = text.splitlines()
    titles = _headings(lines, re.compile(heading))
    if len(titles) < MIN_HEADINGS:
        return _within_limit(_by_page(text)), "pages"

    found: list[Section] = []
    page = 0
    at_page, heading_text = 0, "preamble"
    held: list[str] = []

    def close() -> None:
        """Keep the section, unless it is nothing but its own name.

        A report's table of contents is a list of the headings that
        follow it, and the heading pattern matches every line of it,
        so each becomes a section holding one line: its own name. They
        are not answers and they are decoys -- BM25 favours a short
        text, and doc-qa read `6. Slow 1200mV 125C Model Fmax
        Summary`, all seventeen characters of it, and reported that
        the report does not state the Fmax. Thirty-seven of that
        file's eighty-two sections were these; prose documents have
        none.
        """
        if not held:
            return
        text = "\n".join(held)
        if text.strip() == heading_text.strip():
            return
        found.append(Section(at_page, heading_text, text))

    for at, line in enumerate(lines):
        seen = PAGE.match(line.strip())
        if seen:
            page = int(seen.group(1))
            # A section may not span pages. The page is half the
            # citation, so a section running from page 3 to page 37
            # would be cited as page 3 and the reader sent to the
            # wrong one; it is closed here and continued under the
            # same heading on the new page.
            close()
            held = []
            at_page = page
            continue
        if at in titles:
            close()
            held = []
            at_page, heading_text = page, titles[at]
        held.append(line)
    close()
    return _within_limit(found), "headings"


def _headings(lines: list[str], marker: re.Pattern[str]) -> dict[int, str]:
    """Which lines start a section, and what each one is called.

    Found before the walk rather than during it, because a boxed title
    is recognised by the line above it and because the count decides
    whether this document has headings at all.
    """
    out: dict[int, str] = {}
    for at, line in enumerate(lines):
        if marker.match(line):
            out[at] = line.strip()[:120]
            continue
        boxed = TITLE.match(line.strip())
        if not boxed or not boxed.group(1):
            continue
        # The section starts at the rule above the title and not at
        # the title, so the box is not cut in half: keying it on the
        # title left every section ending in a stray `+----+` and the
        # first one holding nothing but that line.
        if at and RULE.match(lines[at - 1].strip()):
            out[at - 1] = boxed.group(1)[:120]
    return out


def _by_page(text: str) -> list[Section]:
    sections: list[Section] = []
    page = 0
    lines: list[str] = []
    for line in text.splitlines():
        seen = PAGE.match(line.strip())
        if seen:
            if lines:
                sections.append(
                    Section(page, f"page {page}", "\n".join(lines))
                )
            page, lines = int(seen.group(1)), []
            continue
        lines.append(line)
    if lines:
        sections.append(Section(page, f"page {page}", "\n".join(lines)))
    return sections


def _within_limit(sections: list[Section]) -> list[Section]:
    """Split any section the embedding model cannot hold."""
    out: list[Section] = []
    for section in sections:
        if not section.text.strip():
            continue
        for part in chunk(section.text, SECTION_CHARS) or [section.text]:
            out.append(Section(section.page, section.heading, part))
    return out


def forget(db: sqlite3.Connection, file: str) -> int:
    """Drop everything a document contributed. The triggers see it."""
    ids = [
        int(row["id"])
        for row in db.execute(
            "SELECT id FROM chunks WHERE document = ?", (file,)
        )
    ]
    db.executemany("DELETE FROM vec_chunks WHERE id = ?", [(i,) for i in ids])
    db.executemany(
        "DELETE FROM digests WHERE chunk_id = ?", [(i,) for i in ids]
    )
    db.execute("DELETE FROM chunks WHERE document = ?", (file,))
    db.execute("DELETE FROM documents WHERE path = ?", (file,))
    # The sections a digest described are gone, so the line about the
    # document as a whole describes a document that no longer exists
    # in the index.
    db.execute("DELETE FROM document_digests WHERE path = ?", (file,))
    db.execute("DELETE FROM searches")
    return len(ids)


def store(
    db: sqlite3.Connection,
    file: str,
    sections: list[Section],
    ocr_pages: int = 0,
) -> int:
    """Embed the sections and put them in, replacing an earlier pass.

    One transaction, because a document half in the index is worse
    than one not in it: a search would cite the pages that made it in
    and no one would know the rest were missing.
    """
    if not sections:
        raise IndexingError(f"{file}: no sections to index")
    vectors = embed([section.text for section in sections])
    blobs = [_blob(vector) for vector in vectors]
    with db:
        forget(db, file)
        for order, section in enumerate(sections):
            row = db.execute(
                "INSERT INTO chunks"
                " (document, ordinal, page, clause, heading, text)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    file,
                    order,
                    section.page,
                    section.clause,
                    section.heading,
                    section.text,
                ),
            )
            db.execute(
                "INSERT INTO vec_chunks (id, embedding) VALUES (?, ?)",
                (row.lastrowid, blobs[order]),
            )
        db.execute(
            "INSERT INTO documents (path, pages, ocr_pages, chunks,"
            " dimensions, indexed_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                file,
                max((s.page for s in sections), default=0),
                ocr_pages,
                len(sections),
                DIMENSIONS,
                time.time(),
            ),
        )
    return len(sections)


def terms(question: str) -> str:
    """The question as FTS5 reads it.

    Every term is quoted, so a clause number survives as a phrase and
    a stray bracket in the question is text rather than syntax. OR
    rather than AND: BM25 ranks, and requiring every word would mean a
    question with one word the document does not use finds nothing.
    """
    found = []
    for word in TERM.findall(question):
        trimmed = word.strip("./-")
        if len(trimmed) > 1:
            found.append('"' + trimmed.replace('"', '""') + '"')
    return " OR ".join(dict.fromkeys(found))


def _lexical(
    db: sqlite3.Connection, question: str, file: str = ""
) -> list[int]:
    query = terms(question)
    if not query:
        return []
    sql = "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?"
    args: list[Any] = [query]
    if file:
        sql += " AND rowid IN (SELECT id FROM chunks WHERE document = ?)"
        args.append(file)
    sql += " ORDER BY rank LIMIT ?"
    args.append(CANDIDATES)
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
    """The nearest sections, from the whole index or from one document.

    vec0 answers a KNN query over the table and takes no condition on
    the document, so narrowing by taking the top k of everything and
    then dropping the other documents loses a section that is the
    nearest within its file and far down the index. One document holds
    tens of sections, not thousands, so its own vectors are ranked
    here instead -- exactly, and with the cosine the fusion already
    uses.
    """
    if not file:
        rows = db.execute(
            "SELECT id FROM vec_chunks WHERE embedding MATCH ?"
            " AND k = ? ORDER BY distance",
            (_blob(vector), CANDIDATES),
        ).fetchall()
        return [int(row["id"]) for row in rows]
    mine = [
        int(row["id"])
        for row in db.execute(
            "SELECT id FROM chunks WHERE document = ? ORDER BY ordinal"
            " LIMIT ?",
            (file, MAX_IN_DOCUMENT),
        )
    ]
    held = _vectors(db, mine)
    ranked = sorted(held.items(), key=lambda one: -_cosine(vector, one[1]))
    return [one[0] for one in ranked[:CANDIDATES]]


def fuse(ranked: dict[str, list[int]]) -> list[tuple[int, float, str]]:
    """Reciprocal rank fusion, keeping which lists found each row.

    Scores from BM25 and from a cosine distance are not comparable and
    normalising them means inventing a scale; ranks are comparable by
    construction, which is the whole argument for RRF.
    """
    scores: dict[int, float] = {}
    sources: dict[int, list[str]] = {}
    for name, ids in ranked.items():
        for place, row in enumerate(ids, start=1):
            scores[row] = scores.get(row, 0.0) + 1.0 / (RRF_K + place)
            sources.setdefault(row, []).append(name)
    out = [
        (
            row,
            scores[row],
            "both" if len(sources[row]) > 1 else sources[row][0],
        )
        for row in scores
    ]
    out.sort(key=lambda entry: (-entry[1], entry[0]))
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    left = sum(x * x for x in a) ** 0.5
    right = sum(y * y for y in b) ** 0.5
    if not left or not right:
        return 0.0
    return float(dot / (left * right))


def diversify(
    ordered: list[tuple[int, float, str]],
    vectors: dict[int, list[float]],
    limit: int,
) -> list[tuple[int, float, str]]:
    """Maximal marginal relevance over the fused list.

    A datasheet repeats itself: the same note sits under every
    register, and three hits that are the same sentence are one answer
    taking three times the budget. A candidate without an embedding is
    treated as novel rather than dropped, because the alternative is
    quietly losing a lexical hit.
    """
    if limit <= 0:
        return []
    picked: list[tuple[int, float, str]] = []
    rest = list(ordered)
    while rest and len(picked) < limit:
        best: float | None = None
        best_at = 0
        for at, entry in enumerate(rest):
            mine = vectors.get(entry[0])
            penalty = 0.0
            if mine is not None and picked:
                penalty = max(
                    (
                        _cosine(mine, vectors[chosen[0]])
                        for chosen in picked
                        if chosen[0] in vectors
                    ),
                    default=0.0,
                )
            value = MMR_LAMBDA * entry[1] - (1 - MMR_LAMBDA) * penalty
            if best is None or value > best:
                best, best_at = value, at
        picked.append(rest.pop(best_at))
    return picked


def _rows(
    db: sqlite3.Connection, chosen: list[tuple[int, float, str]]
) -> list[Hit]:
    if not chosen:
        return []
    from .digest import summaries

    said = summaries(db, [entry[0] for entry in chosen])
    places = ",".join("?" * len(chosen))
    found = {
        int(row["id"]): row
        for row in db.execute(
            "SELECT id, document, page, clause, heading, text"
            f" FROM chunks WHERE id IN ({places})",
            [entry[0] for entry in chosen],
        )
    }
    hits = []
    for row_id, score, how in chosen:
        row = found.get(row_id)
        if row is None:
            continue
        hits.append(
            Hit(
                file=str(row["document"]),
                page=int(row["page"]),
                clause=str(row["clause"] or ""),
                heading=str(row["heading"] or ""),
                text=str(row["text"]),
                score=score,
                how=how,
                summary=said.get(row_id, ""),
            )
        )
    return hits


def _vectors(
    db: sqlite3.Connection, ids: Iterable[int]
) -> dict[int, list[float]]:
    wanted = list(ids)
    if not wanted:
        return {}
    places = ",".join("?" * len(wanted))
    return {
        int(row["id"]): _floats(row["embedding"])
        for row in db.execute(
            f"SELECT id, embedding FROM vec_chunks WHERE id IN ({places})",
            wanted,
        )
    }


def _trim(
    answer: dict[str, Any], hits: list[Hit], count: Count, budget: int
) -> dict[str, Any]:
    """Fit the whole answer into the budget, and say what it cost.

    The whole answer, not the hits alone: the fields around them cost
    tokens too, and measuring only the hits returned 315 against a
    budget of 300.

    Shortened evenly across the hits, because a long first one
    crowding out the rest would hide the second-best section behind
    the best. When even the shortest excerpts do not fit, the last
    hit goes rather than the budget: three source units cost 277
    tokens in citations alone before a character of their text, so
    holding three of them would have meant thirty characters each,
    and two units somebody can read beat three nobody can.
    """
    from .digest import CAVEAT

    kept = list(hits)
    while True:
        room = EXCERPT_CHARS
        while True:
            reported = [hit.report(room) for hit in kept]
            shortened = answer | {"hits": reported}
            # Said once for the reply rather than once per hit: it is
            # true of every summary in it, and three copies of the
            # same sentence cost thirty tokens of a budget of three
            # hundred.
            if any("summary" in one for one in reported):
                shortened["summaries"] = CAVEAT
                shortened["text_left_out"] = (
                    "no room; ask for fewer hits to see the text"
                )
            if len(kept) < len(hits):
                shortened["hits_dropped"] = len(hits) - len(kept)
            if count(json.dumps(shortened, ensure_ascii=False)) <= budget:
                return shortened
            if room <= MIN_EXCERPT:
                break
            room = max(MIN_EXCERPT, room * 2 // 3)
        if len(kept) <= 1:
            return shortened
        kept.pop()


def stamp(db: sqlite3.Connection) -> str:
    """What the index currently holds, as one short string.

    A cached search is wrong the moment the index changes, and a TTL
    cannot know that. The count and the latest indexing time move for
    a document added, replaced or dropped, including one WEFT wrote,
    so comparing this is cheaper and more honest than hoping an hour
    is short enough.
    """
    row = db.execute(
        "SELECT COUNT(*), COALESCE(MAX(indexed_at), 0) FROM documents"
    ).fetchone()
    return f"{row[0]}:{row[1]:.3f}"


def remembered(
    db: sqlite3.Connection, question: str, limit: int, ttl: float
) -> tuple[list[Hit], dict[str, Any]] | None:
    """A kept answer to this question, if it is still the right one."""
    row = db.execute(
        "SELECT stamp, hits, made_at FROM searches"
        " WHERE question = ? AND wanted = ?",
        (question, limit),
    ).fetchone()
    if row is None:
        return None
    age = time.time() - float(row["made_at"])
    if age > ttl or row["stamp"] != stamp(db):
        db.execute(
            "DELETE FROM searches WHERE question = ? AND wanted = ?",
            (question, limit),
        )
        db.commit()
        return None
    kept = json.loads(row["hits"])
    hits = [Hit(**one) for one in kept["hits"]]
    return hits, {
        "searched": kept["searched"],
        "cached": True,
        "cached_age_seconds": round(age, 1),
    } | ({"degraded": kept["degraded"]} if kept.get("degraded") else {})


def remember(
    db: sqlite3.Connection,
    question: str,
    limit: int,
    hits: list[Hit],
    about: dict[str, Any],
) -> None:
    """Keep the hits, not the report: the budget may be different next
    time, and re-trimming an excerpt is free where re-ranking is not.
    """
    db.execute(
        "INSERT OR REPLACE INTO searches"
        " (question, wanted, stamp, hits, made_at) VALUES (?, ?, ?, ?, ?)",
        (
            question,
            limit,
            stamp(db),
            json.dumps(
                {
                    "hits": [vars(hit) for hit in hits],
                    "searched": about["searched"],
                    "degraded": about.get("degraded", ""),
                }
            ),
            time.time(),
        ),
    )
    db.commit()


def search(
    question: str,
    limit: int = 3,
    count: Count | None = None,
    budget: int = SEARCH_TOKENS,
    db: sqlite3.Connection | None = None,
    ttl: float = SEARCH_TTL,
    file: str = "",
) -> dict[str, Any]:
    """The sections nearest the question, with where each came from.

    Both channels are tried. If the embedding model cannot be reached
    the lexical half still answers and the reply says the other half
    is missing: a search that had quietly become a grep would be worse
    than one that failed.

    The same question asked twice is answered from the file without
    embedding it again, and the reply says so and how old it is. A
    degraded answer is kept like any other, so the note travels with
    it rather than disappearing on the second ask.
    """
    if not question.strip():
        raise IndexingError("a search needs something to search for")
    own = db is None
    db = db or connect()
    try:
        if not db.execute("SELECT 1 FROM chunks LIMIT 1").fetchone():
            raise IndexingError(
                f"{index_file()} holds no documents; index_path adds one"
            )
        file = Path(file).name if file else ""
        if (
            file
            and not db.execute(
                "SELECT 1 FROM chunks WHERE document = ? LIMIT 1", (file,)
            ).fetchone()
        ):
            raise IndexingError(
                f"{file} is not an indexed document; list_indexed says "
                "what is there, and index_path reads one in"
            )
        # The file is part of the key, not a reason to skip the cache.
        # The kept answer is keyed on the question and the number
        # wanted, so without this the narrowed ask would be served
        # the answer about the whole index. The marker is the same
        # device code.search already uses to keep the two shelves'
        # answers apart, and carries the same accepted risk: a
        # question beginning with that literal text would collide.
        unnarrowed = "" if file else _names_asked(db, question)
        key = f"within:{file}:{question}" if file else question
        found = remembered(db, key, limit, ttl) if ttl > 0 else None
        if found is not None:
            hits, about = found
        else:
            hits, about = _look(db, question, limit, file)
            if ttl > 0:
                remember(db, key, limit, hits, about)
        answer: dict[str, Any] = dict(about)
        if unnarrowed:
            answer["not_narrowed"] = (
                f"the question names {unnarrowed}, which is indexed, and "
                "this search was not kept inside it; pass file to do that"
            )
        if count is None:
            answer["trimmed"] = False
            return _trim(answer, hits, lambda _text: 0, budget)
        answer["trimmed"] = True
        answer["budget_tokens"] = budget
        return _trim(answer, hits, count, budget)
    finally:
        if own:
            db.close()


def _names_asked(db: sqlite3.Connection, question: str) -> str:
    """The one indexed document the question names, if it names one.

    A search of the whole index answers about whatever is nearest, and
    the question having named a file is not something the ranking
    knows. Measured: asked `did anything fail in counter.sta.rpt`, the
    first hit was build.log -- a broken make log indexed beside it --
    so the loop's first sight of the question was a pile of real error
    lines out of another file.

    Only when exactly one is named, and only ever as a note: a
    question may mention a document and still want the index. Saying
    which, rather than deciding for the caller.
    """
    low = question.lower()
    found = [
        str(row["path"])
        for row in db.execute("SELECT path FROM documents")
        if str(row["path"]).lower() in low
    ]
    return found[0] if len(found) == 1 else ""


def _look(
    db: sqlite3.Connection, question: str, limit: int, file: str = ""
) -> tuple[list[Hit], dict[str, Any]]:
    """Ask both halves, fuse, thin. The part worth not repeating."""
    ranked = {"lexical": _lexical(db, question, file)}
    note = ""
    try:
        ranked["semantic"] = _semantic(db, embed([question])[0], file)
    except IndexingError as error:
        note = f"the semantic half is missing: {error}"
    fused = fuse(ranked)
    vectors = _vectors(db, (entry[0] for entry in fused))
    hits = _rows(db, diversify(fused, vectors, limit))
    about: dict[str, Any] = {
        "searched": sorted(name for name, ids in ranked.items() if ids),
        "cached": False,
    }
    if file:
        about["within"] = file
        # What the document holds, beside what matched. The ranking
        # answers which section fits the question, and the question
        # the loop sends is its whole request: asked `From
        # top.flow.rpt pull these fields exactly as the report writes
        # them: Revision Name, ...` the prose put `Flow OS Summary`
        # ahead of `Flow Summary`, and the model read three wrong
        # sections without ever learning the right one existed.
        # `read_section` answers with this list when given a file
        # alone, and the model never called it that way; a search is
        # where it always looks. _trim fits the whole reply into the
        # budget, so the names are paid for out of the excerpts.
        about["sections"] = _names(db, file)
    if note:
        about["degraded"] = note
    return hits, about


def section(
    db: sqlite3.Connection,
    file: str,
    heading: str = "",
    page: int = 0,
) -> dict[str, Any]:
    """One stored section of a document, whole rather than excerpted.

    A search answers with a few hundred tokens of each hit, which is
    the right size for choosing where to look and the wrong one for
    reading a register table out of it. This is the other half: name
    a section a search cited and get what the index holds of it.

    It is a lookup, like `find_references`: no model, no embedding, no
    ranking. The index is the allowlist, so a document nobody indexed
    has no sections to read, and nothing here opens a file.
    """
    name = Path(file).name
    if not name:
        raise IndexingError("a section needs a document to read it from")
    if not heading.strip() and page <= 0:
        return _contents(db, name)
    rows = _sections_of(db, name, heading, page)
    if not rows:
        return {"file": name, "sections": [], "note": _nothing(db, name)}
    held = str(rows[0]["text"])
    answer: dict[str, Any] = {
        "file": name,
        "page": int(rows[0]["page"]),
        "section": str(rows[0]["heading"] or ""),
        "text": held[:READ_CHARS],
    }
    if len(held) > READ_CHARS:
        answer["text_cut_at"] = READ_CHARS
    if len(rows) > 1:
        answer["others"] = [
            f"page {row['page']}: {row['heading'] or 'no heading'}"
            for row in rows[1:MORE_SECTIONS]
        ]
    return answer


def _names(db: sqlite3.Connection, name: str) -> list[str]:
    """The document's section names, for a narrowed search to carry.

    A section holding almost nothing is not somewhere to read, and a
    report's table of contents is a run of them -- one line each,
    first in the file. Listing by order alone, the twelve names a
    report of sixty-nine sections could show were twelve lines of its
    contents page and not one place with anything in it.
    """
    rows = db.execute(
        "SELECT page, heading FROM chunks WHERE document = ?"
        " AND LENGTH(text) >= ? ORDER BY ordinal LIMIT ?",
        (name, WORTH_NAMING, IN_SEARCH + 1),
    ).fetchall()
    held = [
        str(row["heading"] or f"page {row['page']}")
        for row in rows[:IN_SEARCH]
    ]
    if len(rows) > IN_SEARCH:
        held.append(f"... and more than {IN_SEARCH}")
    return held


def _contents(db: sqlite3.Connection, name: str) -> dict[str, Any]:
    """What sections the document has, when none was named.

    A document's own table of contents, which a search cannot stand
    in for. Measured on a flow report: asked for the fields of its
    Flow Summary, the ranking put `Flow OS Summary` first and the
    loop never saw that `Flow Summary` existed -- it searched the same
    thing three times and reported every field as null. Returning the
    first section instead, which is a report's legal notice, was no
    use to anybody.
    """
    rows = list(
        db.execute(
            "SELECT page, heading FROM chunks WHERE document = ?"
            " ORDER BY ordinal LIMIT ?",
            (name, MAX_CONTENTS + 1),
        )
    )
    if not rows:
        return {"file": name, "sections": [], "note": _nothing(db, name)}
    answer: dict[str, Any] = {
        "file": name,
        "sections": [
            str(row["heading"] or f"page {row['page']}")
            for row in rows[:MAX_CONTENTS]
        ],
        "note": "name one of these to read it",
    }
    if len(rows) > MAX_CONTENTS:
        answer["more_than"] = MAX_CONTENTS
    return answer


def _sections_of(
    db: sqlite3.Connection, name: str, heading: str, page: int
) -> list[sqlite3.Row]:
    """The sections that answer to this heading or page, best first.

    The heading is tried whole, then as the front of one, because a
    search cites `3.2 Electrical characteristics` and a caller asks
    for `3.2`. A page alone is the fallback for a document whose
    sections were cut by page because it had no headings.
    """
    query = "SELECT page, heading, text FROM chunks WHERE document = ?"
    args: list[Any] = [name]
    if heading.strip():
        query += " AND (heading = ? OR heading LIKE ? ESCAPE '\\')"
        quoted = heading.strip().replace("\\", "\\\\")
        quoted = quoted.replace("%", "\\%").replace("_", "\\_")
        args += [heading.strip(), f"{quoted}%"]
    if page > 0:
        query += " AND page = ?"
        args.append(page)
    query += " ORDER BY ordinal LIMIT ?"
    args.append(MORE_SECTIONS + 1)
    return list(db.execute(query, args))


def _nothing(db: sqlite3.Connection, name: str) -> str:
    """Why there was no section, told apart from each other."""
    row = db.execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE document = ?", (name,)
    ).fetchone()
    if row is None or not int(row["n"]):
        return (
            f"{name} is not an indexed document; index_path reads one in "
            "and list_indexed says what is already there"
        )
    return (
        f"{name} is indexed in {row['n']} sections and none of them "
        "answers to that heading or page; search_docs cites them by the "
        "names they have"
    )


def indexed(db: sqlite3.Connection | None = None) -> dict[str, Any]:
    """Which documents are in the file, and how much of each."""
    own = db is None
    db = db or connect()
    try:
        rows = db.execute(
            "SELECT path, pages, chunks, ocr_pages, indexed_at"
            " FROM documents ORDER BY path"
        ).fetchall()
        return {
            "index": str(index_file()),
            "documents": [
                {
                    "file": str(row["path"]),
                    "sections": int(row["chunks"]),
                    "pages": int(row["pages"]),
                    "ocr_pages": int(row["ocr_pages"]),
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
