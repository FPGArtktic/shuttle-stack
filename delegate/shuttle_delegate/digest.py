# SPDX-License-Identifier: GPL-3.0-only
"""What a document and its sections are about, written down once.

A search hit from a datasheet is often unreadable: the page is a
table, and six hundred characters of it are numbers and units with
nothing saying what they are numbers of. A line naming what the
section holds costs one call overnight and nothing at query time,
which is the whole point of digesting ahead of the question.

A digest is a label, never a citation. Nothing verifies it — a
summary is not a span of the source, so the grounding check that
makes an answer trustworthy has nothing to compare. Every reply that
carries one says so. The verbatim text travels beside it and is what
may be quoted.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Any

from .backend import Server
from .indexing import IndexingError

# One line, not a paragraph. A label long enough to read in a list of
# three hits and short enough that three of them do not eat the token
# budget the search answer is held to.
SECTION_WORDS = 20
DOCUMENT_WORDS = 60
# Below this a section is its own best summary and asking costs a call
# for nothing.
MIN_SECTION = 240
# Said once per reply rather than once per hit, because it is a
# property of every summary here and not of any one of them.
CAVEAT = (
    "summaries are unverified labels written by the local model; the "
    "text beside them is what may be quoted"
)

SECTION_PROMPT = (
    "{text}\n\n---\n"
    "The text above is one section of a document. In at most {words} "
    "words, say what it is about. Name the things it covers rather "
    "than describing them. No preamble, one line."
)
DOCUMENT_PROMPT = (
    "{text}\n\n---\n"
    "The lines above each describe one section of a single document. "
    "In at most {words} words, say what the document as a whole is "
    "and what it covers. No preamble."
)


@dataclass
class Digested:
    """What one pass over a document cost and produced."""

    file: str
    sections: int = 0
    skipped: int = 0
    calls: int = 0
    tokens: int = 0
    document: str = ""

    def report(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "sections_digested": self.sections,
            "sections_short_enough_already": self.skipped,
            "local_calls": self.calls,
            "local_tokens": self.tokens,
            "document_summary": self.document,
            "note": CAVEAT,
        }


def _one_line(text: str) -> str:
    """The model's answer as a label: one line, no bullet, no quotes."""
    first = " ".join(text.split())
    return first.lstrip("-*• ").strip().strip('"')


def sections(
    db: sqlite3.Connection, server: Server, file: str, again: bool = False
) -> Digested:
    """Give every long section of the file a line of its own.

    A section already digested is left alone unless asked for again,
    so a second pass over a watched directory costs nothing and an
    interrupted one resumes where it stopped.
    """
    # Whether the file is in the index at all is a different question
    # from whether anything is left to do about it. Asking only the
    # second conflated a file nobody has indexed with one already
    # digested, and the nightly pass over a watched directory is the
    # second case every night after the first.
    if not db.execute(
        "SELECT 1 FROM chunks WHERE document = ? LIMIT 1", (file,)
    ).fetchone():
        raise IndexingError(
            f"{file}: nothing to digest; index_path reads a file in"
        )
    rows = db.execute(
        "SELECT c.id, c.heading, c.text FROM chunks c"
        " LEFT JOIN digests d ON d.chunk_id = c.id"
        " WHERE c.document = ?"
        + ("" if again else " AND d.chunk_id IS NULL")
        + " ORDER BY c.ordinal",
        (file,),
    ).fetchall()
    done = Digested(file)
    for row in rows:
        body = str(row["text"])
        if len(body) < MIN_SECTION:
            done.skipped += 1
            continue
        answer = server.chat(
            SECTION_PROMPT.format(text=body, words=SECTION_WORDS),
            n_predict=SECTION_WORDS * 4,
        )
        done.calls += 1
        done.tokens += answer.tokens
        line = _one_line(answer.content)
        if not line:
            continue
        db.execute(
            "INSERT OR REPLACE INTO digests (chunk_id, summary, made_at)"
            " VALUES (?, ?, ?)",
            (int(row["id"]), line, time.time()),
        )
        done.sections += 1
    db.commit()
    return done


def document(
    db: sqlite3.Connection, server: Server, file: str, done: Digested
) -> Digested:
    """Fold the section lines into one about the whole document.

    Built from the lines rather than from the document: the folding
    `summarise` does would read the file again, and the lines are
    already the shape a summary of a summary wants.
    """
    lines = [
        f"- {row['heading']}: {row['summary']}"
        for row in db.execute(
            "SELECT c.heading, d.summary FROM digests d"
            " JOIN chunks c ON c.id = d.chunk_id"
            " WHERE c.document = ? ORDER BY c.ordinal",
            (file,),
        )
    ]
    if not lines:
        return done
    answer = server.chat(
        DOCUMENT_PROMPT.format(
            text="\n".join(lines)[: server.context_size() * 2],
            words=DOCUMENT_WORDS,
        ),
        n_predict=DOCUMENT_WORDS * 4,
    )
    done.calls += 1
    done.tokens += answer.tokens
    done.document = _one_line(answer.content)
    if done.document:
        db.execute(
            "INSERT OR REPLACE INTO document_digests"
            " (path, summary, sections, made_at) VALUES (?, ?, ?, ?)",
            (file, done.document, len(lines), time.time()),
        )
        db.commit()
    return done


def whole(
    db: sqlite3.Connection, server: Server, file: str, again: bool = False
) -> Digested:
    """Both passes over one document."""
    return document(db, server, file, sections(db, server, file, again))


def summaries(db: sqlite3.Connection, ids: list[int]) -> dict[int, str]:
    """The lines for these sections, where there are any."""
    if not ids:
        return {}
    places = ",".join("?" * len(ids))
    return {
        int(row["chunk_id"]): str(row["summary"])
        for row in db.execute(
            f"SELECT chunk_id, summary FROM digests"
            f" WHERE chunk_id IN ({places})",
            ids,
        )
    }


def of_documents(db: sqlite3.Connection) -> dict[str, str]:
    """What each digested document is about."""
    return {
        str(row["path"]): str(row["summary"])
        for row in db.execute("SELECT path, summary FROM document_digests")
    }
