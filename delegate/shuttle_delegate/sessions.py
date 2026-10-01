# SPDX-License-Identifier: GPL-3.0-only
"""A document read once and asked about many times.

The transcript is the source of truth and the KV dump is only a cache.
If the dump is missing the document is read again and the answers are
the same, only slower; if the file has changed underneath it the dump
is thrown away rather than trusted, because a cached answer to a
question about an older version of a file is worse than a slow one.

Measured across a restart of the server, restoring a dump takes a
tenth of a second and the next question answers in 6.7 seconds against
11.5 without it.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .backend import Backend
from .grounding import check
from .retrieval import narrow
from .runs import home, new_id

PREFIX = """\
TEXT:
{text}

---
Answer the question below using only the text above. Answer in at most
{words} words. Quote the words from the text that support the answer.

If the text does not contain the answer, reply with exactly this and
nothing else: NOT IN THIS TEXT
Do not answer from anything you know outside the text.

"""
Resolve = Callable[[str], Backend]
MISSING = "NOT IN THIS TEXT"
DUMP = "{session}.bin"
# An id reaches both a file name and the server's slot-save path, so it
# is a plain name or it is refused.
NAME = re.compile(r"^[0-9]{8}-[0-9]{6}-session-[0-9a-f]{6}$")


class SessionError(RuntimeError):
    """No such session, or one that can no longer be honoured."""


@dataclass(frozen=True)
class Opened:
    """What a session was opened on."""

    id: str
    source: str
    words: int
    profile: str
    prefix: str
    fingerprint: str

    def dump(self) -> str:
        return DUMP.format(session=self.id)


def directory() -> Path:
    return home() / "sessions"


def checked(session: str) -> str:
    """The id, or an error: it becomes a path and a slot-save name."""
    if not NAME.match(session):
        raise SessionError(
            f"{session!r} is not a session id; they look like "
            "20261001-161539-session-74018c"
        )
    return session


def transcript(session: str) -> Path:
    return directory() / f"{checked(session)}.jsonl"


def fingerprint(prefix: str) -> str:
    return hashlib.sha256(prefix.encode()).hexdigest()[:32]


def _append(session: str, entry: dict) -> None:
    path = transcript(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _lines(session: str) -> list[dict]:
    path = transcript(session)
    if not path.is_file():
        raise SessionError(
            f"no session {session!r}; session_open starts one and the "
            "transcript under .shuttle/sessions keeps it"
        )
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_prefix(
    path: str,
    words: int,
    pattern: str = "",
    until: str = "",
    context: int = 12,
) -> str:
    """The part of every prompt in this session that does not change."""
    file = Path(path).expanduser()
    if not file.is_file():
        raise SessionError(f"{file}: not a readable file")
    text = file.read_text(errors="replace")
    if pattern:
        text, matched = narrow(text, pattern, context, until)
        if not matched:
            raise SessionError(f"nothing in {file} matches {pattern!r}")
    if not text.strip():
        raise SessionError(f"{file}: holds no text to read")
    return PREFIX.format(text=text, words=words)


def _header(session: str) -> dict:
    head = _lines(session)[0]
    if head.get("kind") != "open":
        raise SessionError(f"session {session!r} has no opening line")
    return head


def _latest_fingerprint(session: str) -> str:
    """The fingerprint of the last version of the document loaded."""
    seen = ""
    for line in _lines(session):
        if line.get("fingerprint"):
            seen = line["fingerprint"]
    return seen


def _must_fit(server: Backend, opened: Opened) -> int:
    """Refuse a document the server cannot hold, and say by how much.

    llama-server answers an oversized prompt with an HTTP 400, which
    reaches the caller as a wall of JSON. A session is opened once and
    asked many times, so the check belongs at the opening.
    """
    needed = server.count_tokens(opened.prefix)
    room = server.context_size() - max(96, opened.words * 3)
    if needed > room:
        raise SessionError(
            f"the document is {needed} tokens and shuttle-"
            f"{server.role} has room for {room} once {opened.words} "
            "words are reserved for each answer; narrow it with a "
            "pattern, ask for a shorter answer, or reinstall with a "
            "larger --ctx"
        )
    return needed


def _prime(server: Backend, opened: Opened) -> int:
    """Read the document once and keep its cache for the next question."""
    _must_fit(server, opened)
    server.chat(opened.prefix + "QUESTION: are you ready?", 1)
    return server.slot_save(opened.dump())


def open_session(
    server: Backend,
    profile: str,
    path: str,
    words: int = 200,
    pattern: str = "",
    until: str = "",
    context: int = 12,
) -> dict:
    """Read a document and keep its cache under a name.

    The profile is recorded, because every later question has to reach
    the same server: a dump written to one role's cache directory is
    invisible from the other's.
    """
    prefix = build_prefix(path, words, pattern, until, context)
    session = new_id("session")
    source = str(Path(path).expanduser())
    opened = Opened(
        session, source, words, profile, prefix, fingerprint(prefix)
    )
    # The transcript is written before the cache is made. A crash
    # between the two leaves a session with no dump, which ask()
    # rebuilds; the other order leaves 305 MiB nothing refers to.
    _must_fit(server, opened)
    _append(
        session,
        {
            "kind": "open",
            "session": session,
            "source": source,
            "words": words,
            "pattern": pattern,
            "until": until,
            "context": context,
            "profile": profile,
            "role": server.role,
            "fingerprint": opened.fingerprint,
        },
    )
    tokens = _prime(server, opened)
    _append(session, {"kind": "cached", "tokens": tokens})
    return {
        "session": session,
        "source": source,
        "tokens": tokens,
        "profile": profile,
        "cached": server.cached(opened.dump()),
        "server": f"shuttle-{server.role}",
    }


def _reopen(session: str) -> Opened:
    """The session as it stands, with the document read again."""
    head = _header(session)
    prefix = build_prefix(
        head["source"],
        head["words"],
        head.get("pattern", ""),
        head.get("until", ""),
        head.get("context", 12),
    )
    return Opened(
        session,
        head["source"],
        head["words"],
        head["profile"],
        prefix,
        _latest_fingerprint(session),
    )


def ask(resolve: Resolve, session: str, question: str) -> dict:
    """Ask the open document, reusing its cache where it still fits.

    The server comes from the session rather than from the caller. A
    dump lives in one role's cache directory and is invisible from the
    other's, so answering a session on a different server would read
    the document again every time while reporting a cache hit of zero.
    """
    if not question.strip():
        raise SessionError("a session needs a question to answer")
    opened = _reopen(session)
    server = resolve(opened.profile)
    current = fingerprint(opened.prefix)
    changed = current != opened.fingerprint
    restored = 0
    if changed:
        server.forget(opened.dump())
    elif server.cached(opened.dump()):
        restored = server.slot_restore(opened.dump())
    answer = server.chat(
        opened.prefix + f"QUESTION: {question.strip()}",
        max(96, opened.words * 3),
    )
    found = MISSING not in answer.content.upper()
    report = {
        "session": session,
        "answer": answer.content if found else "",
        "found": found,
        "restored_tokens": restored,
        "source_changed": changed,
        "local_tokens": answer.tokens,
        "truncated": answer.truncated,
        "server": f"shuttle-{server.role}",
    } | check(answer.content, opened.prefix).report()
    if not found:
        report["note"] = "the document does not answer this question"
    _append(session, {"kind": "ask", "question": question.strip()} | report)
    if changed:
        # The answer is already recorded and returned. Rebuilding the
        # cache is an optimisation for the next question, so a failure
        # here is noted rather than raised: the alternative throws away
        # an answer the caller has paid for.
        try:
            tokens = _prime(server, opened)
        except Exception as error:  # noqa: BLE001 - noted, not raised
            _append(session, {"kind": "stale", "reason": str(error)})
            report["cache_rebuilt"] = False
        else:
            _append(
                session,
                {"kind": "reopen", "fingerprint": current, "tokens": tokens},
            )
            report["cache_rebuilt"] = True
    return report


def close(resolve: Resolve, session: str) -> dict:
    """Drop the cache and leave the transcript.

    On the session's own server: forgetting a dump on the other one
    unlinks nothing and reports success, leaving 305 MiB behind with
    the caller told it is gone.
    """
    head = _header(session)
    server = resolve(head.get("profile", "long"))
    server.forget(DUMP.format(session=checked(session)))
    asked = sum(1 for line in _lines(session) if line.get("kind") == "ask")
    _append(session, {"kind": "close"})
    return {
        "session": session,
        "source": head["source"],
        "questions": asked,
        "transcript": str(transcript(session)),
        "note": "the cache is gone; the transcript is kept",
    }
