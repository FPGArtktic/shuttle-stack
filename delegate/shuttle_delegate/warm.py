# SPDX-License-Identifier: GPL-3.0-only
"""The documents a night leaves in a server's cache.

Reading a document is the expensive part of asking about it, and a
session is the only thing here that pays that once. A morning question
about a document nobody holds a session for pays it again, and a night
has the hours to have paid it already.

Which documents is not a guess. The audit log records every call with
the path it was about, so the documents asked about most are the ones
worth holding, and a document nobody has asked about is not warmed on
the chance that somebody will.

What it costs is the limit. A dump is 305 MiB, sized by the server's
context rather than by the document, so this is a disk decision before
it is a speed one. The pass lets go of what it was holding before it
holds anything new, and a document re-indexed tonight is read again:
a cache about the text as it was is worse than no cache at all.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from . import audit, sessions
from .backend import Cache
from .sessions import Resolve

STATE = "warm.json"
# Tools whose call was about a document worth holding. A search is not
# one: it reads the index rather than the file, so holding the file it
# cited would save nothing.
READERS = frozenset(
    {
        "ask_file",
        "classify_file",
        "extract",
        "session_open",
        "summarize_file",
    }
)
# How many documents to hold, and how many to count before choosing
# them. The second is larger so that a document whose file has gone
# away does not cost the night a slot.
KEEP = 2
CONSIDER = 12
# How many words a held session reserves for an answer. The same
# default `session_open` has: a warmed session is one that was opened
# early, not a different kind of thing.
WORDS = 200


@dataclass(frozen=True)
class Warmed:
    """One document held in a cache, and what holding it cost."""

    path: str
    session: str = ""
    tokens: int = 0
    seconds: float = 0.0
    asked: int = 0
    kept: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def _path_in(line: str) -> str:
    """The document one log line was about, if it was about one.

    Only a call that succeeded counts. A path that failed every time
    is a path that is wrong, and holding it would fail again.
    """
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        return ""
    if not isinstance(entry, dict) or not entry.get("ok"):
        return ""
    if entry.get("tool") not in READERS:
        return ""
    args = entry.get("args")
    if not isinstance(args, dict):
        return ""
    return str(args.get("path", "")).strip()


def asked(limit: int = CONSIDER) -> list[tuple[str, int]]:
    """The documents the log says were asked about, most first."""
    counted: dict[str, int] = {}
    for one in audit.held():
        try:
            text = one.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            where = _path_in(line)
            if where:
                counted[where] = counted.get(where, 0) + 1
    ordered = sorted(counted.items(), key=lambda one: (-one[1], one[0]))
    return ordered[:limit]


def state_path() -> Path:
    from .runs import home

    return home() / STATE


def load_state() -> dict[str, str]:
    """Which session holds each document, from the last pass."""
    path = state_path()
    if not path.is_file():
        return {}
    try:
        held = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        # Unreadable state means the pass opens the sessions again,
        # which is slow and correct. The dumps it cannot name now are
        # the one thing this leaks, and they are disk, not data.
        return {}
    if not isinstance(held, dict):
        return {}
    return {str(key): str(value) for key, value in held.items()}


def save_state(state: dict[str, str]) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, indent=1, sort_keys=True), encoding="utf-8"
    )


def _hold(server: Cache, profile: str, path: str, times: int) -> Warmed:
    """Open a session over one document, reporting rather than raising."""
    began = time.monotonic()
    try:
        made = sessions.open_session(server, profile, path, WORDS)
    except Exception as error:  # noqa: BLE001 - one file is not the night
        return Warmed(
            path,
            asked=times,
            seconds=round(time.monotonic() - began, 1),
            error=f"{type(error).__name__}: {error}",
        )
    return Warmed(
        path,
        session=str(made["session"]),
        tokens=int(made["tokens"]),
        seconds=round(time.monotonic() - began, 1),
        asked=times,
    )


def _let_go(resolve: Resolve, session: str) -> None:
    """Drop a dump nobody is going to ask for.

    A dump left behind is disk and not data, so every failure here is
    suppressed: the night has already done the work that matters.
    """
    with contextlib.suppress(Exception):
        sessions.close(resolve, session)


def refresh(
    resolve: Resolve,
    profile: str = "long",
    keep: int = KEEP,
    changed: frozenset[str] = frozenset(),
) -> list[Warmed]:
    """Hold the most-asked documents, and let go of the rest.

    A session that already holds a wanted document is kept rather than
    reopened, so the id the morning report names stays the same from
    one night to the next. A document re-indexed tonight is not kept:
    its cache is about the text as it was.
    """
    if keep < 1:
        return []
    counted = dict(asked())
    want = [
        path
        for path, _ in sorted(
            counted.items(), key=lambda one: (-one[1], one[0])
        )
        if Path(path).expanduser().is_file()
    ][:keep]
    held = load_state()
    for path, session in list(held.items()):
        if path in want and path not in changed:
            continue
        _let_go(resolve, session)
        del held[path]
    out: list[Warmed] = []
    for path in want:
        if path in held:
            out.append(
                Warmed(path, held[path], asked=counted[path], kept=True)
            )
            continue
        made = _hold(resolve(profile), profile, path, counted[path])
        if made.ok:
            held[path] = made.session
        out.append(made)
    save_state(held)
    return out


def lines(warmed: list[Warmed]) -> list[str]:
    """The held documents, as the morning report puts them."""
    if not warmed:
        return []
    out = ["## Cached", ""]
    for one in warmed:
        name = Path(one.path).name
        if one.error:
            out.append(f"- `{name}`: not cached, {one.error}")
        elif one.kept:
            out.append(f"- `{name}`: still `{one.session}`")
        else:
            out.append(
                f"- `{name}`: `{one.session}`, {one.tokens} tokens in "
                f"{one.seconds:.0f}s, asked {one.asked}x"
            )
    out += [
        "",
        "Ask these with session_ask rather than reading them again.",
        "",
    ]
    return out
