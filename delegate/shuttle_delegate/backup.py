# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mateusz Okulanis
"""One archive holding the index, the sessions and the audit log.

The index being a single file is the reason it can be backed up at
all, and the reason it cannot be backed up with `cp`. A copy taken
while something is writing is a torn database: SQLite's own backup
walks the pages under a read lock and produces a file that opens,
which a byte copy of a live file does not reliably do.

The transcripts are the other thing worth keeping. A session's KV dump
is not: it is 305 MiB of cache that the transcript can rebuild, and
backing it up would make the archive almost entirely the one thing in
it that is derived.

    shuttle-backup [DIRECTORY]
"""

from __future__ import annotations

import sqlite3
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from . import audit, indexing
from .runs import home

# Reported rather than compressed away: an index is mostly float32
# vectors and they do not compress, so gzip costs seconds to save
# almost nothing. The archive is a tar so that one member can be
# pulled out without unpacking the rest.
SUFFIX = ".tar"


class BackupError(RuntimeError):
    """The archive cannot be written, and why."""


@dataclass(frozen=True)
class Made:
    """One archive, and what went into it."""

    path: Path
    index_bytes: int
    sessions: int
    logs: int

    def report(self) -> str:
        return "\n".join(
            [
                f"archive   {self.path}",
                f"size      {self.path.stat().st_size / 1e6:.1f} MB",
                f"index     {self.index_bytes / 1e6:.1f} MB, "
                "copied through SQLite rather than byte for byte",
                f"sessions  {self.sessions} transcript(s)",
                f"audit     {self.logs} file(s)",
            ]
        )


def snapshot(source: Path, into: Path) -> int:
    """A consistent copy of the index, whatever is writing to it."""
    if not source.is_file():
        raise BackupError(f"{source}: no index to back up")
    try:
        with (
            sqlite3.connect(f"file:{source}?mode=ro", uri=True) as live,
            sqlite3.connect(into) as copy,
        ):
            live.backup(copy)
    except sqlite3.Error as error:
        raise BackupError(f"{source}: {error}") from error
    return into.stat().st_size


def make(where: Path | None = None) -> Made:
    """Write the archive, and say what is in it."""
    index = indexing.index_file()
    sessions = sorted((home() / "sessions").glob("*.jsonl"))
    logs = audit.held()
    target = (where or Path.cwd()).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    name = f"shuttle-{time.strftime('%Y%m%d-%H%M%S')}{SUFFIX}"
    archive = target / name
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(scratch) / index.name
        size = snapshot(index, copy)
        try:
            with tarfile.open(archive, "w") as tar:
                tar.add(copy, arcname=index.name)
                for one in sessions:
                    tar.add(one, arcname=f"sessions/{one.name}")
                for one in logs:
                    tar.add(one, arcname=f"audit/{one.name}")
        except OSError as error:
            raise BackupError(f"{archive}: {error}") from error
    return Made(archive, size, len(sessions), len(logs))


def main() -> int:
    """Write an archive into the directory named, or the current one."""
    where = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    try:
        print(make(where).report())
    except BackupError as error:
        print(f"backup: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
