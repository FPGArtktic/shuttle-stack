# SPDX-License-Identifier: GPL-3.0-only
"""The overnight pass: index what is new, and leave a report.

A document dropped into a watched directory in the evening should be
searchable with citations in the morning, and the morning should start
with a page saying what happened rather than with a log to read.

The pass is idempotent. What was indexed is remembered by path, size
and modification time, so a directory of four hundred datasheets costs
nothing on the second night and a document that was edited is read
again. Nothing is deleted and nothing outside the index is written.
"""

from __future__ import annotations

import json
import time
import tomllib
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

from . import code, digest, indexing
from .backend import Server
from .config import config_home
from .documents import SUFFIXES as DOCUMENT_SUFFIXES
from .indexing import Count
from .runs import home

NAME = "sweep.toml"
STATE = "sweep.json"
REPORT = "report.md"
TEXT_SUFFIXES = (".md", ".txt", ".rst")
SUFFIXES = (*DOCUMENT_SUFFIXES, *TEXT_SUFFIXES)
# Directories a sweep has no business reading. A build tree holds
# generated source by the thousand and indexing it buries the file
# somebody wrote under the file a tool wrote.
SKIP = frozenset(
    {
        ".cache",
        ".git",
        ".shuttle",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "db",
        "incremental_db",
        "node_modules",
        "output_files",
        "simulation",
    }
)
REPORT_TOKENS = 1000
MAX_LISTED = 20


class SweepError(RuntimeError):
    """The pass cannot be made as configured."""


@dataclass(frozen=True)
class Settings:
    """Where to look and what to do with what is found."""

    roots: tuple[Path, ...]
    heading: str = indexing.HEADING
    # Digesting a document costs about as long again as indexing it
    # and buys a readable line for every section, which is worth
    # having overnight and not worth waiting for by hand.
    digest: bool = True


@dataclass
class Done:
    """One file, and how it went."""

    path: str
    sections: int = 0
    digested: int = 0
    note: str = ""
    seconds: float = 0.0
    how: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass
class Report:
    """A night's work."""

    started: str
    seconds: float = 0.0
    indexed: list[Done] = field(default_factory=list)
    failed: list[Done] = field(default_factory=list)
    skipped: int = 0


def settings_path() -> Path:
    return config_home() / "shuttle" / NAME


def settings(file: Path | None = None) -> Settings:
    """What to sweep, from the file the operator writes.

    There is no default root. A pass that guessed where documents live
    would read directories nobody offered it.
    """
    file = file or settings_path()
    if not file.is_file():
        raise SweepError(
            f"{file} is missing; it names the directories to sweep, as "
            'in roots = ["~/datasheets"]'
        )
    try:
        written = tomllib.loads(file.read_text())
    except tomllib.TOMLDecodeError as error:
        raise SweepError(f"{file}: {error}") from error
    roots = written.get("roots") or []
    if not isinstance(roots, list) or not roots:
        raise SweepError(f"{file}: roots is empty, so there is nothing to do")
    found = []
    for root in roots:
        where = Path(str(root)).expanduser()
        if not where.is_dir():
            raise SweepError(f"{file}: {where} is not a directory")
        found.append(where)
    return Settings(
        roots=tuple(found),
        heading=str(written.get("heading", indexing.HEADING)),
        digest=bool(written.get("digest", True)),
    )


def wanted(path: Path) -> bool:
    """A file worth indexing, and not one under a build directory."""
    if SKIP & set(path.parts):
        return False
    return path.suffix.lower() in SUFFIXES or code.is_source(path)


def _server() -> Server:
    """The long server, for the digesting.

    Read here rather than passed in: a sweep is started by a timer
    with no caller to hand one over, and the profile that writes a
    section's label is not a choice worth a configuration field.
    """
    from .backend import Backend
    from .config import load_endpoints

    return Backend(load_endpoints()["long"])


def documents(roots: tuple[Path, ...]) -> list[Path]:
    """Every document and source file under the roots, in order."""
    found: set[Path] = set()
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and wanted(path):
                found.add(path.resolve())
    return sorted(found)


def fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{int(stat.st_mtime)}"


def state_path() -> Path:
    return home() / STATE


def load_state() -> dict[str, str]:
    path = state_path()
    if not path.is_file():
        return {}
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError, ValueError):
        # A state file that cannot be read means the pass does the work
        # again, which is slow and correct.
        return {}


def save_state(state: dict[str, str]) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, indent=1, sort_keys=True), encoding="utf-8"
    )


def one(path: Path, chosen: Settings) -> Done:
    """Index one file, reporting rather than raising.

    Source and prose go to their own shelves. The sweep does not need
    to know which; it asks the same question of every file it found
    and the answer says what it was.
    """
    from . import tasks

    started = time.monotonic()
    try:
        if code.is_source(path):
            cut = code.units(str(path))
            with closing(code.connect()) as db:
                stored = code.store(
                    db,
                    path.name,
                    cut,
                    max((u.last_line for u in cut.units), default=0),
                    str(path.resolve()),
                )
            return Done(
                str(path),
                sections=stored,
                seconds=round(time.monotonic() - started, 1),
                how=f"{cut.language or 'source'} by {cut.how}",
            )
        text = tasks.read_text(str(path))
        sections, how = indexing.split(text, chosen.heading)
        with closing(indexing.connect()) as db:
            stored = indexing.store(db, path.name, sections)
    except Exception as error:  # noqa: BLE001 - one bad file is not the night
        return Done(
            str(path),
            seconds=round(time.monotonic() - started, 1),
            error=f"{type(error).__name__}: {error}",
        )
    done = Done(
        str(path),
        sections=stored,
        seconds=round(time.monotonic() - started, 1),
        how=how,
    )
    if chosen.digest:
        _digest(path, done)
    done.seconds = round(time.monotonic() - started, 1)
    return done


def _digest(path: Path, done: Done) -> None:
    """Give the sections their lines, or say why they have none.

    Separate from the indexing, and failing separately. The document
    is in the index by the time this runs: a server that is down or a
    sweep that runs before the stack has started should leave a
    searchable document without a label, not a file the morning
    report calls failed and the next pass indexes all over again.
    """
    try:
        with closing(indexing.connect()) as db:
            done.digested = digest.whole(
                db, _server(), Path(path).name
            ).sections
    except Exception as error:  # noqa: BLE001 - a label is not the file
        done.note = f"not digested: {type(error).__name__}: {error}"


def run(chosen: Settings | None = None, force: bool = False) -> Report:
    """Index everything new or changed, and say what happened.

    One document failing does not end the night: the rest are indexed
    and the failure is in the report, because a pass that stops at the
    first unreadable file leaves the operator with neither.
    """
    chosen = chosen or settings()
    state = {} if force else load_state()
    started = time.monotonic()
    report = Report(started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    for path in documents(chosen.roots):
        try:
            mark = fingerprint(path)
        except OSError as error:
            report.failed.append(Done(str(path), error=str(error)))
            continue
        if state.get(str(path)) == mark:
            report.skipped += 1
            continue
        done = one(path, chosen)
        if done.ok:
            state[str(path)] = mark
            report.indexed.append(done)
        else:
            report.failed.append(done)
    report.seconds = round(time.monotonic() - started, 1)
    save_state(state)
    return report


def _lines(report: Report) -> list[str]:
    out = [
        f"# Sweep {report.started}",
        "",
        f"- indexed: {len(report.indexed)}",
        f"- failed: {len(report.failed)}",
        f"- unchanged: {report.skipped}",
        f"- took: {report.seconds:.0f}s",
        "",
    ]
    if report.failed:
        out += ["## Failed", ""]
        for done in report.failed[:MAX_LISTED]:
            out.append(f"- `{Path(done.path).name}`: {done.error}")
        if len(report.failed) > MAX_LISTED:
            out.append(f"- and {len(report.failed) - MAX_LISTED} more")
        out.append("")
    if report.indexed:
        out += ["## Indexed", ""]
        for done in report.indexed[:MAX_LISTED]:
            out.append(
                f"- `{Path(done.path).name}`: {done.sections} sections "
                f"by {done.how}, {done.seconds:.0f}s"
            )
        if len(report.indexed) > MAX_LISTED:
            out.append(f"- and {len(report.indexed) - MAX_LISTED} more")
        out.append("")
    return out


def write_report(report: Report, count: Count | None = None) -> Path:
    """Leave the report where the morning can find it.

    Failures are listed before successes, and the list is cut before
    the whole report is, so what went wrong survives the trimming. The
    cap is the same thousand tokens every other answer here is held to.
    """
    from .runs import cap

    body = "\n".join(_lines(report))
    if count is not None:
        body, _ = cap(count, body, REPORT_TOKENS)
    path = home() / REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def main() -> int:
    """Run the pass and say where the report is."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="shuttle-sweep",
        description="index new documents and leave a report",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="index everything again, not only what changed",
    )
    parser.add_argument(
        "--settings", default="", help=f"a {NAME} other than the default"
    )
    args = parser.parse_args()
    chosen = settings(Path(args.settings) if args.settings else None)
    report = run(chosen, force=args.force)
    count = None
    try:
        from .server import backend

        count = backend("extract").count_tokens
    except Exception:  # noqa: BLE001 - the report matters more than its size
        count = None
    where = write_report(report, count)
    print(
        f"indexed {len(report.indexed)}, failed {len(report.failed)}, "
        f"unchanged {report.skipped} in {report.seconds:.0f}s -> {where}"
    )
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
