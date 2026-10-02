# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mateusz Okulanis
"""Text out of a PDF, through a container that can do nothing else.

The extraction runs in `shuttle-docs` with `--network=none` and only
the document's own directory mounted read only. A document is data and
never a program: a PDF that tries something cannot reach the network,
cannot write, and cannot see another directory.

The text layer is used where there is one and OCR only for the pages
without, which is the difference between a datasheet read in a third
of a second and one read in a minute.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

IMAGE = os.environ.get("SHUTTLE_DOCS_IMAGE", "localhost/shuttle-docs")
MOUNT = "/docs"
TIMEOUT = 1800.0
MARKER = "--- page {page}"
SUFFIXES = (".pdf",)
# A dot leader is typography, not content: it is how a printed page
# walks the eye from a parameter to its value. pdftotext reproduces
# every dot, which costs tokens and puts them inside the cell, so an
# extracted value came back as "E (PDIP) Package . . . . . . 67oC/W".
# Two spaces take their place, which is what -layout uses between
# columns anyway. Three dots at least, so an ellipsis in a sentence
# and a version number are left alone.
LEADER = re.compile(r"[ \t]*(?:\.[ \t]){3,}\.?[ \t]*")


class DocumentError(RuntimeError):
    """The document cannot be read, and why."""


@dataclass(frozen=True)
class Page:
    """One page, and whether it had to be looked at rather than read."""

    page: int
    ocr: bool
    text: str


def is_document(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUFFIXES


def _podman() -> str:
    found = shutil.which("podman")
    if not found:
        raise DocumentError(
            "podman is not on the path, and the extraction runs in a "
            "container so that a document cannot reach anything"
        )
    return found


def in_container(file: Path, verb: str, *rest: str) -> list[str]:
    """What to run to have the container do one thing to one file.

    The verb comes from this module and not from a caller: the image
    takes it from a closed set, and a path that arrived as a verb is
    the one mistake worth making impossible.
    """
    return [
        _podman(),
        "run",
        "--rm",
        "--network=none",
        "--read-only",
        "-v",
        f"{file.parent}:{MOUNT}:ro",
        IMAGE,
        verb,
        f"{MOUNT}/{file.name}",
        *rest,
    ]


def _command(file: Path, first: int, last: int) -> list[str]:
    pages = [str(first), str(last)] if last else []
    return in_container(file, "extract", *pages)


def pages(path: str, first: int = 1, last: int = 0) -> list[Page]:
    """Every page of the range, read or OCRed."""
    file = Path(path).expanduser().resolve()
    if not file.is_file():
        raise DocumentError(f"{file}: not a readable file")
    if first < 1:
        raise DocumentError(f"the first page is 1, not {first}")
    if last and last < first:
        raise DocumentError(f"pages {first} to {last} is backwards")
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell
            _command(file, first, last),
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise DocumentError(
            f"{file.name}: still extracting after {TIMEOUT:.0f}s; ask "
            "for a page range"
        ) from error
    if done.returncode != 0:
        reason = (done.stderr or done.stdout).strip()[:300]
        raise DocumentError(f"{file.name}: {reason or 'extraction failed'}")
    try:
        answer = json.loads(done.stdout)
    except json.JSONDecodeError as error:
        raise DocumentError(
            f"{file.name}: the extractor did not answer with JSON: "
            f"{done.stdout[:200]}"
        ) from error
    return [
        Page(int(p["page"]), bool(p["ocr"]), str(p["text"]))
        for p in answer.get("pages", [])
    ]


def tidy(text: str) -> str:
    """The page as printed, less the typography."""
    return "\n".join(
        LEADER.sub("  ", line).rstrip() for line in text.splitlines()
    )


def as_text(read: list[Page]) -> str:
    """The pages as one text, each marked with its page number.

    The marker is what lets an answer drawn from the document name the
    page it came from, which is the whole point of reading a standard
    rather than guessing about one.
    """
    return "\n\n".join(
        f"{MARKER.format(page=page.page)}\n{tidy(page.text).strip()}"
        for page in read
        if page.text.strip()
    )


def read(path: str, first: int = 1, last: int = 0) -> str:
    """A document as text, with its page markers."""
    found = pages(path, first, last)
    if not found:
        raise DocumentError(f"{Path(path).name}: no pages in that range")
    text = as_text(found)
    if not text.strip():
        raise DocumentError(
            f"{Path(path).name}: pages {first} onwards hold no text, "
            "read or OCRed"
        )
    return text
