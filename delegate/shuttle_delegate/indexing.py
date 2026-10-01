# SPDX-License-Identifier: GPL-3.0-only
"""Documents cut into sections, embedded, and searched by meaning.

The point is a citation. A section is stored with the file and the page
it came from, so an answer can name where it was read rather than
asking anyone to take its word. The caller gets a few hundred tokens
and three facts about each of them.

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
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .chunking import chunk
from .documents import MARKER

QDRANT = os.environ.get("SHUTTLE_QDRANT_URL", "http://127.0.0.1:6333")
OLLAMA = os.environ.get("SHUTTLE_EMBED_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("SHUTTLE_EMBED_MODEL", "bge-m3")
COLLECTION = os.environ.get("SHUTTLE_COLLECTION", "shuttle_docs")

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
TIMEOUT = 300.0
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
PAGE = re.compile(re.escape(MARKER).replace(r"\{page\}", r"(\d+)"))
Count = Callable[[str], int]
NAMESPACE = uuid.UUID("5f6b2a1e-0000-5000-8000-000000000000")


class IndexingError(RuntimeError):
    """The index cannot be read or written, and why."""


@dataclass(frozen=True)
class Section:
    """One citable piece of a document."""

    page: int
    heading: str
    text: str

    def point_id(self, file: str, order: int) -> str:
        return str(uuid.uuid5(NAMESPACE, f"{file}|{self.page}|{order}"))


def _post(
    url: str, body: dict[str, Any] | None, method: str = "POST"
) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
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
        answer = _post(
            f"{OLLAMA}/api/embed",
            {
                "model": MODEL,
                "input": texts[start : start + BATCH],
                "options": {"num_gpu": 0},
            },
        )
        vectors = answer.get("embeddings") or []
        if len(vectors) != len(texts[start : start + BATCH]):
            raise IndexingError(
                f"{MODEL} returned {len(vectors)} embeddings for "
                f"{len(texts[start : start + BATCH])} inputs"
            )
        out.extend(vectors)
    return out


def split(text: str, heading: str = HEADING) -> tuple[list[Section], str]:
    """The document's sections, and how they were found.

    A heading pattern that matches nothing falls back to one section
    per page, and the answer says which happened: a caller told it
    indexed sections should be able to find out it indexed pages.
    """
    marker = re.compile(heading)
    lines = text.splitlines()
    if sum(bool(marker.match(line)) for line in lines) < MIN_HEADINGS:
        return _within_limit(_by_page(text)), "pages"

    found: list[Section] = []
    page = 0
    at_page, heading_text = 0, "preamble"
    held: list[str] = []

    def close() -> None:
        if held:
            found.append(Section(at_page, heading_text, "\n".join(held)))

    for line in lines:
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
        if marker.match(line):
            close()
            held = []
            at_page, heading_text = page, line.strip()[:120]
        held.append(line)
    close()
    return _within_limit(found), "headings"


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


def collections() -> list[str]:
    answer = _post(f"{QDRANT}/collections", None, "GET")
    return [c["name"] for c in answer.get("result", {}).get("collections", [])]


def ensure(collection: str) -> bool:
    """Make the collection if it is not there; say whether it was made.

    Only the collection named is touched. Qdrant may be shared with
    other work — this machine's own index sits beside ours — and a tool
    that reaches into a collection it did not make has no business
    doing so.
    """
    if collection in collections():
        return False
    _post(
        f"{QDRANT}/collections/{collection}",
        {"vectors": {"size": DIMENSIONS, "distance": "Cosine"}},
        "PUT",
    )
    return True


def store(collection: str, file: str, sections: list[Section]) -> int:
    """Embed the sections and put them in, replacing an earlier pass.

    The point id is derived from the file, the page and the order, so
    indexing a document twice replaces its sections rather than
    doubling them.
    """
    if not sections:
        raise IndexingError(f"{file}: no sections to index")
    vectors = embed([section.text for section in sections])
    points = [
        {
            "id": section.point_id(file, order),
            "vector": vector,
            "payload": {
                "file": file,
                "page": section.page,
                "section": section.heading,
                "text": section.text,
            },
        }
        for order, (section, vector) in enumerate(
            zip(sections, vectors, strict=True)
        )
    ]
    _post(
        f"{QDRANT}/collections/{collection}/points", {"points": points}, "PUT"
    )
    return len(points)


def _trim(answer: dict[str, Any], count: Count, budget: int) -> dict[str, Any]:
    """Shorten the excerpts until the whole answer fits the budget.

    The whole answer, not the hits alone: the collection name and the
    fields around them cost tokens too, and measuring only the hits
    returned 315 against a budget of 300.

    Evenly across the hits, because a long first one crowding out the
    rest would hide the second-best section behind the best.
    """
    room = EXCERPT_CHARS
    while True:
        shortened = answer | {
            "hits": [
                hit | {"text": hit["text"][:room]} for hit in answer["hits"]
            ]
        }
        if count(json.dumps(shortened, ensure_ascii=False)) <= budget:
            return shortened
        if room <= MIN_EXCERPT:
            return shortened
        room = max(MIN_EXCERPT, room * 2 // 3)


def search(
    question: str,
    collection: str = COLLECTION,
    limit: int = 3,
    count: Count | None = None,
    budget: int = SEARCH_TOKENS,
) -> dict[str, Any]:
    """The sections closest to the question, with where each came from.

    With a tokeniser the excerpts are trimmed to the budget; without
    one they come back at their full length and the caller is told, so
    nothing quietly claims to have been measured.
    """
    if not question.strip():
        raise IndexingError("a search needs something to search for")
    if collection not in collections():
        raise IndexingError(
            f"no collection {collection!r}; index_document makes one"
        )
    vector = embed([question])[0]
    answer = _post(
        f"{QDRANT}/collections/{collection}/points/query",
        {"query": vector, "limit": limit, "with_payload": True},
    )
    hits = [
        {
            "file": hit["payload"].get("file", ""),
            "page": hit["payload"].get("page", 0),
            "section": hit["payload"].get("section", ""),
            "score": round(float(hit.get("score", 0.0)), 3),
            "text": hit["payload"].get("text", "")[:EXCERPT_CHARS],
        }
        for hit in answer.get("result", {}).get("points", [])
    ]
    if count is None:
        return {"collection": collection, "hits": hits, "trimmed": False}
    return _trim(
        {
            "collection": collection,
            "hits": hits,
            "trimmed": True,
            "budget_tokens": budget,
        },
        count,
        budget,
    )


def indexed(collection: str = COLLECTION) -> dict[str, Any]:
    """Which documents are in the collection, and how much of each."""
    if collection not in collections():
        return {"collection": collection, "documents": [], "exists": False}
    counts: dict[str, int] = {}
    pages: dict[str, int] = {}
    offset = None
    while True:
        body: dict[str, Any] = {"limit": 256, "with_payload": ["file", "page"]}
        if offset is not None:
            body["offset"] = offset
        answer = _post(
            f"{QDRANT}/collections/{collection}/points/scroll", body
        )
        result = answer.get("result", {})
        for point in result.get("points", []):
            name = point.get("payload", {}).get("file", "")
            page = int(point.get("payload", {}).get("page", 0))
            counts[name] = counts.get(name, 0) + 1
            pages[name] = max(pages.get(name, 0), page)
        offset = result.get("next_page_offset")
        if offset is None:
            break
    return {
        "collection": collection,
        "exists": True,
        "documents": [
            {"file": name, "sections": counts[name], "pages": pages[name]}
            for name in sorted(counts)
        ],
    }
