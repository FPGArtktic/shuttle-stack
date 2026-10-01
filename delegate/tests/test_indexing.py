# SPDX-License-Identifier: GPL-3.0-only
"""Sectioning, citation and the token budget, without Qdrant."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from shuttle_delegate import indexing
from shuttle_delegate.indexing import IndexingError, Section, split

WITH_HEADINGS = (
    "--- page 1\n"
    "front matter\n\n"
    "2.1 Scope\n"
    "the scope is this\n\n"
    "--- page 2\n"
    "still the scope\n\n"
    "2.2 Terms\n"
    "a term\n\n"
    "--- page 3\n"
    "2.3 Registers\n"
    "the register map\n"
)
NO_HEADINGS = "--- page 1\njust prose\n\n--- page 2\nmore prose\n"

HITS = [
    {
        "file": "a.pdf",
        "page": 3,
        "section": "page 3",
        "score": 0.6,
        "text": "x" * 700,
    },
    {
        "file": "a.pdf",
        "page": 9,
        "section": "page 9",
        "score": 0.5,
        "text": "y" * 700,
    },
]


def half(text: str) -> int:
    """A tokeniser that charges half a token per character."""
    return max(1, len(text) // 2)


class SplitTest(unittest.TestCase):
    def test_headings_are_used_when_there_are_several(self) -> None:
        sections, how = split(WITH_HEADINGS)
        self.assertEqual(how, "headings")
        self.assertIn("2.1 Scope", [s.heading for s in sections])

    def test_pages_are_used_when_there_are_none(self) -> None:
        sections, how = split(NO_HEADINGS)
        self.assertEqual(how, "pages")
        self.assertEqual([s.page for s in sections], [1, 2])

    def test_one_match_is_a_footnote_not_a_structure(self) -> None:
        text = "--- page 1\nprose\n\n1. The package is plastic\nmore\n"
        _, how = split(text)
        self.assertEqual(how, "pages")

    def test_a_section_never_spans_pages(self) -> None:
        sections, _ = split(WITH_HEADINGS)
        carried = [s for s in sections if s.heading == "2.1 Scope"]
        self.assertEqual(sorted(s.page for s in carried), [1, 2])

    def test_the_preamble_keeps_the_page_it_starts_on(self) -> None:
        sections, _ = split(WITH_HEADINGS)
        self.assertEqual(sections[0].heading, "preamble")
        self.assertEqual(sections[0].page, 1)

    def test_markdown_headings_count(self) -> None:
        text = "## One\na\n\n## Two\nb\n\n## Three\nc\n"
        sections, how = split(text)
        self.assertEqual(how, "headings")
        self.assertEqual(
            [s.heading for s in sections], ["## One", "## Two", "## Three"]
        )

    def test_a_table_row_is_not_a_heading(self) -> None:
        rows = "".join(f"{n}      {n * 2}      ns\n" for n in range(10))
        _, how = split(f"--- page 1\n{rows}")
        self.assertEqual(how, "pages")

    def test_a_section_too_long_for_the_model_is_split(self) -> None:
        long = "word " * 4000
        sections, _ = split(f"--- page 1\n{long}")
        self.assertGreater(len(sections), 1)
        for section in sections:
            self.assertLessEqual(len(section.text), indexing.SECTION_CHARS)


class PointIdTest(unittest.TestCase):
    def test_the_same_section_keeps_the_same_id(self) -> None:
        section = Section(3, "page 3", "text")
        self.assertEqual(
            section.point_id("a.pdf", 0), section.point_id("a.pdf", 0)
        )

    def test_a_different_file_is_a_different_point(self) -> None:
        section = Section(3, "page 3", "text")
        self.assertNotEqual(
            section.point_id("a.pdf", 0), section.point_id("b.pdf", 0)
        )

    def test_a_different_page_is_a_different_point(self) -> None:
        self.assertNotEqual(
            Section(3, "h", "t").point_id("a.pdf", 0),
            Section(4, "h", "t").point_id("a.pdf", 0),
        )


class TrimTest(unittest.TestCase):
    def test_the_whole_answer_is_measured_not_the_hits_alone(self) -> None:
        answer = indexing._trim(
            {"collection": "c" * 200, "hits": HITS, "trimmed": True},
            half,
            300,
        )
        self.assertLessEqual(half(json.dumps(answer, ensure_ascii=False)), 300)

    def test_every_hit_is_shortened_by_the_same_amount(self) -> None:
        answer = indexing._trim(
            {"collection": "c", "hits": HITS, "trimmed": True}, half, 300
        )
        lengths = {len(hit["text"]) for hit in answer["hits"]}
        self.assertEqual(len(lengths), 1)

    def test_an_impossible_budget_stops_at_the_floor(self) -> None:
        answer = indexing._trim(
            {"collection": "c", "hits": HITS, "trimmed": True}, half, 1
        )
        for hit in answer["hits"]:
            self.assertEqual(len(hit["text"]), indexing.MIN_EXCERPT)


class RefusalTest(unittest.TestCase):
    def test_storing_nothing_is_an_error(self) -> None:
        with self.assertRaises(IndexingError):
            indexing.store("c", "a.pdf", [])

    def test_an_empty_question_is_refused(self) -> None:
        with self.assertRaises(IndexingError):
            indexing.search("   ")

    def test_searching_a_collection_that_is_not_there_says_so(self) -> None:
        with (
            mock.patch.object(indexing, "collections", return_value=[]),
            self.assertRaises(IndexingError) as caught,
        ):
            indexing.search("anything")
        self.assertIn("index_document", str(caught.exception))

    def test_an_existing_collection_is_not_remade(self) -> None:
        with (
            mock.patch.object(
                indexing, "collections", return_value=["shuttle_docs"]
            ),
            mock.patch.object(indexing, "_post") as post,
        ):
            self.assertFalse(indexing.ensure("shuttle_docs"))
        post.assert_not_called()

    def test_listing_a_collection_that_is_not_there_is_not_an_error(
        self,
    ) -> None:
        with mock.patch.object(indexing, "collections", return_value=[]):
            answer = indexing.indexed("absent")
        self.assertFalse(answer["exists"])
        self.assertEqual(answer["documents"], [])


if __name__ == "__main__":
    unittest.main()
