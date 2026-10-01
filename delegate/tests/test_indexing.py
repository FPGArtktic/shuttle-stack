# SPDX-License-Identifier: GPL-3.0-only
"""Sectioning, citation, fusion and the token budget.

The index is a file, so these run against a real one in a temporary
directory. Only the embedding call is replaced: it is the one part
that needs a model on the other end of a socket.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from shuttle_delegate import indexing
from shuttle_delegate.indexing import (
    DIMENSIONS,
    Hit,
    IndexingError,
    Section,
    diversify,
    fuse,
    split,
    terms,
)

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
    Hit("a.pdf", 3, "", "page 3", "x" * 700, 0.6, "both"),
    Hit("a.pdf", 9, "", "page 9", "y" * 700, 0.5, "lexical"),
]


def half(text: str) -> int:
    """A tokeniser that charges half a token per character."""
    return max(1, len(text) // 2)


def vector(seed: float, *, axis: int = 0) -> list[float]:
    """A unit vector along one axis, scaled, so cosines are obvious."""
    out = [0.0] * DIMENSIONS
    out[axis] = seed
    return out


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


class ClauseTest(unittest.TestCase):
    def test_a_numbered_heading_yields_its_number(self) -> None:
        self.assertEqual(Section(1, "4.2.1 Registers", "t").clause, "4.2.1")

    def test_a_markdown_heading_has_none(self) -> None:
        self.assertEqual(Section(1, "## Scope", "t").clause, "")

    def test_a_page_heading_has_none(self) -> None:
        self.assertEqual(Section(1, "page 1", "t").clause, "")


class TermsTest(unittest.TestCase):
    def test_a_clause_number_survives_as_a_phrase(self) -> None:
        self.assertIn('"4.2.1"', terms("what does 4.2.1 say"))

    def test_a_register_name_is_kept_whole(self) -> None:
        self.assertIn('"TIMER0"', terms("the TIMER0 register"))

    def test_punctuation_cannot_become_syntax(self) -> None:
        self.assertEqual(terms("(reset) OR *"), '"reset" OR "OR"')

    def test_a_one_letter_word_is_not_a_term(self) -> None:
        self.assertEqual(terms("a b reset"), '"reset"')

    def test_a_question_of_punctuation_alone_asks_nothing(self) -> None:
        self.assertEqual(terms("?? - ."), "")

    def test_a_repeated_word_is_asked_for_once(self) -> None:
        self.assertEqual(terms("reset the reset"), '"reset" OR "the"')


class FuseTest(unittest.TestCase):
    def test_a_row_in_both_lists_outranks_one_in_either(self) -> None:
        out = fuse({"lexical": [1, 2], "semantic": [3, 1]})
        self.assertEqual(out[0][0], 1)
        self.assertEqual(out[0][2], "both")

    def test_a_row_names_the_list_that_found_it(self) -> None:
        out = {row: how for row, _, how in fuse({"lexical": [7]})}
        self.assertEqual(out[7], "lexical")

    def test_an_empty_list_contributes_nothing(self) -> None:
        self.assertEqual(fuse({"lexical": [], "semantic": []}), [])

    def test_the_order_is_stable_on_a_tie(self) -> None:
        out = fuse({"lexical": [5], "semantic": [2]})
        self.assertEqual([row for row, _, _ in out], [2, 5])


class DiversifyTest(unittest.TestCase):
    def test_a_second_copy_of_the_first_hit_loses_to_a_new_one(self) -> None:
        ordered = [(1, 0.9, "both"), (2, 0.89, "both"), (3, 0.5, "lexical")]
        vectors = {
            1: vector(1.0),
            2: vector(1.0),
            3: vector(1.0, axis=1),
        }
        picked = [row for row, _, _ in diversify(ordered, vectors, 2)]
        self.assertEqual(picked, [1, 3])

    def test_relevance_still_wins_when_nothing_repeats(self) -> None:
        ordered = [(1, 0.9, "both"), (2, 0.8, "both")]
        vectors = {1: vector(1.0), 2: vector(1.0, axis=1)}
        picked = [row for row, _, _ in diversify(ordered, vectors, 2)]
        self.assertEqual(picked, [1, 2])

    def test_a_hit_without_an_embedding_is_kept(self) -> None:
        ordered = [(1, 0.9, "both"), (2, 0.5, "lexical")]
        picked = [row for row, _, _ in diversify(ordered, {1: vector(1.0)}, 2)]
        self.assertEqual(picked, [1, 2])

    def test_asking_for_none_returns_none(self) -> None:
        self.assertEqual(diversify([(1, 0.9, "both")], {}, 0), [])


class TrimTest(unittest.TestCase):
    def test_the_whole_answer_is_measured_not_the_hits_alone(self) -> None:
        answer = indexing._trim(
            {"index": "c" * 200, "trimmed": True}, HITS, half, 300
        )
        self.assertLessEqual(half(json.dumps(answer, ensure_ascii=False)), 300)

    def test_every_hit_is_shortened_by_the_same_amount(self) -> None:
        answer = indexing._trim({"index": "c"}, HITS, half, 300)
        lengths = {len(hit["text"]) for hit in answer["hits"]}
        self.assertEqual(len(lengths), 1)

    def test_an_impossible_budget_stops_at_the_floor(self) -> None:
        answer = indexing._trim({"index": "c"}, HITS, half, 1)
        for hit in answer["hits"]:
            self.assertEqual(len(hit["text"]), indexing.MIN_EXCERPT)

    def test_a_clause_is_reported_only_when_there_is_one(self) -> None:
        with_clause = Hit("a.pdf", 1, "4.2", "4.2 Scope", "t", 0.1, "both")
        self.assertEqual(with_clause.report(10)["clause"], "4.2")
        self.assertNotIn("clause", HITS[0].report(10))


class IndexTest(unittest.TestCase):
    """The file itself: writing, replacing, searching, listing."""

    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "documents.sqlite"
        patch = mock.patch.object(
            indexing, "index_file", return_value=self.path
        )
        patch.start()
        self.addCleanup(patch.stop)
        self.db = indexing.connect()
        self.addCleanup(self.db.close)

    def embedding_for(self, texts: list[str]) -> list[list[float]]:
        """One axis per word of interest, so distance is predictable."""
        words = ("reset", "timer", "clock")
        out = []
        for text in texts:
            low = text.lower()
            made = [0.0] * DIMENSIONS
            for at, word in enumerate(words):
                made[at] = 1.0 if word in low else 0.0
            if not any(made):
                made[len(words)] = 1.0
            out.append(made)
        return out

    def feed(self, **files: list[Section]) -> None:
        with mock.patch.object(
            indexing, "embed", side_effect=self.embedding_for
        ):
            for name, sections in files.items():
                indexing.store(self.db, name.replace("_", "."), sections)

    def ask(self, question: str, limit: int = 3) -> dict[str, Any]:
        with mock.patch.object(
            indexing, "embed", side_effect=self.embedding_for
        ):
            return indexing.search(question, limit, db=self.db)

    def test_the_schema_is_the_one_weft_writes(self) -> None:
        names = {
            row[0]
            for row in self.db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        self.assertLessEqual({"documents", "chunks", "vec_chunks"}, names)

    def test_a_document_is_found_by_the_words_it_uses(self) -> None:
        self.feed(
            a_pdf=[
                Section(1, "1 Reset", "the reset is active low"),
                Section(2, "2 Timer", "the timer counts up"),
            ]
        )
        answer = self.ask("reset", limit=1)
        self.assertEqual(answer["hits"][0]["section"], "1 Reset")
        self.assertEqual(answer["hits"][0]["page"], 1)

    def test_a_clause_number_is_found_lexically(self) -> None:
        self.feed(
            a_pdf=[
                Section(7, "4.2.1 Pin assignment", "the pins are these"),
                Section(8, "4.2.2 Timing", "the timing is this"),
            ]
        )
        answer = self.ask("4.2.1", limit=1)
        self.assertEqual(answer["hits"][0]["clause"], "4.2.1")
        # The lexical half on its own, because the point of keeping
        # the clause as a phrase is that BM25 can hit it.  Whether
        # fusion then labels the row "lexical" or "both" depends on
        # what the vectors happened to rank, which is not the claim.
        first = indexing._lexical(self.db, "4.2.1")[0]
        clause = self.db.execute(
            "SELECT clause FROM chunks WHERE id = ?", (first,)
        ).fetchone()[0]
        self.assertEqual(clause, "4.2.1")

    def test_both_halves_are_asked(self) -> None:
        self.feed(a_pdf=[Section(1, "1 Reset", "the reset is active low")])
        self.assertEqual(
            self.ask("reset")["searched"], ["lexical", "semantic"]
        )

    def test_indexing_again_replaces_rather_than_doubles(self) -> None:
        self.feed(a_pdf=[Section(1, "1 Reset", "first pass")])
        self.feed(a_pdf=[Section(1, "1 Reset", "second pass")])
        rows = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()
        self.assertEqual(rows[0], 1)
        held = self.db.execute("SELECT text FROM chunks").fetchone()
        self.assertEqual(held[0], "second pass")

    def test_the_vectors_go_with_the_sections_they_replace(self) -> None:
        self.feed(
            a_pdf=[Section(1, "1 Reset", "first"), Section(2, "2 T", "b")]
        )
        self.feed(a_pdf=[Section(1, "1 Reset", "only one now")])
        chunks = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        vectors = self.db.execute(
            "SELECT COUNT(*) FROM vec_chunks"
        ).fetchone()[0]
        self.assertEqual((chunks, vectors), (1, 1))

    def test_the_lexical_index_follows_a_deletion(self) -> None:
        self.feed(a_pdf=[Section(1, "1 Reset", "a unique word: frobnicate")])
        self.feed(a_pdf=[Section(1, "1 Reset", "nothing like that now")])
        found = self.db.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?",
            ('"frobnicate"',),
        ).fetchone()[0]
        self.assertEqual(found, 0)

    def test_the_listing_names_every_document(self) -> None:
        self.feed(
            a_pdf=[Section(1, "1 Reset", "a")],
            b_pdf=[Section(3, "1 Timer", "b"), Section(4, "2 Clock", "c")],
        )
        listed = indexing.indexed(self.db)["documents"]
        self.assertEqual([d["file"] for d in listed], ["a.pdf", "b.pdf"])
        self.assertEqual([d["sections"] for d in listed], [1, 2])
        self.assertEqual([d["pages"] for d in listed], [1, 4])

    def test_a_search_with_no_documents_says_what_to_call(self) -> None:
        with self.assertRaises(IndexingError) as caught:
            self.ask("anything")
        self.assertIn("index_path", str(caught.exception))

    def test_an_empty_question_is_refused(self) -> None:
        with self.assertRaises(IndexingError):
            indexing.search("   ", db=self.db)

    def test_storing_nothing_is_an_error(self) -> None:
        with self.assertRaises(IndexingError):
            indexing.store(self.db, "a.pdf", [])

    def test_a_failed_embedding_leaves_nothing_behind(self) -> None:
        self.feed(a_pdf=[Section(1, "1 Reset", "the one good pass")])
        with (
            mock.patch.object(
                indexing, "embed", side_effect=IndexingError("ollama is down")
            ),
            self.assertRaises(IndexingError),
        ):
            indexing.store(self.db, "a.pdf", [Section(1, "1 R", "replaced")])
        held = self.db.execute("SELECT text FROM chunks").fetchone()
        self.assertEqual(held[0], "the one good pass")

    def test_a_missing_embedder_still_answers_and_says_so(self) -> None:
        self.feed(a_pdf=[Section(1, "1 Reset", "the reset is active low")])
        with mock.patch.object(
            indexing, "embed", side_effect=IndexingError("ollama is down")
        ):
            answer = indexing.search("reset", db=self.db)
        self.assertEqual(answer["searched"], ["lexical"])
        self.assertIn("ollama is down", answer["degraded"])
        self.assertEqual(len(answer["hits"]), 1)

    def test_a_wrong_sized_vector_is_refused(self) -> None:
        with (
            mock.patch.object(indexing, "embed", return_value=[[0.1, 0.2]]),
            self.assertRaises(IndexingError) as caught,
        ):
            indexing.store(self.db, "a.pdf", [Section(1, "1 R", "t")])
        self.assertIn("1024", str(caught.exception))

    def test_an_existing_file_without_the_lexical_index_gains_one(
        self,
    ) -> None:
        """What a file written by WEFT alone looks like."""
        self.db.close()
        plain = sqlite3.connect(self.path)
        plain.executescript(indexing.SCHEMA)
        plain.execute(
            "INSERT INTO chunks (document, ordinal, page, clause, heading,"
            " text) VALUES ('w.pdf', 0, 5, '3.1', '3.1 Scope', 'frobnicate')"
        )
        plain.execute("DROP TABLE chunks_fts")
        plain.commit()
        plain.close()
        self.db = indexing.connect()
        self.addCleanup(self.db.close)
        found = self.db.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?",
            ('"frobnicate"',),
        ).fetchall()
        self.assertEqual(len(found), 1)


if __name__ == "__main__":
    unittest.main()
