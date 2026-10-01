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

    def test_a_budget_the_citations_fill_drops_a_hit(self) -> None:
        answer = indexing._trim({"index": "c"}, HITS, half, 150)
        self.assertEqual(len(answer["hits"]), 1)
        self.assertEqual(answer["hits_dropped"], 1)

    def test_the_last_hit_is_kept_even_if_it_does_not_fit(self) -> None:
        answer = indexing._trim({"index": "c"}, HITS, half, 1)
        self.assertEqual(len(answer["hits"]), 1)
        self.assertEqual(len(answer["hits"][0]["text"]), indexing.MIN_EXCERPT)

    def test_nothing_is_dropped_when_everything_fits(self) -> None:
        answer = indexing._trim({"index": "c"}, HITS, half, 1000)
        self.assertEqual(len(answer["hits"]), 2)
        self.assertNotIn("hits_dropped", answer)

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

    def ask_in(
        self, question: str, file: str, limit: int = 3
    ) -> dict[str, Any]:
        with mock.patch.object(
            indexing, "embed", side_effect=self.embedding_for
        ):
            return indexing.search(question, limit, db=self.db, file=file)

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

    def test_the_same_question_twice_is_answered_from_the_file(
        self,
    ) -> None:
        self.feed(a_pdf=[Section(1, "1 Reset", "the reset is active low")])
        first = self.ask("reset")
        self.assertFalse(first["cached"])
        with mock.patch.object(
            indexing, "embed", side_effect=AssertionError("asked again")
        ):
            again = indexing.search("reset", 3, db=self.db)
        self.assertTrue(again["cached"])
        self.assertEqual(
            [h["page"] for h in again["hits"]],
            [h["page"] for h in first["hits"]],
        )

    def test_a_different_limit_is_a_different_question(self) -> None:
        self.feed(
            a_pdf=[
                Section(1, "1 Reset", "the reset is active low"),
                Section(2, "2 Timer", "the timer counts the reset"),
            ]
        )
        self.ask("reset", limit=1)
        self.assertFalse(self.ask("reset", limit=2)["cached"])

    def test_indexing_anything_drops_the_kept_answers(self) -> None:
        self.feed(a_pdf=[Section(1, "1 Reset", "the reset is active low")])
        self.ask("reset")
        self.feed(b_pdf=[Section(1, "1 Clock", "the clock is 50 MHz")])
        self.assertFalse(self.ask("reset")["cached"])

    def test_a_stale_answer_is_not_served(self) -> None:
        self.feed(a_pdf=[Section(1, "1 Reset", "the reset is active low")])
        self.ask("reset")
        with mock.patch.object(
            indexing, "embed", side_effect=self.embedding_for
        ):
            answer = indexing.search("reset", 3, db=self.db, ttl=-1.0)
        self.assertFalse(answer["cached"])

    def test_a_cached_answer_says_how_old_it_is(self) -> None:
        self.feed(a_pdf=[Section(1, "1 Reset", "the reset is active low")])
        self.ask("reset")
        again = self.ask("reset")
        self.assertIn("cached_age_seconds", again)
        self.assertGreaterEqual(again["cached_age_seconds"], 0.0)

    def test_a_degraded_answer_stays_marked_when_it_is_served_again(
        self,
    ) -> None:
        self.feed(a_pdf=[Section(1, "1 Reset", "the reset is active low")])
        with mock.patch.object(
            indexing, "embed", side_effect=IndexingError("ollama is down")
        ):
            first = indexing.search("reset", 3, db=self.db)
            again = indexing.search("reset", 3, db=self.db)
        self.assertIn("ollama is down", first["degraded"])
        self.assertTrue(again["cached"])
        self.assertIn("ollama is down", again["degraded"])

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


class SectionTest(IndexTest):
    """Reading one section whole, which a search deliberately does not."""

    def sheet(self) -> None:
        """Fed per test, not in setUp: the tests inherited from
        IndexTest assume an index with nothing in it."""
        self.feed(
            sheet_pdf=[
                Section(3, "3.1 Supply", "VDD is 3.3 V nominal."),
                Section(4, "3.2 Timing", "The reset is 10 clock cycles."),
                Section(5, "", "a page of numbers with no heading"),
            ]
        )

    def test_a_file_alone_answers_with_its_sections(self) -> None:
        """A document's own table of contents. Measured on a flow
        report: the ranking put `Flow OS Summary` ahead of `Flow
        Summary` and the loop never saw the second existed."""
        self.sheet()
        got = indexing.section(self.db, "sheet.pdf")
        self.assertEqual(
            got["sections"], ["3.1 Supply", "3.2 Timing", "page 5"]
        )
        self.assertNotIn("text", got)
        self.assertIn("name one of these", got["note"])

    def test_a_long_contents_says_it_was_cut(self) -> None:
        self.feed(
            many_md=[
                Section(1, f"{n} One", "x")
                for n in range(indexing.MAX_CONTENTS + 5)
            ]
        )
        got = indexing.section(self.db, "many.md")
        self.assertEqual(len(got["sections"]), indexing.MAX_CONTENTS)
        self.assertEqual(got["more_than"], indexing.MAX_CONTENTS)

    def test_a_document_nobody_indexed_has_no_contents(self) -> None:
        self.sheet()
        got = indexing.section(self.db, "other.pdf")
        self.assertEqual(got["sections"], [])
        self.assertIn("not an indexed document", got["note"])

    def test_a_section_comes_back_whole_rather_than_excerpted(self) -> None:
        self.sheet()
        got = indexing.section(self.db, "sheet.pdf", "3.1 Supply")
        self.assertEqual(got["text"], "VDD is 3.3 V nominal.")
        self.assertEqual(got["page"], 3)

    def test_the_front_of_a_heading_is_enough(self) -> None:
        """A caller asks for 3.2; a search cited the whole heading."""
        self.sheet()
        got = indexing.section(self.db, "sheet.pdf", "3.2")
        self.assertEqual(got["section"], "3.2 Timing")

    def test_a_page_serves_a_document_with_no_headings(self) -> None:
        self.sheet()
        got = indexing.section(self.db, "sheet.pdf", page=5)
        self.assertIn("page of numbers", got["text"])

    def test_a_path_is_taken_by_its_name(self) -> None:
        self.sheet()
        got = indexing.section(self.db, "/elsewhere/sheet.pdf", "3.1")
        self.assertEqual(got["page"], 3)

    def test_a_document_nobody_indexed_says_so(self) -> None:
        got = indexing.section(self.db, "other.pdf", "3.1")
        self.assertEqual(got["sections"], [])
        self.assertIn("not an indexed document", got["note"])

    def test_an_indexed_document_with_no_such_heading_says_that(
        self,
    ) -> None:
        """Told apart from the one above: the two need different fixes."""
        self.sheet()
        got = indexing.section(self.db, "sheet.pdf", "9 Appendix")
        self.assertIn("indexed in 3 sections", got["note"])

    def test_several_matches_name_the_others(self) -> None:
        self.feed(
            many_md=[
                Section(1, "3 One", "alpha"),
                Section(2, "3 Two", "beta"),
            ]
        )
        got = indexing.section(self.db, "many.md", "3 ")
        self.assertEqual(got["text"], "alpha")
        self.assertEqual(len(got["others"]), 1)
        self.assertIn("3 Two", got["others"][0])

    def test_a_long_section_is_cut_and_says_where(self) -> None:
        """What is stored and what is handed back are two caps.

        Giving them one name quietly shrank the stored one, and this
        test did not catch it because it compared a section against
        whatever the constant happened to say.
        """
        self.feed(big_md=[Section(1, "1 Long", "x" * 9000)])
        got = indexing.section(self.db, "big.md", "1 Long")
        self.assertEqual(len(got["text"]), indexing.READ_CHARS)
        self.assertEqual(got["text_cut_at"], indexing.READ_CHARS)
        stored = self.db.execute(
            "SELECT LENGTH(text) AS n FROM chunks WHERE document = 'big.md'"
        ).fetchone()
        self.assertGreater(int(stored["n"]), indexing.READ_CHARS)

    def test_a_wildcard_in_a_heading_is_text_not_a_pattern(self) -> None:
        self.sheet()
        got = indexing.section(self.db, "sheet.pdf", "3.%")
        self.assertEqual(got["sections"], [])

    def test_a_document_with_no_name_is_refused(self) -> None:
        with self.assertRaises(IndexingError):
            indexing.section(self.db, "")


class BoxedHeadingTest(unittest.TestCase):
    """A report titles its sections in a box, not with a number.

    Measured on a 631-line Quartus timing report: its table of
    contents went in as 39 headings of forty characters each, because
    those lines are numbered and the real titles are not, and then
    every table in the file landed in 23 sections all called
    "40. Timing Analyzer Messages". Recognising the box takes it to
    89 sections with the Fmax table under its own name.
    """

    REPORT = (
        "+------------------------+\n"
        "; Fmax Summary           ;\n"
        "+------------+-----------+\n"
        "; Fmax       ; Clock     ;\n"
        "+------------+-----------+\n"
        "; 154.23 MHz ; clk       ;\n"
        "+------------+-----------+\n"
        "+------------------------+\n"
        "; Setup Summary          ;\n"
        "+------------+-----------+\n"
        "; Slack      ; 1.234     ;\n"
        "+------------+-----------+\n"
        "+------------------------+\n"
        "; Hold Summary           ;\n"
        "+------------+-----------+\n"
        "; Slack      ; 0.321     ;\n"
        "+------------+-----------+\n"
    )

    def test_a_boxed_title_is_a_heading(self) -> None:
        sections, how = split(self.REPORT)
        self.assertEqual(how, "headings")
        self.assertEqual(
            [one.heading for one in sections],
            ["Fmax Summary", "Setup Summary", "Hold Summary"],
        )

    def test_the_table_goes_with_its_title(self) -> None:
        sections, _ = split(self.REPORT)
        self.assertIn("154.23 MHz", sections[0].text)
        self.assertNotIn("154.23 MHz", sections[1].text)

    def test_a_header_row_is_not_a_title(self) -> None:
        """One cell is what tells a title from a row of columns."""
        sections, _ = split(self.REPORT)
        self.assertNotIn("Fmax       ; Clock", [s.heading for s in sections])

    def test_a_cell_with_no_rule_above_it_is_not_a_title(self) -> None:
        text = "--- page 1\n; not a title ;\n; nor this ;\n; nor this ;\n"
        _, how = split(text)
        self.assertEqual(how, "pages")

    def test_an_empty_cell_is_not_a_title(self) -> None:
        text = "--- page 1\n" + "+-----+\n;     ;\n" * 4
        _, how = split(text)
        self.assertEqual(how, "pages")

    def test_markdown_is_left_alone(self) -> None:
        sections, how = split("## One\na\n\n## Two\nb\n\n## Three\nc\n")
        self.assertEqual(how, "headings")
        self.assertEqual(len(sections), 3)


class WithinTest(IndexTest):
    """A search narrowed to one document.

    Measured on five Quartus reports: asked for the fields of
    top.flow.rpt, the loop searched the whole index, read
    counter.flow.rpt's Flow Summary -- four of the five documents were
    counter's -- and reported counter's revision, entity, device and
    register count. Grounding passed, because those strings were in
    what the tools returned. It checks where a string came from, not
    which document was asked about.
    """

    def two(self) -> None:
        self.feed(
            one_rpt=[
                Section(1, "Flow Summary", "Revision Name counter reset"),
                Section(2, "Timing", "the timer is here"),
            ],
            two_rpt=[
                Section(1, "Flow Summary", "Revision Name top reset"),
            ],
        )

    def test_a_search_within_one_document_stays_there(self) -> None:
        self.two()
        got = self.ask_in("what is the revision name", "two.rpt")
        self.assertEqual({one["file"] for one in got["hits"]}, {"two.rpt"})
        self.assertEqual(got["within"], "two.rpt")

    def test_without_a_file_the_whole_index_is_searched(self) -> None:
        self.two()
        got = self.ask("what is the revision name")
        self.assertGreaterEqual(len({one["file"] for one in got["hits"]}), 2)
        self.assertNotIn("within", got)

    def test_a_path_is_taken_by_its_name(self) -> None:
        self.two()
        got = self.ask_in("revision", "/elsewhere/two.rpt")
        self.assertEqual({one["file"] for one in got["hits"]}, {"two.rpt"})

    def test_a_document_nobody_indexed_is_refused(self) -> None:
        self.two()
        with self.assertRaises(IndexingError) as caught:
            self.ask_in("revision", "three.rpt")
        self.assertIn("not an indexed document", str(caught.exception))

    def test_the_narrowed_answer_is_not_served_from_the_cache(self) -> None:
        """The kept answer is keyed on the question alone, so the
        narrowed ask would be given the answer about everything."""
        self.two()
        broad = self.ask("what is the revision name")
        self.assertFalse(broad["cached"])
        narrow = self.ask_in("what is the revision name", "two.rpt")
        self.assertFalse(narrow["cached"])
        self.assertEqual({one["file"] for one in narrow["hits"]}, {"two.rpt"})
        again = self.ask("what is the revision name")
        self.assertTrue(again["cached"])

    def test_the_semantic_half_ranks_the_document_it_was_given(
        self,
    ) -> None:
        """vec0 takes no condition on the document, so narrowing by
        cutting the top k of everything would lose a section that is
        nearest within its file and far down the index."""
        self.two()
        got = self.ask_in("timer", "one.rpt")
        self.assertIn("semantic", got["searched"])
        self.assertEqual(got["hits"][0]["section"], "Timing")


if __name__ == "__main__":
    unittest.main()
