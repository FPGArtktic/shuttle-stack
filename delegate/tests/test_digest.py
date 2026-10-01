# SPDX-License-Identifier: GPL-3.0-only
"""Section labels and document labels, against a real index file."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from shuttle_delegate import digest, indexing
from shuttle_delegate.backend import Completion
from shuttle_delegate.indexing import DIMENSIONS, IndexingError, Section

LONG = "the register map, in detail, " * 12


class FakeServer:
    """Answers with a line, and remembers what it was shown."""

    role = "long"

    def __init__(self, answer: str = "what the section covers") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def context_size(self) -> int:
        return 8192

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 3)

    def chat(
        self,
        prompt: str,
        n_predict: int,
        temperature: float | None = None,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        self.prompts.append(prompt)
        return Completion(self.answer, tokens_in=20, tokens_out=8)


class LineTest(unittest.TestCase):
    def test_a_bullet_is_not_part_of_the_label(self) -> None:
        self.assertEqual(digest._one_line("- the scope"), "the scope")

    def test_quotes_are_not_part_of_it_either(self) -> None:
        self.assertEqual(digest._one_line('"the scope"'), "the scope")

    def test_several_lines_become_one(self) -> None:
        self.assertEqual(digest._one_line("the\n scope\n"), "the scope")


class DigestTest(unittest.TestCase):
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
        self.server = FakeServer()

    def embedding_for(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * (DIMENSIONS - 1) for _ in texts]

    def feed(self, *sections: Section, file: str = "a.pdf") -> None:
        with mock.patch.object(
            indexing, "embed", side_effect=self.embedding_for
        ):
            indexing.store(self.db, file, list(sections))

    def test_every_long_section_gets_a_line(self) -> None:
        self.feed(Section(1, "1 Scope", LONG), Section(2, "2 Registers", LONG))
        done = digest.sections(self.db, self.server, "a.pdf")
        self.assertEqual(done.sections, 2)
        self.assertEqual(done.calls, 2)

    def test_a_short_section_is_its_own_summary(self) -> None:
        self.feed(Section(1, "1 Scope", "short"))
        done = digest.sections(self.db, self.server, "a.pdf")
        self.assertEqual((done.sections, done.skipped), (0, 1))
        self.assertEqual(self.server.prompts, [])

    def test_a_second_pass_costs_nothing(self) -> None:
        self.feed(Section(1, "1 Scope", LONG))
        digest.sections(self.db, self.server, "a.pdf")
        again = digest.sections(self.db, self.server, "a.pdf")
        self.assertEqual(again.calls, 0)

    def test_a_second_pass_is_not_an_unindexed_file(self) -> None:
        """What the nightly sweep does every night after the first."""
        self.feed(Section(1, "1 Scope", LONG))
        digest.whole(self.db, self.server, "a.pdf")
        again = digest.whole(self.db, self.server, "a.pdf")
        self.assertEqual(again.sections, 0)
        self.assertEqual(
            digest.of_documents(self.db)["a.pdf"], "what the section covers"
        )

    def test_asking_again_redoes_it(self) -> None:
        self.feed(Section(1, "1 Scope", LONG))
        digest.sections(self.db, self.server, "a.pdf")
        again = digest.sections(self.db, self.server, "a.pdf", again=True)
        self.assertEqual(again.calls, 1)

    def test_the_document_line_is_built_from_the_section_lines(self) -> None:
        self.feed(Section(1, "1 Scope", LONG), Section(2, "2 Registers", LONG))
        done = digest.whole(self.db, self.server, "a.pdf")
        self.assertEqual(done.document, "what the section covers")
        self.assertIn("1 Scope", self.server.prompts[-1])
        self.assertIn("2 Registers", self.server.prompts[-1])
        self.assertEqual(digest.of_documents(self.db)["a.pdf"], done.document)

    def test_a_document_with_no_lines_gets_none_of_its_own(self) -> None:
        self.feed(Section(1, "1 Scope", "short"))
        done = digest.whole(self.db, self.server, "a.pdf")
        self.assertEqual(done.document, "")
        self.assertEqual(digest.of_documents(self.db), {})

    def test_a_file_that_is_not_indexed_says_what_to_call(self) -> None:
        with self.assertRaises(IndexingError) as caught:
            digest.sections(self.db, self.server, "absent.pdf")
        self.assertIn("index_path", str(caught.exception))

    def search(self, question: str, budget: int = 300) -> dict[str, Any]:
        """Search with a tokeniser, so the trimming really runs."""
        with mock.patch.object(
            indexing, "embed", side_effect=self.embedding_for
        ):
            return indexing.search(
                question,
                3,
                lambda text: max(1, len(text) // 3),
                budget,
                db=self.db,
                ttl=0.0,
            )

    def test_the_line_stands_in_when_there_is_no_room_for_the_text(
        self,
    ) -> None:
        self.feed(Section(1, "1 Scope", LONG))
        digest.whole(self.db, self.server, "a.pdf")
        answer = self.search("register", budget=90)
        self.assertEqual(
            answer["hits"][0]["summary"], "what the section covers"
        )
        self.assertNotIn("text", answer["hits"][0])
        self.assertEqual(answer["summaries"], digest.CAVEAT)
        self.assertIn("no room", answer["text_left_out"])

    def test_the_text_is_preferred_while_it_fits(self) -> None:
        self.feed(Section(1, "1 Scope", LONG))
        digest.whole(self.db, self.server, "a.pdf")
        answer = self.search("register")
        self.assertIn("text", answer["hits"][0])
        self.assertNotIn("summary", answer["hits"][0])
        self.assertNotIn("summaries", answer)

    def test_a_hit_without_a_line_carries_no_caveat(self) -> None:
        self.feed(Section(1, "1 Scope", LONG))
        answer = self.search("register", budget=90)
        self.assertNotIn("summary", answer["hits"][0])
        self.assertNotIn("summaries", answer)

    def test_reindexing_throws_the_lines_away(self) -> None:
        """A line describes sections that no longer exist."""
        self.feed(Section(1, "1 Scope", LONG))
        digest.whole(self.db, self.server, "a.pdf")
        self.feed(Section(1, "1 Scope", LONG + "and now something else"))
        held = self.db.execute("SELECT COUNT(*) FROM digests").fetchone()[0]
        self.assertEqual(held, 0)
        self.assertEqual(digest.of_documents(self.db), {})

    def test_the_caveat_says_the_lines_are_not_citations(self) -> None:
        self.assertIn("unverified", digest.CAVEAT)
        self.assertIn("quoted", digest.CAVEAT)


if __name__ == "__main__":
    unittest.main()
