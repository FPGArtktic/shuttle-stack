# SPDX-License-Identifier: GPL-3.0-only
"""The answer cache: what it serves, and what it refuses to.

The embedding is replaced by one that puts a vector on an axis per
keyword, so distance is arithmetic rather than a model's opinion. What
is being tested is the policy, not bge-m3.
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from shuttle_delegate import answers, indexing
from shuttle_delegate.answers import Kept
from shuttle_delegate.indexing import DIMENSIONS, IndexingError

VERIFIED = {
    "answer": "the maximum junction temperature is 150oC",
    "found": True,
    "verified": True,
    "local_calls": 3,
    "local_tokens": 1800,
    "quotes": 1,
    "quotes_grounded": 1,
}
UNVERIFIED = VERIFIED | {"verified": False, "quotes_not_in_source": ["x"]}


def along(axis: int, tilt: float = 0.0) -> list[float]:
    """A vector mostly on one axis, tilted towards the next."""
    out = [0.0] * DIMENSIONS
    out[axis] = 1.0
    out[axis + 1] = tilt
    return out


class KeptTest(unittest.TestCase):
    def test_a_served_answer_costs_nothing_and_says_so(self) -> None:
        report = Kept("how hot", VERIFIED, 0.02, 12.0).report()
        self.assertTrue(report["from_cache"])
        self.assertEqual(report["local_calls"], 0)
        self.assertEqual(report["local_tokens"], 0)

    def test_it_names_the_question_it_was_stored_under(self) -> None:
        report = Kept("how hot can it get", VERIFIED, 0.02, 1.0).report()
        self.assertEqual(report["asked_as"], "how hot can it get")
        self.assertEqual(report["distance"], 0.02)

    def test_the_answer_itself_comes_through(self) -> None:
        report = Kept("q", VERIFIED, 0.0, 0.0).report()
        self.assertEqual(report["answer"], VERIFIED["answer"])
        self.assertTrue(report["verified"])


class CacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "documents.sqlite"
        patch = mock.patch.object(
            indexing, "index_file", return_value=self.path
        )
        patch.start()
        self.addCleanup(patch.stop)
        self.file = Path(self.dir.name) / "datasheet.pdf"
        self.file.write_bytes(b"%PDF-1.4\n")
        self.db = answers.connect()
        self.addCleanup(self.db.close)
        self.vectors: dict[str, list[float]] = {}

    def embedding_for(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            if text not in self.vectors:
                self.vectors[text] = along(len(self.vectors) * 2)
            out.append(self.vectors[text])
        return out

    def place(self, question: str, vector: list[float]) -> None:
        self.vectors[question] = vector

    def store(self, question: str, reply: dict[str, Any] = VERIFIED) -> None:
        with mock.patch.object(
            indexing, "embed", side_effect=self.embedding_for
        ):
            answers.remember(self.db, str(self.file), question, reply)

    def ask(self, question: str, **rest: Any) -> Kept | None:
        with mock.patch.object(
            indexing, "embed", side_effect=self.embedding_for
        ):
            return answers.recall(self.db, str(self.file), question, **rest)

    def test_the_same_question_is_served(self) -> None:
        self.store("how hot can it get")
        found = self.ask("how hot can it get")
        assert found is not None
        self.assertEqual(found.distance, 0.0)
        self.assertEqual(found.reply["answer"], VERIFIED["answer"])

    def test_a_near_question_is_served(self) -> None:
        self.place("how hot can it get", along(0))
        self.place("how hot can it get?", along(0, tilt=0.2))
        self.store("how hot can it get")
        self.assertIsNotNone(self.ask("how hot can it get?"))

    def test_a_question_too_far_is_not_served(self) -> None:
        self.place("output sink current", along(0))
        self.place("output source current", along(0, tilt=0.8))
        self.store("output sink current")
        self.assertIsNone(self.ask("output source current"))

    def test_the_nearest_of_several_wins(self) -> None:
        self.place("near", along(0, tilt=0.05))
        self.place("far", along(0, tilt=0.3))
        self.place("asked", along(0))
        self.store("near", VERIFIED | {"answer": "the near one"})
        self.store("far", VERIFIED | {"answer": "the far one"})
        found = self.ask("asked", distance=0.5)
        assert found is not None
        self.assertEqual(found.reply["answer"], "the near one")

    def test_an_unverified_answer_is_never_kept(self) -> None:
        self.store("how hot can it get", UNVERIFIED)
        self.assertIsNone(self.ask("how hot can it get"))

    def test_an_empty_cache_answers_nothing(self) -> None:
        self.assertIsNone(self.ask("how hot can it get"))

    def test_a_changed_file_throws_its_answers_away(self) -> None:
        self.store("how hot can it get")
        time.sleep(0.01)
        self.file.write_bytes(b"%PDF-1.4\nchanged\n")
        self.assertIsNone(self.ask("how hot can it get"))
        held = self.db.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
        self.assertEqual(held, 0)

    def test_another_file_does_not_answer_for_this_one(self) -> None:
        other = Path(self.dir.name) / "standard.pdf"
        other.write_bytes(b"%PDF-1.4\n")
        self.store("how hot can it get")
        with mock.patch.object(
            indexing, "embed", side_effect=self.embedding_for
        ):
            self.assertIsNone(
                answers.recall(self.db, str(other), "how hot can it get")
            )

    def test_an_expired_answer_is_dropped(self) -> None:
        self.store("how hot can it get")
        self.assertIsNone(self.ask("how hot can it get", ttl=-1.0))
        held = self.db.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
        self.assertEqual(held, 0)

    def test_asking_twice_keeps_one_entry(self) -> None:
        self.store("how hot can it get")
        self.store("how hot can it get")
        held = self.db.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
        vectors = self.db.execute(
            "SELECT COUNT(*) FROM vec_answers"
        ).fetchone()[0]
        self.assertEqual((held, vectors), (1, 1))

    def test_a_changed_file_leaves_no_orphan_vector(self) -> None:
        self.store("how hot can it get")
        time.sleep(0.01)
        self.file.write_bytes(b"%PDF-1.4\nchanged\n")
        self.ask("how hot can it get")
        vectors = self.db.execute(
            "SELECT COUNT(*) FROM vec_answers"
        ).fetchone()[0]
        self.assertEqual(vectors, 0)

    def test_an_expired_answer_leaves_no_orphan_vector(self) -> None:
        self.store("how hot can it get")
        self.ask("how hot can it get", ttl=-1.0)
        vectors = self.db.execute(
            "SELECT COUNT(*) FROM vec_answers"
        ).fetchone()[0]
        self.assertEqual(vectors, 0)

    def test_a_file_that_is_not_there_is_an_error(self) -> None:
        with self.assertRaises(IndexingError):
            answers.fingerprint(str(self.file.parent / "absent.pdf"))

    def test_the_listing_counts_per_file(self) -> None:
        self.store("one")
        self.store("two")
        listed = answers.kept(self.db)["answers"]
        self.assertEqual(listed[0]["file"], str(self.file))
        self.assertEqual(listed[0]["questions"], 2)


class QuietTest(unittest.TestCase):
    """A cache that cannot be reached must not break the call."""

    def test_a_broken_index_is_a_miss_not_a_failure(self) -> None:
        with mock.patch.object(
            answers, "connect", side_effect=sqlite3.Error("no such file")
        ):
            self.assertIsNone(answers.held("/absent.pdf", "how hot"))

    def test_a_broken_index_does_not_stop_a_store(self) -> None:
        with mock.patch.object(
            answers, "connect", side_effect=sqlite3.Error("read-only")
        ):
            answers.keep("/absent.pdf", "how hot", VERIFIED)

    def test_an_embedder_that_is_down_is_a_miss(self) -> None:
        with mock.patch.object(
            answers, "connect", side_effect=IndexingError("ollama is down")
        ):
            self.assertIsNone(answers.held("/absent.pdf", "how hot"))

    def test_a_ttl_of_zero_switches_it_off(self) -> None:
        with (
            mock.patch.object(answers, "ANSWER_TTL", 0.0),
            mock.patch.object(answers, "connect") as opened,
        ):
            self.assertIsNone(answers.held("/absent.pdf", "how hot"))
            answers.keep("/absent.pdf", "how hot", VERIFIED)
        opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
