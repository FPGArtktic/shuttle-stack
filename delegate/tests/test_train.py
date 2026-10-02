# SPDX-License-Identifier: GPL-3.0-only
"""What goes into the adapter, and what is kept out of it.

Nothing here loads torch. The training call is the one part of the
module a test cannot exercise, and everything that decides what it
would be given is here.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from shuttle_delegate import dataset, indexing, runs, train
from shuttle_delegate.train import TrainError

ANSWER = {
    "answer": 'The subject may be at most 72 characters. "at most 72 '
    'characters" is what the text says.',
    "found": True,
    "verified": True,
    "local_tokens": 1457,
    "run": ".shuttle/runs/x.md",
}


class TrainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.home = Path(self.dir.name)
        patch = mock.patch.dict(os.environ, {"SHUTTLE_HOME": str(self.home)})
        patch.start()
        self.addCleanup(patch.stop)
        where = mock.patch.object(
            indexing, "index_file", return_value=self.home / "documents.sqlite"
        )
        where.start()
        self.addCleanup(where.stop)
        self.db = dataset.connect()
        self.addCleanup(self.db.close)
        self.doc = self.home / "CONTRIBUTING.md"
        self.doc.write_text(
            "# Commits\n\nSubject: at most 72 characters, no full stop.\n",
            encoding="utf-8",
        )

    def graded(
        self,
        verdict: str = "good",
        answer: dict[str, Any] | None = None,
        **request: Any,
    ) -> None:
        asked = {
            "path": str(self.doc),
            "question": "how long may a commit subject be",
        } | request
        made = runs.write("ask_file", asked, answer or ANSWER)
        dataset.grade(self.db, made.id, verdict)

    def only(self) -> train.Example:
        kept, dropped = train.examples(self.db)
        self.assertEqual(dropped, {}, dropped)
        self.assertEqual(len(kept), 1)
        return kept[0]

    def test_a_good_case_becomes_a_prompt_and_an_answer(self) -> None:
        self.graded()
        one = self.only()
        self.assertEqual(one.tool, "ask_file")
        self.assertIn("at most 72 characters", one.prompt)
        self.assertIn("how long may a commit subject be", one.prompt)
        self.assertTrue(one.completion.startswith("The subject may be"))

    def test_the_prompt_is_the_one_the_tool_sends(self) -> None:
        """A second prompt format here is how a fine-tune comes out
        useless: the adapter would learn a shape it never sees."""
        self.graded()
        one = self.only()
        self.assertTrue(one.prompt.startswith("TEXT:"))
        self.assertIn("QUESTION: ", one.prompt)
        self.assertIn("NOT IN THIS TEXT", one.prompt)

    def test_the_word_limit_is_the_one_that_was_asked_for(self) -> None:
        self.graded(words=40)
        self.assertIn("at most 40 words", self.only().prompt)

    def test_the_pattern_narrows_the_prompt_as_it_narrowed_the_call(
        self,
    ) -> None:
        self.doc.write_text(
            "# Commits\n\nSubject: at most 72 characters.\n\n"
            "Something else entirely about tests.\n",
            encoding="utf-8",
        )
        self.graded(pattern="^Subject:", context=0)
        prompt = self.only().prompt
        self.assertIn("at most 72 characters", prompt)
        self.assertNotIn("entirely about tests", prompt)

    def test_only_the_answer_is_the_target(self) -> None:
        """A model has no business learning to emit `found` or a
        token count."""
        self.graded()
        said = self.only().completion
        for key in ("found", "verified", "local_tokens", "run"):
            self.assertNotIn(key, said)

    def test_a_verdict_other_than_good_is_not_trained_on(self) -> None:
        self.graded(verdict="wrong")
        kept, dropped = train.examples(self.db)
        self.assertEqual(kept, [])
        self.assertIn("verdict is wrong", dropped)

    def test_a_tool_with_no_rebuild_is_left_out_rather_than_guessed(
        self,
    ) -> None:
        made = runs.write(
            "classify_file", {"path": str(self.doc)}, {"answer": "docs"}
        )
        dataset.grade(self.db, made.id, "good")
        kept, dropped = train.examples(self.db)
        self.assertEqual(kept, [])
        self.assertIn("no prompt to rebuild for classify_file", dropped)

    def test_a_document_that_has_moved_on_drops_the_pair(self) -> None:
        """The file is read again to rebuild the prompt, so the answer
        may no longer quote it, and then the two no longer belong
        together."""
        self.graded()
        self.doc.write_text(
            "# Commits\n\nSubject: at most 50 characters now.\n",
            encoding="utf-8",
        )
        kept, dropped = train.examples(self.db)
        self.assertEqual(kept, [])
        self.assertIn("no longer quotes the document", str(dropped))

    def test_a_document_that_is_gone_drops_the_pair(self) -> None:
        self.graded()
        self.doc.unlink()
        kept, dropped = train.examples(self.db)
        self.assertEqual(kept, [])
        self.assertTrue(dropped)

    def test_a_prompt_too_long_for_the_card_is_dropped(self) -> None:
        self.doc.write_text(
            "# Commits\n\nSubject: at most 72 characters.\n"
            + "filler sentence to make this document long. " * 400,
            encoding="utf-8",
        )
        self.graded()
        kept, dropped = train.examples(self.db)
        self.assertEqual(kept, [])
        self.assertTrue(any("characters" in why for why in dropped), dropped)

    def test_an_answer_with_no_text_is_dropped(self) -> None:
        self.graded(answer={"answer": "   ", "found": False})
        kept, dropped = train.examples(self.db)
        self.assertEqual(kept, [])
        self.assertIn("holds no text", str(dropped))

    def test_a_run_with_no_question_says_so_rather_than_training(
        self,
    ) -> None:
        with self.assertRaises(TrainError):
            train._ask_prompt({"path": str(self.doc)})


class PlanTest(TrainTest):
    def test_the_plan_counts_what_it_would_train_on(self) -> None:
        self.graded()
        made = train.plan(least=1, db=self.db)
        self.assertEqual(made["examples"], 1)
        self.assertEqual(made["tools"], ["ask_file"])
        self.assertTrue(made["enough"])
        self.assertNotIn("note", made)

    def test_too_few_cases_is_said_and_not_trained_on(self) -> None:
        """Seven will not move a benchmark, and an adapter trained on
        seven and reported as an improvement is worse than none."""
        self.graded()
        made = train.plan(least=32, db=self.db)
        self.assertFalse(made["enough"])
        self.assertIn("not enough", made["note"])
        self.assertIn("grade_run", made["note"])

    def test_the_plan_needs_no_card_and_no_torch(self) -> None:
        """It is the first question on any machine: is there enough."""
        self.graded()
        with mock.patch.dict("sys.modules", {"torch": None}):
            made = train.plan(least=1, db=self.db)
        self.assertEqual(made["examples"], 1)

    def test_an_empty_set_plans_nothing_rather_than_failing(self) -> None:
        made = train.plan(least=1, db=self.db)
        self.assertEqual(made["examples"], 0)
        self.assertFalse(made["enough"])


class DigestTest(unittest.TestCase):
    def test_the_hash_is_of_the_file(self) -> None:
        """M7 asks for the base model's hash beside the adapter,
        because nothing in a LoRA file says which weights it fits."""
        import hashlib

        with tempfile.TemporaryDirectory() as where:
            path = Path(where) / "weights.bin"
            path.write_bytes(b"x" * (3 * 1024 * 1024))
            self.assertEqual(
                train.digest_of(path),
                hashlib.sha256(b"x" * (3 * 1024 * 1024)).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
