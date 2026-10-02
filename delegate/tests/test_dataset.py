# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mateusz Okulanis
"""Grades, and the set they make: the half of M7 that is not training."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from shuttle_delegate import dataset, indexing, runs
from shuttle_delegate.dataset import GradeError

ANSWER = {
    "answer": "at most 72 characters",
    "found": True,
    "verified": True,
    "local_tokens": 1457,
    "local_calls": 3,
    "run": ".shuttle/runs/x.md",
    "server": "shuttle-long",
}


class GradeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.home = Path(self.dir.name)
        patch = mock.patch.dict(os.environ, {"SHUTTLE_HOME": str(self.home)})
        patch.start()
        self.addCleanup(patch.stop)
        where = mock.patch.object(
            indexing,
            "index_file",
            return_value=self.home / "documents.sqlite",
        )
        where.start()
        self.addCleanup(where.stop)
        self.db = dataset.connect()
        self.addCleanup(self.db.close)

    def delegated(
        self, question: str = "how long", answer: dict[str, Any] | None = None
    ) -> str:
        """A run file as the tools write one."""
        made = runs.write(
            "ask_file",
            {"path": "/a/CONTRIBUTING.md", "question": question},
            answer or ANSWER,
        )
        return made.id

    def test_a_good_answer_is_recorded(self) -> None:
        run = self.delegated()
        got = dataset.grade(self.db, run, "good")
        self.assertTrue(got["graded"])
        self.assertFalse(got["corrected"])
        self.assertEqual(got["cases"], {"good": 1})

    def test_a_correction_is_kept_with_the_verdict(self) -> None:
        run = self.delegated()
        dataset.grade(self.db, run, "wrong", "it left mypy out")
        case = dataset.cases(self.db)[0]
        self.assertEqual(case.verdict, "wrong")
        self.assertEqual(case.correction, "it left mypy out")

    def test_grading_twice_replaces_rather_than_doubles(self) -> None:
        run = self.delegated()
        dataset.grade(self.db, run, "good")
        dataset.grade(self.db, run, "wrong", "on reflection")
        self.assertEqual(dataset.counted(self.db), {"wrong": 1})

    def test_a_verdict_outside_the_three_is_refused(self) -> None:
        run = self.delegated()
        with self.assertRaises(GradeError) as caught:
            dataset.grade(self.db, run, "excellent")
        self.assertIn("good", str(caught.exception))

    def test_a_run_that_is_not_there_is_refused(self) -> None:
        with self.assertRaises(GradeError) as caught:
            dataset.grade(self.db, "20260101-000000-ask_file-aaaaaa", "good")
        self.assertIn("no such run", str(caught.exception))

    def test_something_that_is_not_a_run_id_cannot_become_a_path(
        self,
    ) -> None:
        for bad in ("../../etc/passwd", "", "runs/*", "a b"):
            with self.subTest(bad=bad), self.assertRaises(GradeError):
                dataset.grade(self.db, bad, "good")

    def test_the_reference_a_tool_returns_is_accepted(self) -> None:
        """Tools report the run as a path, not as a bare id."""
        run = self.delegated()
        got = dataset.grade(self.db, f".shuttle/runs/{run}.md", "good")
        self.assertEqual(got["run"], run)


class ExportTest(GradeTest):
    def test_the_input_and_the_output_both_reach_the_set(self) -> None:
        dataset.grade(self.db, self.delegated("how long"), "good")
        case = dataset.cases(self.db)[0]
        self.assertEqual(case.request["question"], "how long")
        self.assertEqual(case.answer["answer"], "at most 72 characters")

    def test_what_the_call_cost_is_not_part_of_the_answer(self) -> None:
        """A model has no business learning to emit a token count."""
        dataset.grade(self.db, self.delegated(), "good")
        answer = dataset.cases(self.db)[0].answer
        for key in ("local_tokens", "local_calls", "run", "server"):
            self.assertNotIn(key, answer)

    def test_the_tool_is_read_off_the_run_id(self) -> None:
        dataset.grade(self.db, self.delegated(), "good")
        self.assertEqual(dataset.cases(self.db)[0].tool, "ask_file")

    def test_a_run_whose_file_is_gone_is_skipped_not_fatal(self) -> None:
        run = self.delegated()
        dataset.grade(self.db, run, "good")
        (self.home / "runs" / f"{run}.md").unlink()
        self.assertEqual(dataset.cases(self.db), [])
        self.assertEqual(dataset.counted(self.db), {"good": 1})

    def test_an_unreadable_run_file_is_skipped(self) -> None:
        run = self.delegated()
        dataset.grade(self.db, run, "good")
        (self.home / "runs" / f"{run}.md").write_text("not a run file")
        self.assertEqual(dataset.cases(self.db), [])

    def test_the_file_is_one_json_object_a_line(self) -> None:
        dataset.grade(self.db, self.delegated("one"), "good")
        dataset.grade(self.db, self.delegated("two"), "wrong", "no")
        into = self.home / "set.jsonl"
        got = dataset.export(into, self.db)
        lines = into.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(got["cases"], 2)
        self.assertEqual(got["corrections"], 1)
        self.assertEqual(got["by_verdict"], {"good": 1, "wrong": 1})
        for line in lines:
            self.assertIn("input", json.loads(line))

    def test_an_empty_set_is_a_file_and_not_an_error(self) -> None:
        into = self.home / "set.jsonl"
        got = dataset.export(into, self.db)
        self.assertEqual(got["cases"], 0)
        self.assertTrue(into.is_file())


if __name__ == "__main__":
    unittest.main()
