# SPDX-License-Identifier: GPL-3.0-only
"""The hard limit on what comes back, and the file holding the rest."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from shuttle_delegate import audit, runs


def count(text: str) -> int:
    """A tokeniser stand-in: three characters to the token."""
    return max(1, len(text) // 3)


class HomeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        # patch.dict restores what was there; popping the variable
        # would strip it for every test that runs afterwards.
        patch = mock.patch.dict(os.environ, {"SHUTTLE_HOME": self.dir.name})
        patch.start()
        self.addCleanup(patch.stop)

    def test_the_environment_decides_where_runs_land(self) -> None:
        self.assertEqual(runs.home(), Path(self.dir.name))

    def test_without_it_runs_belong_to_the_working_directory(self) -> None:
        del os.environ["SHUTTLE_HOME"]
        self.assertEqual(runs.home(), Path.cwd() / ".shuttle")


class CapTest(unittest.TestCase):
    def test_a_short_text_is_returned_whole(self) -> None:
        text, cut = runs.cap(count, "a brief answer", limit=100)
        self.assertEqual(text, "a brief answer")
        self.assertFalse(cut)

    def test_a_long_text_is_brought_under_the_limit(self) -> None:
        text, cut = runs.cap(count, "word " * 4000, limit=100)
        self.assertTrue(cut)
        self.assertLessEqual(count(text.replace(runs.CUT, "")), 100)

    def test_a_cut_text_says_where_the_rest_is(self) -> None:
        text, _ = runs.cap(count, "word " * 4000, limit=50)
        self.assertIn("run file", text)

    def test_the_limit_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            runs.cap(count, "anything", limit=0)


class ReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        # patch.dict restores what was there; popping the variable
        # would strip it for every test that runs afterwards.
        patch = mock.patch.dict(os.environ, {"SHUTTLE_HOME": self.dir.name})
        patch.start()
        self.addCleanup(patch.stop)

    def test_the_whole_answer_reaches_the_file(self) -> None:
        whole = "sentence. " * 2000
        result = runs.report(
            count, "ask_file", {"path": "x"}, {"answer": whole}
        )
        self.assertTrue(result["report_cut"])
        self.assertLess(len(result["answer"]), len(whole))
        written = (Path(self.dir.name) / "runs").glob("*.md")
        self.assertIn(whole.strip(), next(written).read_text())

    def test_the_request_is_kept_beside_the_result(self) -> None:
        runs.report(
            count,
            "ask_file",
            {"path": "x", "question": "why?"},
            {"answer": "because"},
        )
        body = next((Path(self.dir.name) / "runs").glob("*.md")).read_text()
        self.assertIn("why?", body)
        self.assertIn("because", body)

    def test_structured_fields_are_not_cut(self) -> None:
        fields = {"a": "b" * 10000}
        result = runs.report(count, "extract", {}, {"fields": fields})
        self.assertEqual(result["fields"], fields)
        self.assertFalse(result["report_cut"])

    def test_two_runs_do_not_share_a_file(self) -> None:
        for _ in range(2):
            runs.report(count, "ask_file", {}, {"answer": "a"})
        self.assertEqual(
            len(list((Path(self.dir.name) / "runs").glob("*.md"))), 2
        )


class AuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        # patch.dict restores what was there; popping the variable
        # would strip it for every test that runs afterwards.
        patch = mock.patch.dict(os.environ, {"SHUTTLE_HOME": self.dir.name})
        patch.start()
        self.addCleanup(patch.stop)

    def path(self) -> Path:
        return Path(self.dir.name) / audit.NAME

    def test_every_call_is_one_line_of_json(self) -> None:
        audit.log({"tool": "ask_file", "ok": True})
        audit.log({"tool": "extract", "ok": False})
        lines = self.path().read_text().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[1])["tool"], "extract")

    def test_the_time_is_recorded_without_being_asked(self) -> None:
        audit.log({"tool": "ask_file"})
        self.assertIn("at", json.loads(self.path().read_text()))

    def test_an_unwritable_log_does_not_break_the_answer(self) -> None:
        os.environ["SHUTTLE_HOME"] = "/proc/nowhere/at/all"
        audit.log({"tool": "ask_file"})


class GradientTest(unittest.TestCase):
    """A text denser at the front than the back still gets under."""

    @staticmethod
    def dense(text: str) -> int:
        """Half a token per character for the first 2000, then a quarter."""
        head = min(len(text), 2000)
        return max(1, head // 2 + (len(text) - head) // 4)

    def test_the_cut_is_repeated_until_the_count_agrees(self) -> None:
        text = "x" * 22000
        cut, was_cut = runs.cap(self.dense, text, limit=1000)
        self.assertTrue(was_cut)
        body = cut.replace(runs.CUT, "")
        self.assertLessEqual(self.dense(body), 1000)

    def test_a_steeper_gradient_also_gets_under(self) -> None:
        def steeper(text: str) -> int:
            head = min(len(text), 2000)
            return max(1, head - (len(text) - head) // 4)

        cut, _ = runs.cap(steeper, "y" * 40000, limit=500)
        self.assertLessEqual(steeper(cut.replace(runs.CUT, "")), 500)


if __name__ == "__main__":
    unittest.main()
