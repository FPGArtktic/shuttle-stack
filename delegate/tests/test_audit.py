# SPDX-License-Identifier: GPL-3.0-only
"""The log that records every call, and how it is kept from growing."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from shuttle_delegate import audit


class LogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        patch = mock.patch.dict(os.environ, {"SHUTTLE_HOME": self.dir.name})
        patch.start()
        self.addCleanup(patch.stop)
        self.path = Path(self.dir.name) / audit.NAME

    def test_a_line_carries_the_time_and_the_entry(self) -> None:
        audit.log({"tool": "ask_file", "ok": True})
        held = json.loads(self.path.read_text().strip())
        self.assertEqual(held["tool"], "ask_file")
        self.assertIn("at", held)

    def test_a_log_that_cannot_be_written_is_not_an_error(self) -> None:
        with mock.patch.object(Path, "open", side_effect=OSError("read-only")):
            audit.log({"tool": "ask_file"})

    def test_an_unserialisable_value_still_reaches_the_log(self) -> None:
        audit.log({"tool": "extract", "args": {"path": Path("/a/b")}})
        self.assertIn("/a/b", self.path.read_text())


class RotateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        patch = mock.patch.dict(os.environ, {"SHUTTLE_HOME": self.dir.name})
        patch.start()
        self.addCleanup(patch.stop)
        self.path = Path(self.dir.name) / audit.NAME

    def fill(self, size: int) -> None:
        self.path.write_text("x" * size)

    def test_a_small_log_is_left_alone(self) -> None:
        self.fill(10)
        self.assertFalse(audit.rotate(self.path, limit=1000))
        self.assertTrue(self.path.exists())

    def test_a_full_log_moves_aside(self) -> None:
        self.fill(2000)
        self.assertTrue(audit.rotate(self.path, limit=1000))
        self.assertFalse(self.path.exists())
        self.assertEqual(
            audit.generation(self.path, 1).read_text(), "x" * 2000
        )

    def test_the_generations_shift_up(self) -> None:
        audit.generation(self.path, 1).write_text("older")
        self.fill(2000)
        audit.rotate(self.path, limit=1000)
        self.assertEqual(audit.generation(self.path, 2).read_text(), "older")

    def test_the_oldest_generation_is_dropped(self) -> None:
        for number in range(1, 3):
            audit.generation(self.path, number).write_text(f"gen{number}")
        self.fill(2000)
        audit.rotate(self.path, limit=1000, keep=2)
        self.assertFalse(audit.generation(self.path, 3).exists())
        self.assertEqual(audit.generation(self.path, 2).read_text(), "gen1")

    def test_keeping_none_just_drops_it(self) -> None:
        self.fill(2000)
        audit.rotate(self.path, limit=1000, keep=0)
        self.assertFalse(self.path.exists())
        self.assertFalse(audit.generation(self.path, 1).exists())

    def test_a_limit_of_zero_never_rotates(self) -> None:
        self.fill(2000)
        self.assertFalse(audit.rotate(self.path, limit=0))

    def test_a_log_that_is_not_there_is_not_rotated(self) -> None:
        self.assertFalse(audit.rotate(self.path, limit=1))

    def test_a_rotation_that_fails_keeps_the_log(self) -> None:
        """A disk problem must not become a lost record."""
        self.fill(2000)
        with mock.patch.object(Path, "replace", side_effect=OSError("busy")):
            self.assertFalse(audit.rotate(self.path, limit=1000))
        self.assertTrue(self.path.exists())

    def test_the_line_is_written_even_when_rotation_fails(self) -> None:
        self.fill(2000)
        with (
            mock.patch.object(audit, "MAX_BYTES", 1000),
            mock.patch.object(Path, "replace", side_effect=OSError("busy")),
        ):
            audit.log({"tool": "ask_file"})
        self.assertIn("ask_file", self.path.read_text())

    def test_logging_past_the_limit_rotates_first(self) -> None:
        self.fill(2000)
        with mock.patch.object(audit, "MAX_BYTES", 1000):
            audit.log({"tool": "ask_file"})
        self.assertEqual(self.path.read_text().count("\n"), 1)
        self.assertIn("ask_file", self.path.read_text())
        self.assertTrue(audit.generation(self.path, 1).exists())

    def test_what_is_held_names_the_live_file_and_its_generations(
        self,
    ) -> None:
        self.fill(10)
        audit.generation(self.path, 1).write_text("older")
        self.assertEqual(
            [one.name for one in audit.held()],
            [audit.NAME, f"audit.1{Path(audit.NAME).suffix}"],
        )


if __name__ == "__main__":
    unittest.main()
