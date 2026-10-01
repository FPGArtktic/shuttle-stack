# SPDX-License-Identifier: GPL-3.0-only
"""The archive: what goes in, what stays out, and that it opens."""

from __future__ import annotations

import os
import sqlite3
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from shuttle_delegate import backup, indexing
from shuttle_delegate.backup import BackupError


class BackupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.home = Path(self.dir.name) / "state"
        patch = mock.patch.dict(os.environ, {"SHUTTLE_HOME": str(self.home)})
        patch.start()
        self.addCleanup(patch.stop)
        self.index = self.home / "documents.sqlite"
        where = mock.patch.object(
            indexing, "index_file", return_value=self.index
        )
        where.start()
        self.addCleanup(where.stop)
        self.into = Path(self.dir.name) / "backups"

    def build(self) -> None:
        db = indexing.connect()
        db.execute(
            "INSERT INTO chunks (document, ordinal, page, clause, heading,"
            " text) VALUES ('a.pdf', 0, 3, '3.1', '3.1 Scope', 'the scope')"
        )
        db.commit()
        db.close()

    def transcript(self, name: str = "s.jsonl") -> None:
        where = self.home / "sessions"
        where.mkdir(parents=True, exist_ok=True)
        (where / name).write_text('{"question": "how hot"}\n')

    def dump(self, name: str = "s.bin") -> None:
        where = self.home / "sessions"
        where.mkdir(parents=True, exist_ok=True)
        (where / name).write_bytes(b"\x00" * 1024)

    def names(self, made: backup.Made) -> list[str]:
        with tarfile.open(made.path) as tar:
            return sorted(tar.getnames())

    def test_the_index_and_the_transcripts_go_in(self) -> None:
        self.build()
        self.transcript()
        made = backup.make(self.into)
        self.assertEqual(
            self.names(made), ["documents.sqlite", "sessions/s.jsonl"]
        )
        self.assertEqual(made.sessions, 1)

    def test_a_kv_dump_stays_out(self) -> None:
        """305 MiB of cache the transcript can rebuild."""
        self.build()
        self.transcript()
        self.dump()
        self.assertNotIn("sessions/s.bin", self.names(backup.make(self.into)))

    def test_the_audit_log_goes_in_with_its_generations(self) -> None:
        self.build()
        (self.home / "audit.jsonl").write_text('{"tool": "ask_file"}\n')
        (self.home / "audit.1.jsonl").write_text('{"tool": "extract"}\n')
        made = backup.make(self.into)
        self.assertEqual(made.logs, 2)
        self.assertIn("audit/audit.1.jsonl", self.names(made))

    def test_the_copy_opens_and_holds_the_rows(self) -> None:
        self.build()
        made = backup.make(self.into)
        out = Path(self.dir.name) / "restored"
        out.mkdir()
        with tarfile.open(made.path) as tar:
            tar.extract("documents.sqlite", out, filter="data")
        db = sqlite3.connect(out / "documents.sqlite")
        self.addCleanup(db.close)
        held = db.execute("SELECT heading FROM chunks").fetchone()
        self.assertEqual(held[0], "3.1 Scope")

    def test_the_index_is_read_only_and_copied_page_by_page(self) -> None:
        """A byte copy of a live database does not reliably open, and
        a backup has no business being able to write to its source."""
        self.build()
        opened: list[str] = []
        real = sqlite3.connect

        def watch(target: Any, *rest: Any, **named: Any) -> Any:
            opened.append(str(target))
            return real(target, *rest, **named)

        with mock.patch("sqlite3.connect", side_effect=watch):
            backup.make(self.into)
        self.assertTrue(any("mode=ro" in one for one in opened), opened)

    def test_backing_up_does_not_touch_the_source(self) -> None:
        self.build()
        before = self.index.stat()
        backup.make(self.into)
        after = self.index.stat()
        self.assertEqual(
            (before.st_size, before.st_mtime_ns),
            (after.st_size, after.st_mtime_ns),
        )

    def test_an_absent_index_says_so(self) -> None:
        with self.assertRaises(BackupError) as caught:
            backup.make(self.into)
        self.assertIn("no index to back up", str(caught.exception))

    def test_the_directory_is_made_if_it_is_not_there(self) -> None:
        self.build()
        made = backup.make(self.into / "deeper")
        self.assertTrue(made.path.is_file())

    def test_the_report_names_the_archive_and_its_parts(self) -> None:
        self.build()
        self.transcript()
        text = backup.make(self.into).report()
        self.assertIn("archive", text)
        self.assertIn("1 transcript(s)", text)
        self.assertIn("through SQLite", text)

    def test_a_second_archive_does_not_overwrite_the_first(self) -> None:
        self.build()
        first = backup.make(self.into)
        with mock.patch("time.strftime", return_value="later"):
            second = backup.make(self.into)
        self.assertNotEqual(first.path, second.path)
        self.assertTrue(first.path.is_file())


if __name__ == "__main__":
    unittest.main()
