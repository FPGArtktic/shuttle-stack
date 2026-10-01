# SPDX-License-Identifier: GPL-3.0-only
"""Which documents a night holds, and what it lets go of.

No server is started. The sessions module is where opening and
closing are tested; what is tested here is the choosing.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from shuttle_delegate import audit, sessions, warm


def line(tool: str, path: str, ok: bool = True) -> str:
    return json.dumps(
        {
            "at": "2026-10-01T03:00:00+0200",
            "tool": tool,
            "args": {"path": path},
            "ok": ok,
        }
    )


class HomeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.home = Path(self.dir.name)
        patch = mock.patch.dict(os.environ, {"SHUTTLE_HOME": str(self.home)})
        patch.start()
        self.addCleanup(patch.stop)

    def logged(self, *lines: str) -> None:
        (self.home / audit.NAME).write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def made(self, name: str) -> str:
        """A real file, so that a choice can be checked against disk."""
        where = self.home / name
        where.write_text(f"about {name}\n", encoding="utf-8")
        return str(where)


class AskedTest(HomeTest):
    def test_nothing_logged_is_nothing_to_hold(self) -> None:
        self.assertEqual(warm.asked(), [])

    def test_the_most_asked_document_comes_first(self) -> None:
        self.logged(
            line("ask_file", "/a/one.pdf"),
            line("ask_file", "/a/two.pdf"),
            line("extract", "/a/two.pdf"),
            line("summarize_file", "/a/two.pdf"),
        )
        self.assertEqual(warm.asked(), [("/a/two.pdf", 3), ("/a/one.pdf", 1)])

    def test_a_call_that_failed_does_not_count(self) -> None:
        self.logged(
            line("ask_file", "/a/gone.pdf", ok=False),
            line("ask_file", "/a/here.pdf"),
        )
        self.assertEqual(warm.asked(), [("/a/here.pdf", 1)])

    def test_a_search_is_not_a_document_read(self) -> None:
        """It reads the index, so holding the file saves nothing."""
        self.logged(
            line("search_docs", "/a/one.pdf"),
            line("find_references", "/a/one.pdf"),
            line("ask_file", "/a/two.pdf"),
        )
        self.assertEqual(warm.asked(), [("/a/two.pdf", 1)])

    def test_a_line_that_is_not_json_is_skipped(self) -> None:
        self.logged("not json at all", line("ask_file", "/a/one.pdf"))
        self.assertEqual(warm.asked(), [("/a/one.pdf", 1)])

    def test_an_entry_with_no_path_is_skipped(self) -> None:
        self.logged(
            json.dumps({"tool": "ask_file", "args": {}, "ok": True}),
            line("ask_file", "/a/one.pdf"),
        )
        self.assertEqual(warm.asked(), [("/a/one.pdf", 1)])

    def test_an_older_generation_counts_too(self) -> None:
        self.logged(line("ask_file", "/a/one.pdf"))
        audit.generation(self.home / audit.NAME, 1).write_text(
            line("ask_file", "/a/one.pdf") + "\n", encoding="utf-8"
        )
        self.assertEqual(warm.asked(), [("/a/one.pdf", 2)])


class RefreshTest(HomeTest):
    def setUp(self) -> None:
        super().setUp()
        self.opened: list[str] = []
        self.closed: list[str] = []
        self.at = 0

        def open_session(
            server: Any, profile: str, path: str, words: int = 200
        ) -> dict[str, Any]:
            self.at += 1
            self.opened.append(path)
            return {"session": f"s{self.at}", "tokens": 1404}

        def close(resolve: Any, session: str) -> dict[str, Any]:
            self.closed.append(session)
            return {"closed": True}

        for name, what in (("open_session", open_session), ("close", close)):
            patch = mock.patch.object(sessions, name, what)
            patch.start()
            self.addCleanup(patch.stop)

    def resolve(self, profile: str) -> Any:
        return mock.Mock(role="long")

    def test_keeping_none_holds_nothing(self) -> None:
        self.logged(line("ask_file", self.made("one.md")))
        self.assertEqual(warm.refresh(self.resolve, keep=0), [])
        self.assertEqual(self.opened, [])

    def test_the_budget_is_the_number_of_documents(self) -> None:
        self.logged(
            *[
                line("ask_file", self.made(name))
                for name in ("one.md", "two.md", "three.md")
            ]
        )
        got = warm.refresh(self.resolve, keep=2)
        self.assertEqual(len(got), 2)
        self.assertEqual(len(self.opened), 2)

    def test_a_document_whose_file_has_gone_is_not_held(self) -> None:
        self.logged(
            line("ask_file", "/a/vanished.pdf"),
            line("ask_file", "/a/vanished.pdf"),
            line("ask_file", self.made("here.md")),
        )
        got = warm.refresh(self.resolve, keep=1)
        self.assertEqual(
            [one.path for one in got], [str(self.home / "here.md")]
        )

    def test_the_session_is_written_down(self) -> None:
        where = self.made("one.md")
        self.logged(line("ask_file", where))
        warm.refresh(self.resolve, keep=1)
        self.assertEqual(warm.load_state(), {where: "s1"})

    def test_a_document_still_wanted_keeps_its_session(self) -> None:
        """So the id the morning report names survives the night."""
        where = self.made("one.md")
        self.logged(line("ask_file", where))
        warm.refresh(self.resolve, keep=1)
        got = warm.refresh(self.resolve, keep=1)
        self.assertTrue(got[0].kept)
        self.assertEqual(got[0].session, "s1")
        self.assertEqual(len(self.opened), 1)
        self.assertEqual(self.closed, [])

    def test_a_document_reindexed_tonight_is_read_again(self) -> None:
        where = self.made("one.md")
        self.logged(line("ask_file", where))
        warm.refresh(self.resolve, keep=1)
        got = warm.refresh(self.resolve, keep=1, changed=frozenset({where}))
        self.assertFalse(got[0].kept)
        self.assertEqual(self.closed, ["s1"])
        self.assertEqual(warm.load_state(), {where: "s2"})

    def test_a_document_nobody_asks_about_any_more_is_let_go(self) -> None:
        one = self.made("one.md")
        self.logged(line("ask_file", one))
        warm.refresh(self.resolve, keep=1)
        two = self.made("two.md")
        self.logged(line("ask_file", two), line("ask_file", two))
        got = warm.refresh(self.resolve, keep=1)
        self.assertEqual(self.closed, ["s1"])
        self.assertEqual([one.path for one in got], [two])
        self.assertEqual(warm.load_state(), {two: "s2"})

    def test_a_document_that_will_not_open_is_reported_not_raised(
        self,
    ) -> None:
        where = self.made("one.md")
        self.logged(line("ask_file", where))
        with mock.patch.object(
            sessions,
            "open_session",
            side_effect=sessions.SessionError("too big by 400 tokens"),
        ):
            got = warm.refresh(self.resolve, keep=1)
        self.assertFalse(got[0].ok)
        self.assertIn("too big", got[0].error)
        self.assertEqual(warm.load_state(), {})

    def test_unreadable_state_opens_again_rather_than_failing(self) -> None:
        where = self.made("one.md")
        self.logged(line("ask_file", where))
        warm.state_path().write_text("[not a map]", encoding="utf-8")
        got = warm.refresh(self.resolve, keep=1)
        self.assertEqual(got[0].session, "s1")


class LinesTest(unittest.TestCase):
    def test_nothing_held_says_nothing(self) -> None:
        self.assertEqual(warm.lines([]), [])

    def test_a_held_document_names_the_session_to_ask(self) -> None:
        got = "\n".join(
            warm.lines(
                [
                    warm.Warmed(
                        "/a/one.pdf", "s1", tokens=1404, seconds=2.4, asked=7
                    )
                ]
            )
        )
        self.assertIn("one.pdf", got)
        self.assertIn("s1", got)
        self.assertIn("1404 tokens", got)
        self.assertIn("asked 7x", got)
        self.assertIn("session_ask", got)

    def test_a_failure_says_why_in_the_report(self) -> None:
        got = "\n".join(
            warm.lines(
                [warm.Warmed("/a/one.pdf", error="SessionError: too big")]
            )
        )
        self.assertIn("not cached", got)
        self.assertIn("too big", got)


if __name__ == "__main__":
    unittest.main()
