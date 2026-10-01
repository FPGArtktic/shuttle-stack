# SPDX-License-Identifier: GPL-3.0-only
"""A document read once, asked many times, and changed underneath."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from shuttle_delegate import sessions
from shuttle_delegate.backend import Completion

DOCUMENT = "The gateway is 10.89.7.1.\n\nThe subnet is 10.89.7.0/24.\n"


class FakeServer:
    """Answers, and remembers what was saved and restored."""

    role = "long"

    def __init__(self, answer: str = "the gateway is 10.89.7.1"):
        self.answer = answer
        self.prompts: list[str] = []
        self.saved: dict[str, int] = {}
        self.restored: list[str] = []
        self.forgotten: list[str] = []

    def context_size(self) -> int:
        return 8192

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 3)

    def chat(self, prompt: str, n_predict: int, **_: object) -> Completion:
        self.prompts.append(prompt)
        return Completion(self.answer, tokens_in=10, tokens_out=4)

    def slot_save(self, name: str) -> int:
        self.saved[name] = self.count_tokens(self.prompts[-1])
        return self.saved[name]

    def slot_restore(self, name: str) -> int:
        self.restored.append(name)
        return self.saved.get(name, 0)

    def cached(self, name: str) -> bool:
        return name in self.saved

    def forget(self, name: str) -> None:
        self.forgotten.append(name)
        self.saved.pop(name, None)


class SessionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        patch = mock.patch.dict(os.environ, {"SHUTTLE_HOME": self.dir.name})
        patch.start()
        self.addCleanup(patch.stop)
        self.doc = Path(self.dir.name) / "doc.txt"
        self.doc.write_text(DOCUMENT)
        self.server = FakeServer()

    def open(self, **kwargs: object) -> str:
        return sessions.open_session(self.server, str(self.doc), **kwargs)[
            "session"
        ]

    def test_opening_reads_the_document_and_keeps_its_cache(self) -> None:
        result = sessions.open_session(self.server, str(self.doc))
        self.assertTrue(result["cached"])
        self.assertGreater(result["tokens"], 0)
        self.assertIn(DOCUMENT.strip(), self.server.prompts[0])

    def test_asking_restores_instead_of_sending_it_again(self) -> None:
        session = self.open()
        answer = sessions.ask(self.server, session, "where is the gateway?")
        self.assertEqual(self.server.restored, [f"{session}.bin"])
        self.assertGreater(answer["restored_tokens"], 0)
        self.assertTrue(answer["found"])

    def test_every_exchange_reaches_the_transcript(self) -> None:
        session = self.open()
        for question in ("first?", "second?"):
            sessions.ask(self.server, session, question)
        lines = [
            json.loads(line)
            for line in sessions.transcript(session).read_text().splitlines()
        ]
        kinds = [line["kind"] for line in lines]
        self.assertEqual(kinds, ["open", "ask", "ask"])
        self.assertEqual(lines[2]["question"], "second?")

    def test_a_changed_document_is_read_again_not_trusted(self) -> None:
        session = self.open()
        sessions.ask(self.server, session, "before?")
        self.doc.write_text(DOCUMENT + "\nThe gateway moved.\n")
        answer = sessions.ask(self.server, session, "after?")
        self.assertTrue(answer["source_changed"])
        self.assertEqual(answer["restored_tokens"], 0)
        self.assertIn(f"{session}.bin", self.server.forgotten)
        self.assertIn("The gateway moved.", self.server.prompts[-1])

    def test_the_cache_is_rebuilt_after_the_document_changes(self) -> None:
        session = self.open()
        self.doc.write_text(DOCUMENT + "\nmore\n")
        sessions.ask(self.server, session, "after?")
        answer = sessions.ask(self.server, session, "and again?")
        self.assertFalse(answer["source_changed"])
        self.assertGreater(answer["restored_tokens"], 0)

    def test_a_document_that_does_not_fit_is_refused_at_the_opening(
        self,
    ) -> None:
        self.doc.write_text("word " * 40000)
        with self.assertRaises(sessions.SessionError) as caught:
            sessions.open_session(self.server, str(self.doc))
        self.assertIn("--ctx", str(caught.exception))

    def test_an_answer_the_document_does_not_hold_says_so(self) -> None:
        self.server.answer = "NOT IN THIS TEXT"
        session = self.open()
        answer = sessions.ask(self.server, session, "the phone number?")
        self.assertFalse(answer["found"])
        self.assertEqual(answer["answer"], "")
        self.assertIn("does not answer", answer["note"])

    def test_closing_drops_the_cache_and_keeps_the_transcript(self) -> None:
        session = self.open()
        sessions.ask(self.server, session, "one?")
        result = sessions.close(self.server, session)
        self.assertEqual(result["questions"], 1)
        self.assertIn(f"{session}.bin", self.server.forgotten)
        self.assertTrue(sessions.transcript(session).is_file())

    def test_an_unknown_session_names_what_starts_one(self) -> None:
        with self.assertRaises(sessions.SessionError) as caught:
            sessions.ask(self.server, "no-such-session", "anything?")
        self.assertIn("session_open", str(caught.exception))

    def test_an_empty_question_is_refused(self) -> None:
        session = self.open()
        with self.assertRaises(sessions.SessionError):
            sessions.ask(self.server, session, "   ")

    def test_a_pattern_narrows_what_the_session_holds(self) -> None:
        session = self.open(pattern="subnet", context=0)
        sessions.ask(self.server, session, "which subnet?")
        self.assertIn("10.89.7.0/24", self.server.prompts[-1])
        self.assertNotIn("gateway", self.server.prompts[-1])

    def test_a_pattern_matching_nothing_is_refused(self) -> None:
        with self.assertRaises(sessions.SessionError):
            sessions.open_session(
                self.server, str(self.doc), pattern="absent-from-the-file"
            )


if __name__ == "__main__":
    unittest.main()
