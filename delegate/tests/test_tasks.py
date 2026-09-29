# SPDX-License-Identifier: GPL-3.0-only
"""The task logic, against a backend that answers from this process."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from shuttle_delegate.backend import Completion
from shuttle_delegate.tasks import (
    TaskError,
    classify,
    extract,
    read_text,
    summarise,
)

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}


class FakeBackend:
    """Answers whatever it is told to, and remembers what it was asked."""

    role = "fake"

    def __init__(self, n_ctx: int = 4096, answers: object = "a summary"):
        self._n_ctx = n_ctx
        self._answers = answers
        self.prompts: list[str] = []
        self.schemas: list[dict | None] = []

    def context_size(self) -> int:
        return self._n_ctx

    def count_tokens(self, text: str) -> int:
        return len(text) // 3

    def chat(
        self,
        prompt: str,
        n_predict: int,
        temperature: float = 0.2,
        schema: dict | None = None,
    ) -> Completion:
        self.prompts.append(prompt)
        self.schemas.append(schema)
        answer = self._answers
        if callable(answer):
            answer = answer(len(self.prompts), prompt)
        return Completion(content=str(answer), tokens_in=10, tokens_out=2)


class ReadTextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def test_a_missing_file_is_named(self) -> None:
        with self.assertRaises(TaskError):
            read_text(str(Path(self.dir.name) / "absent.txt"))

    def test_an_empty_file_is_refused_not_summarised(self) -> None:
        path = Path(self.dir.name) / "empty.txt"
        path.write_text("   \n\n\t\n")
        with self.assertRaises(TaskError) as caught:
            read_text(str(path))
        self.assertIn("no text", str(caught.exception))


class SummariseTest(unittest.TestCase):
    def test_a_short_text_costs_one_call_and_no_folding(self) -> None:
        backend = FakeBackend()
        result = summarise(backend, "a short document.", words=50)
        self.assertEqual(result["rounds"], 0)
        self.assertEqual(result["local_calls"], 1)
        self.assertEqual(result["summary"], "a summary")

    def test_a_long_text_is_folded_and_every_piece_is_read(self) -> None:
        backend = FakeBackend(n_ctx=1200, answers=lambda n, _: f"part {n}")
        result = summarise(backend, "paragraph.\n\n" * 400, words=20)
        self.assertGreaterEqual(result["rounds"], 1)
        self.assertGreater(result["local_calls"], 2)
        self.assertEqual(result["summary"], f"part {result['local_calls']}")

    def test_tokens_spent_locally_are_reported(self) -> None:
        result = summarise(FakeBackend(), "a short document.", words=50)
        self.assertEqual(result["local_tokens"], 12)

    def test_folding_that_does_not_shrink_stops(self) -> None:
        backend = FakeBackend(n_ctx=1200, answers="x" * 4000)
        with self.assertRaises(TaskError) as caught:
            summarise(backend, "paragraph.\n\n" * 400, words=20)
        self.assertIn("shorter", str(caught.exception))

    def test_an_absurd_word_count_is_refused(self) -> None:
        with self.assertRaises(TaskError):
            summarise(FakeBackend(), "text", words=3)

    def test_a_context_with_no_room_says_so(self) -> None:
        with self.assertRaises(TaskError) as caught:
            summarise(FakeBackend(n_ctx=512), "text", words=200)
        self.assertIn("context", str(caught.exception))


class ClassifyTest(unittest.TestCase):
    labels = ["bug", "feature"]

    def answer(self, label: str) -> str:
        return json.dumps({"label": label})

    def test_the_label_comes_back_with_full_agreement(self) -> None:
        backend = FakeBackend(answers=self.answer("bug"))
        result = classify(backend, "a short report.", self.labels)
        self.assertEqual(result["label"], "bug")
        self.assertEqual(result["agreement"], 1.0)

    def test_the_labels_are_enforced_by_a_schema(self) -> None:
        backend = FakeBackend(answers=self.answer("bug"))
        classify(backend, "a short report.", self.labels)
        self.assertEqual(
            backend.schemas[0]["properties"]["label"]["enum"], self.labels
        )

    def test_a_split_vote_is_reported_not_hidden(self) -> None:
        answers = ["bug", "bug", "feature"]
        backend = FakeBackend(
            n_ctx=1200,
            answers=lambda n, _: self.answer(answers[(n - 1) % 3]),
        )
        result = classify(backend, "paragraph.\n\n" * 400, self.labels)
        self.assertLess(result["agreement"], 1.0)
        self.assertEqual(sum(result["votes"].values()), result["local_calls"])

    def test_one_label_is_not_a_classification(self) -> None:
        with self.assertRaises(TaskError):
            classify(FakeBackend(), "text", ["only"])

    def test_an_answer_that_is_not_json_is_an_error(self) -> None:
        backend = FakeBackend(answers="I think it is a bug")
        with self.assertRaises(TaskError) as caught:
            classify(backend, "a short report.", self.labels)
        self.assertIn("JSON", str(caught.exception))


class ExtractTest(unittest.TestCase):
    def test_the_fields_come_back_decoded(self) -> None:
        backend = FakeBackend(answers=json.dumps({"a": "b"}))
        result = extract(backend, "a short note.", SCHEMA)
        self.assertEqual(result["fields"], {"a": "b"})
        self.assertEqual(backend.schemas[0], SCHEMA)

    def test_a_text_needing_several_pieces_is_refused(self) -> None:
        backend = FakeBackend(n_ctx=1200, answers=json.dumps({"a": "b"}))
        with self.assertRaises(TaskError) as caught:
            extract(backend, "paragraph.\n\n" * 400, SCHEMA)
        self.assertIn("summarise it first", str(caught.exception))

    def test_a_schema_that_is_not_an_object_is_refused(self) -> None:
        with self.assertRaises(TaskError):
            extract(FakeBackend(), "text", {"type": "array"})


if __name__ == "__main__":
    unittest.main()
