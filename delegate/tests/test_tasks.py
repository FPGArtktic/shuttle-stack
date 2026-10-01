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
    ask,
    brainstorm,
    classify,
    extract,
    read_text,
    summarise,
    temperatures,
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
        self.temperatures: list[float | None] = []

    def context_size(self) -> int:
        return self._n_ctx

    def count_tokens(self, text: str) -> int:
        return len(text) // 3

    def chat(
        self,
        prompt: str,
        n_predict: int,
        temperature: float | None = None,
        schema: dict | None = None,
    ) -> Completion:
        self.prompts.append(prompt)
        self.schemas.append(schema)
        self.temperatures.append(temperature)
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


class AskTest(unittest.TestCase):
    QUESTION = "what does the phase do?"

    def test_a_short_file_is_asked_once(self) -> None:
        backend = FakeBackend(answers="it installs packages")
        result = ask(backend, "a short document.", self.QUESTION)
        self.assertTrue(result["found"])
        self.assertEqual(result["answer"], "it installs packages")
        self.assertEqual(result["parts"], 1)
        self.assertEqual(result["parts_answering"], 1)

    def test_a_file_with_no_answer_says_so_instead_of_inventing(
        self,
    ) -> None:
        backend = FakeBackend(answers="NOT IN THIS TEXT")
        result = ask(backend, "a short document.", self.QUESTION)
        self.assertFalse(result["found"])
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["parts_answering"], 0)
        self.assertIn("no part", result["note"])

    def test_only_the_parts_that_answer_are_combined(self) -> None:
        def answer(n: int, prompt: str) -> str:
            if "ANSWERS:" in prompt:
                return "combined"
            return "found it" if n % 2 else "NOT IN THIS TEXT"

        backend = FakeBackend(n_ctx=1200, answers=answer)
        result = ask(backend, "paragraph.\n\n" * 400, self.QUESTION, words=20)
        self.assertTrue(result["found"])
        self.assertLess(result["parts_answering"], result["parts"])

    def test_the_question_reaches_the_server(self) -> None:
        backend = FakeBackend(answers="yes")
        ask(backend, "a short document.", self.QUESTION)
        self.assertIn(self.QUESTION, backend.prompts[0])

    def test_an_empty_question_is_refused(self) -> None:
        with self.assertRaises(TaskError):
            ask(FakeBackend(), "text", "   ")

    def test_a_context_with_no_room_for_the_answer_says_so(self) -> None:
        with self.assertRaises(TaskError) as caught:
            ask(FakeBackend(n_ctx=1200), "text", self.QUESTION)
        self.assertIn("no room", str(caught.exception))


class BrainstormTest(unittest.TestCase):
    REQUEST = "ways to test the installer without a GPU"

    def ideas(self, *ideas: str) -> str:
        return json.dumps({"ideas": list(ideas)})

    def test_the_options_come_back_as_a_list(self) -> None:
        backend = FakeBackend(answers=self.ideas("one", "two", "three"))
        result = brainstorm(backend, "", self.REQUEST, count=3)
        self.assertEqual(result["ideas"], ["one", "two", "three"])

    def test_it_says_that_nothing_was_verified(self) -> None:
        backend = FakeBackend(answers=self.ideas("one"))
        result = brainstorm(backend, "", self.REQUEST)
        self.assertFalse(result["grounded"])
        self.assertIn("not findings", result["note"])

    def test_the_shape_is_fixed_by_a_schema(self) -> None:
        backend = FakeBackend(answers=self.ideas("one"))
        brainstorm(backend, "", self.REQUEST)
        schema = backend.schemas[0]
        self.assertEqual(schema["properties"]["ideas"]["type"], "array")

    def test_a_file_is_considered_when_given(self) -> None:
        backend = FakeBackend(answers=self.ideas("one"))
        brainstorm(backend, "a document about ports.", self.REQUEST)
        self.assertIn("a document about ports.", backend.prompts[0])

    def test_without_a_file_nothing_is_quoted_at_it(self) -> None:
        backend = FakeBackend(answers=self.ideas("one"))
        brainstorm(backend, "", self.REQUEST)
        self.assertNotIn("TEXT:", backend.prompts[0])

    def test_an_empty_request_is_refused(self) -> None:
        with self.assertRaises(TaskError):
            brainstorm(FakeBackend(), "", "  ")

    def test_an_absurd_count_is_refused(self) -> None:
        with self.assertRaises(TaskError):
            brainstorm(FakeBackend(), "", self.REQUEST, count=99)

    def test_no_ideas_at_all_is_an_error(self) -> None:
        backend = FakeBackend(answers=json.dumps({"ideas": []}))
        with self.assertRaises(TaskError):
            brainstorm(backend, "", self.REQUEST)


class AttemptsTest(unittest.TestCase):
    """A sample that fails its verifier is drawn again, warmer."""

    SOURCE = "The gateway is 10.89.7.1 and the subnet is 10.89.7.0/24.\n"
    SCHEMA = {
        "type": "object",
        "properties": {"gateway": {"type": "string"}},
        "required": ["gateway"],
    }

    def test_the_first_attempt_keeps_the_profile_temperature(self) -> None:
        self.assertEqual(temperatures(1), [None])
        self.assertIsNone(temperatures(3)[0])

    def test_later_attempts_are_warmer_and_rise(self) -> None:
        warm = temperatures(4)[1:]
        self.assertEqual(warm, sorted(warm))
        self.assertTrue(all(t and t > 0.5 for t in warm))

    def test_an_absurd_number_of_attempts_is_refused(self) -> None:
        with self.assertRaises(TaskError):
            temperatures(99)

    def test_a_value_in_the_source_passes_on_the_first_try(self) -> None:
        backend = FakeBackend(answers=json.dumps({"gateway": "10.89.7.1"}))
        result = extract(backend, self.SOURCE, self.SCHEMA)
        self.assertEqual(result["attempts"], 1)
        self.assertTrue(result["verified"])
        self.assertIsNone(backend.temperatures[0])

    def test_an_invented_value_is_drawn_again(self) -> None:
        def answer(n: int, _: str) -> str:
            value = "192.168.0.1" if n == 1 else "10.89.7.1"
            return json.dumps({"gateway": value})

        backend = FakeBackend(answers=answer)
        result = extract(backend, self.SOURCE, self.SCHEMA, attempts=3)
        self.assertEqual(result["attempts"], 2)
        self.assertTrue(result["verified"])
        self.assertIsNotNone(backend.temperatures[1])

    def test_a_value_never_found_is_reported_not_hidden(self) -> None:
        backend = FakeBackend(answers=json.dumps({"gateway": "192.168.0.1"}))
        result = extract(backend, self.SOURCE, self.SCHEMA, attempts=2)
        self.assertEqual(result["attempts"], 2)
        self.assertFalse(result["verified"])
        self.assertIn("gateway", result["values_not_in_source"][0])

    def test_a_refusal_is_never_drawn_again(self) -> None:
        backend = FakeBackend(answers="NOT IN THIS TEXT")
        result = ask(backend, self.SOURCE, "the phone number?", attempts=3)
        self.assertFalse(result["found"])
        self.assertEqual(result["attempts"], 1)
        self.assertTrue(result["verified"])

    def test_an_answer_quoting_what_is_absent_is_drawn_again(self) -> None:
        def answer(n: int, _: str) -> str:
            if n == 1:
                return 'it is "a sentence nobody ever wrote here"'
            return 'it is "The gateway is 10.89.7.1"'

        backend = FakeBackend(answers=answer)
        result = ask(backend, self.SOURCE, "the gateway?", attempts=3)
        self.assertTrue(result["verified"])
        self.assertEqual(result["attempts"], 2)
