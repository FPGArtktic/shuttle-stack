# SPDX-License-Identifier: GPL-3.0-only
"""The loop: what bounds it, what verifies it, what it refuses.

The server is a fake that returns a scripted sequence of tool calls,
so what is tested is the loop's own behaviour rather than a model's.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from shuttle_delegate import agent, code, indexing
from shuttle_delegate.agent import AgentError, Preset
from shuttle_delegate.backend import Call, Completion


def call(tool: str, /, **args: Any) -> Call:
    """One tool call, as the server would have returned it."""
    return Call(id=f"c-{tool}", name=tool, arguments=args)


class FakeServer:
    """Answers with the calls it was given, in order."""

    role = "long"

    def __init__(self, *turns: Any) -> None:
        self.turns = list(turns)
        self.asked: list[list[dict[str, Any]]] = []
        self.tools: list[Any] = []

    def context_size(self) -> int:
        return 8192

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 3)

    def chat(
        self,
        prompt: str,
        n_predict: int,
        temperature: float | None = None,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        return self.converse([{"role": "user", "content": prompt}], n_predict)

    def converse(
        self,
        messages: list[dict[str, Any]],
        n_predict: int,
        temperature: float | None = None,
        schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion:
        self.asked.append(list(messages))
        self.tools.append(tools)
        turn = self.turns.pop(0) if self.turns else "nothing left to say"
        if isinstance(turn, str):
            return Completion(turn, 10, 5)
        return Completion("", 10, 5, calls=tuple(turn))


SHAPED = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


class PresetTest(unittest.TestCase):
    def test_a_preset_with_no_verifier_is_refused(self) -> None:
        with self.assertRaises(AgentError) as caught:
            Preset(name="loose", tools=("finish",)).verified()
        self.assertIn("belongs with Claude", str(caught.exception))

    def test_a_verifier_without_an_image_is_refused(self) -> None:
        with self.assertRaises(AgentError) as caught:
            Preset(
                name="loose", tools=("finish",), verifier=("bash", "-n")
            ).verified()
        self.assertIn("not run on the host", str(caught.exception))

    def test_a_schema_is_verifier_enough(self) -> None:
        Preset(name="shaped", tools=("finish",), schema=SHAPED).verified()

    def test_every_built_in_preset_has_a_verifier(self) -> None:
        for name, one in agent.BUILT_IN.items():
            with self.subTest(preset=name):
                one.verified()

    def test_a_tool_that_is_not_one_is_refused(self) -> None:
        with self.assertRaises(AgentError) as caught:
            agent.describe(Preset(name="x", tools=("rm_rf",), schema=SHAPED))
        self.assertIn("search_docs", str(caught.exception))

    def test_the_report_schema_goes_inside_the_finish_tool(self) -> None:
        """So the shape is held by the grammar, not only checked."""
        described = agent.describe(
            Preset(name="x", tools=("finish",), schema=SHAPED)
        )
        inner = described[0]["function"]["parameters"]["properties"]
        self.assertEqual(inner["report"], SHAPED)


class ContentCallTest(unittest.TestCase):
    def test_a_call_written_as_a_message_is_read(self) -> None:
        got = agent._in_content(
            '{"name": "finish", "arguments": {"report": {"a": 1}}}',
            ("finish",),
        )
        assert got is not None
        self.assertEqual(got.name, "finish")
        self.assertEqual(got.arguments["report"], {"a": 1})

    def test_a_tool_outside_the_allowlist_is_not_read(self) -> None:
        self.assertIsNone(
            agent._in_content('{"name": "write_file"}', ("finish",))
        )

    def test_prose_is_not_a_call(self) -> None:
        self.assertIsNone(
            agent._in_content("I would call finish", ("finish",))
        )

    def test_broken_json_is_not_a_call(self) -> None:
        self.assertIsNone(agent._in_content('{"name": ', ("finish",)))


class LoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        patch = mock.patch.dict(os.environ, {"SHUTTLE_HOME": self.dir.name})
        patch.start()
        self.addCleanup(patch.stop)
        where = mock.patch.object(
            indexing,
            "index_file",
            return_value=Path(self.dir.name) / "documents.sqlite",
        )
        where.start()
        self.addCleanup(where.stop)

    def run_with(self, preset: Preset, *turns: Any) -> agent.Run:
        server = FakeServer(*turns)
        self.server = server
        return agent.start(lambda _p: server, preset, "do the thing")

    def test_a_finished_report_that_fits_the_schema_verifies(self) -> None:
        preset = Preset(name="x", tools=("finish",), schema=SHAPED)
        run = self.run_with(preset, [call("finish", report={"answer": "a"})])
        self.assertTrue(run.verified)
        self.assertEqual(run.report, {"answer": "a"})
        self.assertEqual(run.stopped, "finished and verified")

    def test_a_report_missing_a_field_does_not_verify(self) -> None:
        preset = Preset(name="x", tools=("finish",), schema=SHAPED)
        run = self.run_with(
            preset,
            [call("finish", report={"other": "a"})],
            [call("finish", report={"other": "a"})],
            [call("finish", report={"other": "a"})],
            [call("finish", report={"other": "a"})],
        )
        self.assertFalse(run.verified)
        self.assertIn("not moving", run.stopped)

    def test_the_step_budget_stops_it(self) -> None:
        preset = Preset(
            name="x", tools=("finish",), schema=SHAPED, max_steps=2
        )
        run = self.run_with(
            preset, "thinking", "thinking more", "and more", "and more"
        )
        self.assertEqual(len(run.steps), 2)
        self.assertIn("step budget", run.stopped)

    def test_the_token_budget_stops_it(self) -> None:
        preset = Preset(
            name="x", tools=("finish",), schema=SHAPED, max_tokens=20
        )
        run = self.run_with(preset, "a", "b", "c", "d", "e")
        self.assertIn("token budget", run.stopped)

    def test_asking_a_lookup_the_same_thing_stops_it(self) -> None:
        """Measured: a succeeding tool called with identical arguments
        seven times, because the detector only counted failures."""
        preset = Preset(
            name="x", tools=("search_docs", "finish"), schema=SHAPED
        )
        with mock.patch.object(
            indexing, "search", return_value={"hits": []}
        ) as searched:
            run = self.run_with(
                preset,
                *[[call("search_docs", question="same")] for _ in range(6)],
            )
        self.assertIn("asked the same thing", run.stopped)
        self.assertEqual(searched.call_count, 2)
        # As many calls in the transcript as the reason counts.
        self.assertEqual(len(run.steps), 3)
        self.assertFalse(run.steps[-1].ok)

    def test_a_different_question_is_not_a_repeat(self) -> None:
        preset = Preset(
            name="x", tools=("search_docs", "finish"), schema=SHAPED
        )
        with mock.patch.object(indexing, "search", return_value={"hits": []}):
            run = self.run_with(
                preset,
                [call("search_docs", question="one")],
                [call("search_docs", question="two")],
                [call("search_docs", question="three")],
                [call("finish", report={"answer": "a"})],
            )
        self.assertTrue(run.verified)

    def test_the_verifier_may_be_run_again_after_a_write(self) -> None:
        """It takes no arguments, so every call looks identical; that
        is the shape of verify-loop and not a loop going nowhere."""
        preset = Preset(
            name="x",
            tools=("write_file", "run_verifier", "finish"),
            verifier=("true",),
            image="localhost/shuttle-docs",
        )
        with mock.patch.object(
            agent, "_verify", return_value=(True, "passed")
        ):
            run = self.run_with(
                preset,
                [call("run_verifier")],
                [call("write_file", name="a.sh", text="x")],
                [call("run_verifier")],
                [call("write_file", name="a.sh", text="y")],
                [call("run_verifier")],
                [call("finish", report={"done": True})],
            )
        self.assertTrue(run.verified)
        self.assertNotIn("the same thing", run.stopped)

    def test_repeated_prose_stops_it(self) -> None:
        preset = Preset(name="x", tools=("finish",), schema=SHAPED)
        run = self.run_with(preset, "same", "same", "same", "same")
        self.assertIn("the same prose", run.stopped)

    def test_a_call_with_broken_arguments_is_a_step_not_a_crash(
        self,
    ) -> None:
        preset = Preset(name="x", tools=("finish",), schema=SHAPED)
        broken = Call(id="c", name="finish", arguments={}, broken="not JSON")
        run = self.run_with(
            preset, [broken], [call("finish", report={"answer": "a"})]
        )
        self.assertTrue(run.verified)
        self.assertFalse(run.steps[0].ok)

    def test_an_empty_task_is_refused(self) -> None:
        preset = Preset(name="x", tools=("finish",), schema=SHAPED)
        with self.assertRaises(AgentError):
            agent.start(lambda _p: FakeServer(), preset, "   ")

    def test_the_tools_offered_are_the_allowlist(self) -> None:
        preset = Preset(
            name="x", tools=("finish", "search_docs"), schema=SHAPED
        )
        self.run_with(preset, [call("finish", report={"answer": "a"})])
        offered = {
            one["function"]["name"] for one in (self.server.tools[0] or [])
        }
        self.assertEqual(offered, {"finish", "search_docs"})

    def test_the_steps_are_written_down_as_they_happen(self) -> None:
        preset = Preset(name="x", tools=("finish",), schema=SHAPED)
        run = self.run_with(preset, [call("finish", report={"answer": "a"})])
        written = (run.directory / "steps.jsonl").read_text()
        self.assertIn("finish", written)
        kept = json.loads((run.directory / "report.json").read_text())
        self.assertTrue(kept["verified"])

    def test_write_file_stays_in_the_output_directory(self) -> None:
        preset = Preset(
            name="x", tools=("write_file", "finish"), schema=SHAPED
        )
        run = self.run_with(
            preset,
            [call("write_file", name="../escape.txt", text="no")],
            [call("write_file", name="kept.txt", text="yes")],
            [call("finish", report={"answer": "a"})],
        )
        self.assertTrue((run.output / "escape.txt").is_file())
        self.assertFalse(Path(self.dir.name, "escape.txt").exists())
        self.assertEqual((run.output / "kept.txt").read_text(), "yes")

    def test_the_transcript_is_not_in_what_the_verifier_checks(
        self,
    ) -> None:
        """The run's own steps.jsonl was being handed to the operator's
        verifier as a file to check, and bash reads only the first of
        its arguments, so the agent's file went unchecked."""
        preset = Preset(
            name="x", tools=("write_file", "finish"), schema=SHAPED
        )
        run = self.run_with(
            preset,
            [call("write_file", name="fix.sh", text="echo hi\n")],
            [call("finish", report={"answer": "a"})],
        )
        self.assertEqual(
            [one.name for one in run.output.iterdir()], ["fix.sh"]
        )
        self.assertTrue((run.directory / "steps.jsonl").is_file())

    def test_a_dotfile_is_not_written(self) -> None:
        preset = Preset(
            name="x", tools=("write_file", "finish"), schema=SHAPED
        )
        run = self.run_with(
            preset,
            [call("write_file", name=".profile", text="no")],
            [call("finish", report={"answer": "a"})],
        )
        self.assertFalse((run.output / ".profile").exists())
        self.assertFalse(run.steps[0].ok)

    def test_grounding_refuses_a_place_no_tool_returned(self) -> None:
        preset = Preset(
            name="x",
            tools=("find_references", "finish"),
            schema={
                "type": "object",
                "properties": {"places": {"type": "array"}},
                "required": ["places"],
            },
            ground=("places",),
        )
        with (
            mock.patch.object(code, "connect"),
            mock.patch.object(
                code, "references", return_value={"references": []}
            ),
        ):
            run = self.run_with(
                preset,
                [call("find_references", name="counter")],
                [call("finish", report={"places": ["invented.sv:42"]})],
                [call("finish", report={"places": ["invented.sv:42"]})],
                [call("finish", report={"places": ["invented.sv:42"]})],
                [call("finish", report={"places": ["invented.sv:42"]})],
            )
        self.assertFalse(run.verified)
        self.assertTrue(
            any("not in what the tools returned" in one for one in run.checks)
        )

    def test_a_verifier_that_never_ran_does_not_verify(self) -> None:
        preset = Preset(
            name="x",
            tools=("finish",),
            verifier=("bash", "-n"),
            image="localhost/shuttle-docs",
        )
        run = self.run_with(
            preset,
            [call("finish", report={"done": True})],
            [call("finish", report={"done": True})],
            [call("finish", report={"done": True})],
            [call("finish", report={"done": True})],
        )
        self.assertFalse(run.verified)
        self.assertIn("the verifier was never run", run.checks)


class SettingsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        patch = mock.patch.dict(os.environ, {"SHUTTLE_HOME": self.dir.name})
        patch.start()
        self.addCleanup(patch.stop)
        self.path = Path(self.dir.name) / agent.FILE

    def test_the_built_in_presets_are_there_without_a_file(self) -> None:
        self.assertIn("repo-scout", agent.settings())

    def test_a_file_can_add_one(self) -> None:
        self.path.write_text(
            '[log-triage]\ntools = ["search_docs", "finish"]\n'
            'schema = { type = "object", required = ["answer"] }\n'
        )
        held = agent.settings()
        self.assertIn("log-triage", held)
        self.assertEqual(held["log-triage"].tools, ("search_docs", "finish"))

    def test_a_file_can_tighten_a_built_in_budget(self) -> None:
        self.path.write_text("[repo-scout]\nmax_steps = 3\n")
        held = agent.settings()
        self.assertEqual(held["repo-scout"].max_steps, 3)
        self.assertEqual(
            held["repo-scout"].tools, agent.BUILT_IN["repo-scout"].tools
        )

    def test_a_name_that_is_not_one_is_refused(self) -> None:
        self.path.write_text('["../etc"]\nmax_steps = 3\n')
        with self.assertRaises(AgentError):
            agent.settings()

    def test_a_file_that_does_not_parse_says_so(self) -> None:
        self.path.write_text("[oops\n")
        with self.assertRaises(AgentError):
            agent.settings()


class NonAsciiGroundingTest(LoopTest):
    """A correct answer may hold a character that is not ASCII.

    The domain is full of them: µA and Ω and °C in a datasheet, the
    curly quotes gcc writes its messages with. Serialising the cited
    fields the default way turns each into a \\uXXXX escape and then
    looks for the escape in the source, so a right answer is refused
    for holding the right character.
    """

    SAID = "expected \u2018;\u2019 before \u2018return\u2019"

    def ground_on(self, *turns: Any) -> agent.Run:
        preset = Preset(
            name="x",
            tools=("search_docs", "finish"),
            schema={
                "type": "object",
                "properties": {"errors": {"type": "array"}},
                "required": ["errors"],
            },
            ground=("errors",),
        )
        with mock.patch.object(
            indexing, "search", return_value={"hits": [{"text": self.SAID}]}
        ):
            return self.run_with(preset, *turns)

    def test_a_line_with_curly_quotes_grounds(self) -> None:
        run = self.ground_on(
            [call("search_docs", question="what failed")],
            [call("finish", report={"errors": [self.SAID]})],
        )
        self.assertTrue(run.verified, run.checks)

    def test_a_line_the_source_does_not_hold_is_still_refused(self) -> None:
        """The fix must not be a loosening of the check."""
        invented = "expected \u2018}\u2019 before \u2018while\u2019"
        run = self.ground_on(
            [call("search_docs", question="what failed")],
            [call("finish", report={"errors": [invented]})],
            [call("finish", report={"errors": [invented]})],
            [call("finish", report={"errors": [invented]})],
            [call("finish", report={"errors": [invented]})],
        )
        self.assertFalse(run.verified)

    def test_the_replayed_call_keeps_the_characters_it_was_given(
        self,
    ) -> None:
        """The model reads its own earlier call back, and an escape
        there is text it did not write."""
        preset = Preset(
            name="x", tools=("search_docs", "finish"), schema=SHAPED
        )
        with mock.patch.object(indexing, "search", return_value={"hits": []}):
            self.run_with(
                preset,
                [call("search_docs", question="\u2018reset\u2019")],
                [call("finish", report={"answer": "a"})],
            )
        replayed = json.dumps(self.server.asked[-1], ensure_ascii=False)
        self.assertIn("\u2018reset\u2019", replayed)
        self.assertNotIn("\\u2018", replayed)


if __name__ == "__main__":
    unittest.main()
