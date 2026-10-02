# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mateusz Okulanis
"""The MCP surface: what a client sees before it calls anything."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mcp.server.mcpserver.exceptions import ToolError

from shuttle_delegate import code, server
from shuttle_delegate.code import Cut, Unit
from shuttle_delegate.profiles import ProfileError
from shuttle_delegate.server import anticipated, backend, mcp
from shuttle_delegate.tasks import TaskError

JOBLESS = {
    "status",
    "grade_run",
    "find_references",
    "expand_symbol",
    "agent_presets",
    "agent_start",
    "index_path",
    "search_docs",
    "search_code",
    "list_indexed",
    "start_job",
    "get_status",
    "get_result",
    "session_open",
    "session_ask",
    "session_close",
}

EXPECTED = {
    "status",
    "index_path",
    "search_docs",
    "search_code",
    "find_references",
    "expand_symbol",
    "grade_run",
    "agent_start",
    "agent_presets",
    "list_indexed",
    "session_open",
    "session_ask",
    "session_close",
    "brainstorm",
    "start_job",
    "get_status",
    "get_result",
    "summarize_file",
    "ask_file",
    "classify_file",
    "extract",
}


class SurfaceTest(unittest.TestCase):
    def test_every_expected_tool_is_registered(self) -> None:
        names = {tool.name for tool in asyncio.run(mcp.list_tools())}
        self.assertTrue(names >= EXPECTED, f"missing: {EXPECTED - names}")

    def test_every_tool_describes_itself(self) -> None:
        for tool in asyncio.run(mcp.list_tools()):
            with self.subTest(tool=tool.name):
                self.assertTrue(tool.description)

    def test_instructions_tell_the_client_to_pass_a_path(self) -> None:
        self.assertIn("path", mcp.instructions or "")

    def test_an_unknown_profile_is_refused(self) -> None:
        with self.assertRaises(ProfileError):
            backend("no-such-profile")


class ToolErrorTest(unittest.TestCase):
    """A failure the caller can act on must reach it as a message.

    Nothing here may depend on the stack being installed: a missing
    stack.env and a missing file are both anticipated, and a test that
    tells them apart passes on a developer's machine and fails on a
    runner that has neither.
    """

    def test_an_anticipated_failure_becomes_a_tool_error(self) -> None:
        @anticipated
        def boom() -> None:
            raise TaskError("the file holds no text")

        with self.assertRaises(ToolError) as caught:
            boom()
        self.assertIn("holds no text", str(caught.exception))

    def test_a_crash_is_not_dressed_up_as_one(self) -> None:
        @anticipated
        def boom() -> None:
            raise RuntimeError("a bug, not a bad argument")

        with self.assertRaises(RuntimeError):
            boom()

    def test_a_failing_call_carries_a_message_not_a_traceback(self) -> None:
        with self.assertRaises(ToolError) as caught:
            asyncio.run(
                mcp.call_tool(
                    "ask_file",
                    {"path": "/nonexistent/file.txt", "question": "what?"},
                )
            )
        self.assertTrue(str(caught.exception).strip())

    def test_ask_offers_a_pattern_and_a_context(self) -> None:
        tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
        fields = tools["ask_file"].input_schema["properties"]
        self.assertIn("pattern", fields)
        self.assertIn("context", fields)

    def test_the_tools_take_a_profile_not_a_server(self) -> None:
        for tool in asyncio.run(mcp.list_tools()):
            if tool.name in JOBLESS:
                continue
            with self.subTest(tool=tool.name):
                fields = tool.input_schema["properties"]
                self.assertIn("profile", fields)
                self.assertNotIn("server", fields)


class IndexPathTest(unittest.TestCase):
    """Indexing has to leave behind enough to open the file again.

    A citation is a name and a line, so the name alone was enough for
    a long time and nothing noticed it was all that got stored. It is
    not enough for an agent: read_lines resolves a name through the
    index, and a source indexed without its path cannot be read.
    """

    def test_a_source_file_is_indexed_with_where_it_came_from(self) -> None:
        with tempfile.TemporaryDirectory() as where:
            source = Path(where) / "top.sv"
            source.write_text("module top; endmodule\n")
            cut = Cut(
                "top.sv",
                "systemverilog",
                "tree-sitter",
                "",
                [Unit("module_declaration", "top", 1, 1, "module top")],
            )
            with (
                mock.patch.object(code, "units", return_value=cut),
                mock.patch.object(code, "connect"),
                mock.patch.object(code, "store", return_value=1) as stored,
            ):
                server.index_path(str(source))
            self.assertEqual(stored.call_args.args[-1], str(source.resolve()))


class SearchDocsTest(unittest.TestCase):
    def test_a_file_can_narrow_the_search(self) -> None:
        fields = {
            tool.name: tool.input_schema["properties"]
            for tool in asyncio.run(mcp.list_tools())
        }
        self.assertIn("file", fields["search_docs"])
        self.assertIn("file", fields["search_code"])


if __name__ == "__main__":
    unittest.main()
