# SPDX-License-Identifier: GPL-3.0-only
"""The MCP surface: what a client sees before it calls anything."""

from __future__ import annotations

import asyncio
import unittest

from mcp.server.mcpserver.exceptions import ToolError

from shuttle_delegate.profiles import ProfileError
from shuttle_delegate.server import anticipated, backend, mcp
from shuttle_delegate.tasks import TaskError

EXPECTED = {
    "status",
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


if __name__ == "__main__":
    unittest.main()


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
            if tool.name == "status":
                continue
            with self.subTest(tool=tool.name):
                fields = tool.input_schema["properties"]
                self.assertIn("profile", fields)
                self.assertNotIn("server", fields)


if __name__ == "__main__":
    unittest.main()
