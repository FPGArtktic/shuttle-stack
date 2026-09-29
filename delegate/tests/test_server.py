# SPDX-License-Identifier: GPL-3.0-only
"""The MCP surface: what a client sees before it calls anything."""

from __future__ import annotations

import asyncio
import unittest

from shuttle_delegate.server import backend, mcp

EXPECTED = {
    "shuttle_status",
    "shuttle_summarise",
    "shuttle_ask",
    "shuttle_classify",
    "shuttle_extract",
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

    def test_an_unknown_server_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            backend("medium")


if __name__ == "__main__":
    unittest.main()


class ToolErrorTest(unittest.TestCase):
    """A failure the caller can act on must reach it as a message."""

    def test_a_missing_file_is_an_anticipated_failure(self) -> None:
        from mcp.server.mcpserver.exceptions import ToolError

        with self.assertRaises(ToolError) as caught:
            asyncio.run(
                mcp.call_tool(
                    "shuttle_ask",
                    {"path": "/nonexistent/file.txt", "question": "what?"},
                )
            )
        self.assertIn("nonexistent", str(caught.exception))

    def test_ask_offers_a_pattern_and_a_context(self) -> None:
        tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
        fields = tools["shuttle_ask"].input_schema["properties"]
        self.assertIn("pattern", fields)
        self.assertIn("context", fields)
