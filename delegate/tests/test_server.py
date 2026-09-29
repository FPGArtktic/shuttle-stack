# SPDX-License-Identifier: GPL-3.0-only
"""The MCP surface: what a client sees before it calls anything."""

from __future__ import annotations

import asyncio
import unittest

from shuttle_delegate.server import backend, server

EXPECTED = {"shuttle_status"}


class SurfaceTest(unittest.TestCase):
    def test_every_expected_tool_is_registered(self) -> None:
        names = {tool.name for tool in asyncio.run(server.list_tools())}
        self.assertTrue(names >= EXPECTED, f"missing: {EXPECTED - names}")

    def test_every_tool_describes_itself(self) -> None:
        for tool in asyncio.run(server.list_tools()):
            with self.subTest(tool=tool.name):
                self.assertTrue(tool.description)

    def test_instructions_tell_the_client_to_pass_a_path(self) -> None:
        self.assertIn("path", server.instructions or "")

    def test_an_unknown_server_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            backend("medium")


if __name__ == "__main__":
    unittest.main()
