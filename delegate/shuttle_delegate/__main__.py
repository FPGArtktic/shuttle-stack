# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mateusz Okulanis
"""Command line entry point."""

from __future__ import annotations

import argparse

from .server import QUEUE, mcp

TRANSPORTS = ("stdio", "sse", "streamable-http")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="shuttle-delegate",
        description="MCP server in front of the local llama-servers",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=TRANSPORTS,
        help="stdio for a client on this machine (default)",
    )
    try:
        mcp.run(transport=parser.parse_args().transport)
    finally:
        # The job queue holds a worker thread that the interpreter
        # would otherwise wait for at exit.
        QUEUE.close()


if __name__ == "__main__":
    main()
