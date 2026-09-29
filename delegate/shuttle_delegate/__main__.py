# SPDX-License-Identifier: GPL-3.0-only
"""Command line entry point."""

from __future__ import annotations

import argparse

from .server import server

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
    server.run(transport=parser.parse_args().transport)


if __name__ == "__main__":
    main()
