# SPDX-License-Identifier: GPL-3.0-only
"""The MCP surface: argument checking, wording, and nothing else."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from . import tasks
from .backend import Backend, BackendError
from .config import ROLES, ConfigError, load_endpoints

INSTRUCTIONS = """\
SHUTTLE hands bulk text work to two llama-servers on this machine, so
that the text itself never enters your context.

Pass a file path, not the file's contents. The delegate reads the file,
splits it if it does not fit the server's context, and returns only the
result. Reading a file yourself and pasting it into a tool call spends
the tokens these tools exist to save.

shuttle-long holds the larger model and answers better; shuttle-fast
holds a smaller one and answers sooner. Every answer reports the tokens
the local servers spent, which are the tokens you did not.
"""

mcp = MCPServer(name="shuttle", instructions=INSTRUCTIONS)

_backends: dict[str, Backend] = {}


def backends() -> dict[str, Backend]:
    """Connect on first use, so the server starts without the stack."""
    if not _backends:
        for role, endpoint in load_endpoints().items():
            _backends[role] = Backend(endpoint)
    return _backends


def backend(role: str) -> Backend:
    if role not in ROLES:
        raise ValueError(f"unknown server '{role}'; use one of {ROLES}")
    return backends()[role]


def _describe(server: Backend) -> dict:
    entry: dict = {
        "server": f"shuttle-{server.role}",
        "url": server.endpoint.url,
    }
    try:
        props = server.props()
    except BackendError as error:
        return entry | {"reachable": False, "reason": str(error)}
    settings = props.get("default_generation_settings", {})
    return entry | {
        "reachable": True,
        "model": props.get("model_path", "").rsplit("/", 1)[-1],
        "context": settings.get("n_ctx"),
    }


@mcp.tool(
    description="Report which local servers answer, which model each "
    "holds and how large its context is. Call this when another tool "
    "fails, or to choose between the two servers."
)
def shuttle_status() -> dict:
    try:
        return {"servers": [_describe(b) for b in backends().values()]}
    except ConfigError as error:
        return {"servers": [], "error": str(error)}


@mcp.tool(
    description="Summarise a file on this machine. Give the path, not "
    "the contents: the file is read here and only the summary comes "
    "back, so a large file costs you nothing to read. A file longer "
    "than the server's context is summarised in parts and the parts "
    "folded together. Use shuttle-long unless speed matters more than "
    "quality."
)
def shuttle_summarise(
    path: str,
    words: int = 200,
    focus: str = "",
    server: str = "long",
) -> dict:
    return tasks.summarise(
        backend(server), tasks.read_text(path), words, focus
    )


@mcp.tool(
    description="Put a file into one of the labels you give. The label "
    "is constrained by a schema, so the answer is always one of them. "
    "A file longer than the context is classified in parts and the "
    "parts vote; the answer carries the agreement between them. "
    "shuttle-fast is usually enough for this."
)
def shuttle_classify(
    path: str,
    labels: list[str],
    question: str = "",
    server: str = "fast",
) -> dict:
    return tasks.classify(
        backend(server), tasks.read_text(path), labels, question
    )


@mcp.tool(
    description="Pull structured fields out of a file, following a JSON "
    "schema you supply; the server is held to the schema, so the shape "
    "of the answer is guaranteed. The file must fit the context in one "
    "piece: fields cannot be merged across parts without inventing "
    "precedence, and a file too large is refused rather than guessed at."
)
def shuttle_extract(
    path: str,
    schema: dict,
    instructions: str = "",
    server: str = "long",
) -> dict:
    return tasks.extract(
        backend(server), tasks.read_text(path), schema, instructions
    )
