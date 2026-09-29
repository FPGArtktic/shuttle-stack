# SPDX-License-Identifier: GPL-3.0-only
"""The MCP surface: argument checking, wording, and nothing else."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import tasks
from .backend import Backend, BackendError
from .config import ROLES, ConfigError, load_endpoints
from .retrieval import PatternError
from .tasks import TaskError

# Failures the caller can do something about: a path that is not there,
# a pattern that matches nothing, a server that is down.  They reach the
# model as a message rather than as a crash it cannot read.
EXPECTED = (
    TaskError,
    BackendError,
    ConfigError,
    PatternError,
    ValueError,
    OSError,
)

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

Give shuttle_ask a `pattern` whenever you can name what you are looking
for. It does not change what the model can answer, but it changes the
price: the same question over one matching function instead of a whole
35 KB file costs 408 tokens rather than 13079, and three seconds rather
than twenty-two. Repeated questions about the same text are cheaper
again, because the server keeps the cache of a prefix it has seen.
"""

mcp = MCPServer(name="shuttle", instructions=INSTRUCTIONS)

_backends: dict[str, Backend] = {}


def backends() -> dict[str, Backend]:
    """Connect on first use, so the server starts without the stack."""
    if not _backends:
        for role, endpoint in load_endpoints().items():
            _backends[role] = Backend(endpoint)
    return _backends


def anticipated(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except EXPECTED as error:
            raise ToolError(str(error)) from error

    return wrapper


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
@anticipated
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
@anticipated
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
    description="Answer a question about a file on this machine. Give "
    "the path, not the contents. Pass a regexp as `pattern` whenever "
    "you can name what you are looking for, and widen `context` until "
    "the region covers a whole function or section: the same question "
    "over one matching function rather than a whole 35 KB file costs "
    "408 tokens instead of 13079 and answers in three seconds instead "
    "of twenty-two. Asking several questions about the same region is "
    "cheaper still. Without a pattern the whole file is read in parts "
    "and the parts that answer are combined. If nothing in the file "
    "answers, the reply says so rather than inventing one."
)
@anticipated
def shuttle_ask(
    path: str,
    question: str,
    pattern: str = "",
    context: int = 12,
    words: int = 200,
    server: str = "long",
) -> dict:
    return tasks.ask(
        backend(server),
        tasks.read_text(path),
        question,
        words,
        pattern,
        context,
    )


@mcp.tool(
    description="Put a file into one of the labels you give. The label "
    "is constrained by a schema, so the answer is always one of them. "
    "A file longer than the context is classified in parts and the "
    "parts vote; the answer carries the agreement between them. "
    "shuttle-fast is usually enough for this."
)
@anticipated
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
    "of the answer is guaranteed. Pass a `pattern` to read the fields "
    "from one region of a large file; without one the file must fit "
    "the context in a single piece, because fields cannot be merged "
    "across parts without inventing a precedence, and a file too "
    "large is refused rather than guessed at."
)
@anticipated
def shuttle_extract(
    path: str,
    schema: dict,
    instructions: str = "",
    pattern: str = "",
    context: int = 12,
    server: str = "long",
) -> dict:
    return tasks.extract(
        backend(server),
        tasks.read_text(path),
        schema,
        instructions,
        pattern,
        context,
    )
