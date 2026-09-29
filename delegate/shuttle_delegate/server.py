# SPDX-License-Identifier: GPL-3.0-only
"""The MCP server and the tools it exposes."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from .backend import Backend, BackendError
from .config import ROLES, ConfigError, load_endpoints

INSTRUCTIONS = """\
SHUTTLE hands bulk text work to two llama-servers on this machine, so
that the text itself never has to enter your context.

Pass a file path, not the file's contents. The delegate reads the file,
splits it if it does not fit the server's context, and returns only the
result. Reading a file yourself and pasting it into a tool call defeats
the purpose.

shuttle-long is the larger model and answers better; shuttle-fast is
smaller and answers sooner. The tools pick one unless you say otherwise.
"""

server = MCPServer(name="shuttle", instructions=INSTRUCTIONS)

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


def _describe(server_backend: Backend) -> dict:
    entry: dict = {
        "server": f"shuttle-{server_backend.role}",
        "url": server_backend.endpoint.url,
    }
    try:
        props = server_backend.props()
    except BackendError as error:
        return entry | {"reachable": False, "reason": str(error)}
    settings = props.get("default_generation_settings", {})
    model = props.get("model_path", "")
    return entry | {
        "reachable": True,
        "model": model.rsplit("/", 1)[-1],
        "context": settings.get("n_ctx"),
    }


@server.tool(
    description="Report which local servers answer, which model each "
    "holds and how large its context is. Call this when a delegation "
    "fails, or to choose between the two servers."
)
def shuttle_status() -> dict:
    try:
        return {"servers": [_describe(b) for b in backends().values()]}
    except ConfigError as error:
        return {"servers": [], "error": str(error)}
