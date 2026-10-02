# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mateusz Okulanis
"""Where the servers are.

The installer writes stack.env; this reads it rather than repeating the
addresses, so moving a port is a change in one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROLES = ("long", "fast")


class ConfigError(RuntimeError):
    """The stack is not installed, or not reachable from this host."""


def config_home() -> Path:
    value = os.environ.get("XDG_CONFIG_HOME")
    return Path(value) if value else Path.home() / ".config"


def stack_env_path() -> Path:
    return config_home() / "shuttle" / "stack.env"


def read_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ConfigError(
            f"{path} is missing; run ./install.sh quadlets to write it"
        )
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


@dataclass(frozen=True)
class Endpoint:
    """One llama-server, as this host can reach it."""

    role: str
    url: str
    # Where the server writes its KV slot dumps, on this side of the
    # container.  The server names them by filename; the delegate needs
    # the directory to see whether one is there and to remove it.
    cache: Path | None = None


def load_endpoints(path: Path | None = None) -> dict[str, Endpoint]:
    """Map each role to the address the delegate can actually use.

    The delegate runs on the host and the shuttle network has no route
    out, so only a published port is usable. An installation made with
    --no-expose-direct is reported as such instead of being guessed at.
    """
    env = read_env(path or stack_env_path())
    endpoints: dict[str, Endpoint] = {}
    for role in ROLES:
        port = env.get(f"SHUTTLE_{role.upper()}_DIRECT_PORT")
        if not port:
            raise ConfigError(
                f"shuttle-{role} publishes no host port; the delegate runs "
                "outside the shuttle network, so reinstall without "
                "--no-expose-direct"
            )
        cache = env.get(f"SHUTTLE_{role.upper()}_CACHE")
        endpoints[role] = Endpoint(
            role,
            f"http://127.0.0.1:{port}",
            Path(cache) if cache else None,
        )
    return endpoints
