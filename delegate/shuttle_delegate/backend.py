# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mateusz Okulanis
"""The llama-server HTTP API, narrowed to what the tools need."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .config import Endpoint

HEALTH_TIMEOUT = 5.0
DEFAULT_TIMEOUT = 600.0


class BackendError(RuntimeError):
    """A server refused a request or did not answer."""


def _calls(raw: list[dict[str, Any]]) -> tuple[Call, ...]:
    """The tool calls in an answer, with their arguments parsed."""
    out = []
    for one in raw:
        asked = one.get("function", {}) or {}
        text = asked.get("arguments") or "{}"
        broken = ""
        parsed: dict[str, Any] = {}
        try:
            got = json.loads(text)
            if isinstance(got, dict):
                parsed = got
            else:
                broken = (
                    f"the arguments are {type(got).__name__}, not an object"
                )
        except json.JSONDecodeError as error:
            broken = f"the arguments are not JSON: {error}"
        out.append(
            Call(
                id=str(one.get("id", "")),
                name=str(asked.get("name", "")),
                arguments=parsed,
                broken=broken,
            )
        )
    return tuple(out)


@dataclass(frozen=True)
class Call:
    """One tool the model asked for, with the arguments it gave.

    The arguments arrive as a JSON string and are parsed here rather
    than at the caller: a model that writes `{"path": }` has asked for
    a tool and got the arguments wrong, which is a step the loop can
    tell it about, not a reason for the loop to stop.
    """

    id: str
    name: str
    arguments: dict[str, Any]
    broken: str = ""


@dataclass(frozen=True)
class Completion:
    """One answer, with what it cost and whether it is whole."""

    content: str
    tokens_in: int
    tokens_out: int
    truncated: bool = False
    calls: tuple[Call, ...] = ()

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out


class Server(Protocol):
    """What a pipeline needs of an inference server.

    The tasks are written against this and not against Backend. A
    summary needs a tokeniser, a context size and a way to ask once;
    it has no business being handed something that can also delete a
    KV dump, and a test of a prompt has no business starting an HTTP
    server to check it.
    """

    @property
    def role(self) -> str: ...

    def context_size(self) -> int: ...

    def count_tokens(self, text: str) -> int: ...

    def chat(
        self,
        prompt: str,
        n_predict: int,
        temperature: float | None = ...,
        schema: dict[str, Any] | None = ...,
    ) -> Completion: ...


class Conversational(Server, Protocol):
    """A server that can hold a conversation and offer tools.

    Separate from Server for the same reason Cache is: a summary has
    no business being handed something that can run a tool loop, and
    the protocol a function takes is the clearest statement of what it
    is allowed to do.
    """

    def converse(
        self,
        messages: list[dict[str, Any]],
        n_predict: int,
        temperature: float | None = ...,
        schema: dict[str, Any] | None = ...,
        tools: list[dict[str, Any]] | None = ...,
    ) -> Completion: ...


class Cache(Server, Protocol):
    """A server whose KV cache can be named, kept and dropped.

    Only sessions need this much; it is separate so that nothing else
    can reach the dumps by accident.
    """

    def slot_save(self, name: str) -> int: ...

    def slot_restore(self, name: str) -> int: ...

    def cached(self, name: str) -> bool: ...

    def forget(self, name: str) -> None: ...


class Backend:
    def __init__(
        self,
        endpoint: Endpoint,
        timeout: float = DEFAULT_TIMEOUT,
        temperature: float = 0.2,
    ):
        self.endpoint = endpoint
        self.timeout = timeout
        self.temperature = temperature

    @property
    def role(self) -> str:
        return self.endpoint.role

    def _request(
        self, path: str, payload: dict[str, Any] | None, timeout: float
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.endpoint.url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if data is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as answer:
                parsed = json.loads(answer.read())
        except urllib.error.HTTPError as error:
            body = error.read().decode(errors="replace")[:400]
            raise BackendError(
                f"shuttle-{self.role} {path}: HTTP {error.code}: {body}"
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise BackendError(
                f"shuttle-{self.role} {path}: {error}; check "
                f"'systemctl --user status shuttle-{self.role}.service'"
            ) from error
        if not isinstance(parsed, dict):
            raise BackendError(
                f"shuttle-{self.role} {path}: expected a JSON object, "
                f"got {type(parsed).__name__}"
            )
        return parsed

    def healthy(self) -> bool:
        try:
            self._request("/health", None, HEALTH_TIMEOUT)
        except BackendError:
            return False
        return True

    def props(self) -> dict[str, Any]:
        return self._request("/props", None, HEALTH_TIMEOUT)

    def context_size(self) -> int:
        settings = self.props().get("default_generation_settings", {})
        return int(settings.get("n_ctx", 0))

    def count_tokens(self, text: str) -> int:
        answer = self._request("/tokenize", {"content": text}, HEALTH_TIMEOUT)
        return len(answer.get("tokens", []))

    def chat(
        self,
        prompt: str,
        n_predict: int,
        temperature: float | None = None,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        """Ask once, through the template the model was trained on.

        Thinking is off: Qwen3 reasons before answering by default, and
        a budget spent on reasoning is a budget not spent on the answer.
        Left on, a short n_predict returns an empty string.
        """
        return self.converse(
            [{"role": "user", "content": prompt}],
            n_predict,
            temperature,
            schema,
        )

    def converse(
        self,
        messages: list[dict[str, Any]],
        n_predict: int,
        temperature: float | None = None,
        schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion:
        """A whole exchange, with the tools the model may ask for.

        What `chat` is built on. An agent loop needs the turns kept —
        what it asked, what the tools answered — because a model told
        only its own last sentence repeats it.
        """
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": n_predict,
            "temperature": (
                self.temperature if temperature is None else temperature
            ),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema},
            }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        answer = self._request("/v1/chat/completions", payload, self.timeout)
        try:
            choice = answer["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
        except (KeyError, IndexError) as error:
            raise BackendError(
                f"shuttle-{self.role}: unexpected answer shape: "
                f"{str(answer)[:200]}"
            ) from error
        usage = answer.get("usage", {})
        return Completion(
            content=content.strip(),
            tokens_in=int(usage.get("prompt_tokens", 0)),
            tokens_out=int(usage.get("completion_tokens", 0)),
            truncated=choice.get("finish_reason") == "length",
            calls=_calls(message.get("tool_calls") or []),
        )

    def slot_save(self, name: str) -> int:
        """Write slot zero's KV cache out; returns the tokens saved.

        The server names the file inside its own cache directory, so
        the caller gives a name rather than a path.
        """
        answer = self._request(
            "/slots/0?action=save", {"filename": name}, self.timeout
        )
        return int(answer.get("n_saved", 0))

    def slot_restore(self, name: str) -> int:
        """Read a KV cache back into slot zero; returns its tokens."""
        answer = self._request(
            "/slots/0?action=restore", {"filename": name}, self.timeout
        )
        return int(answer.get("n_restored", 0))

    def cached(self, name: str) -> bool:
        """Whether that dump is on disk where the server would find it."""
        cache = self.endpoint.cache
        return bool(cache and (cache / name).is_file())

    def forget(self, name: str) -> None:
        """Remove a dump. A cache nobody can clear is a leak."""
        cache = self.endpoint.cache
        if cache:
            (cache / name).unlink(missing_ok=True)
