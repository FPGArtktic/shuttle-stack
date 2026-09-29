# SPDX-License-Identifier: GPL-3.0-only
"""The llama-server HTTP API, narrowed to what the tools need."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import Endpoint

HEALTH_TIMEOUT = 5.0
DEFAULT_TIMEOUT = 600.0


class BackendError(RuntimeError):
    """A server refused a request or did not answer."""


@dataclass(frozen=True)
class Completion:
    """One answer, with what it cost."""

    content: str
    tokens_in: int
    tokens_out: int

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out


class Backend:
    def __init__(self, endpoint: Endpoint, timeout: float = DEFAULT_TIMEOUT):
        self.endpoint = endpoint
        self.timeout = timeout

    @property
    def role(self) -> str:
        return self.endpoint.role

    def _request(
        self, path: str, payload: dict | None, timeout: float
    ) -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.endpoint.url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if data is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as answer:
                return json.loads(answer.read())
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

    def healthy(self) -> bool:
        try:
            self._request("/health", None, HEALTH_TIMEOUT)
        except BackendError:
            return False
        return True

    def props(self) -> dict:
        return self._request("/props", None, HEALTH_TIMEOUT)

    def context_size(self) -> int:
        settings = self.props().get("default_generation_settings", {})
        return int(settings.get("n_ctx", 0))

    def count_tokens(self, text: str) -> int:
        answer = self._request("/tokenize", {"content": text}, HEALTH_TIMEOUT)
        return len(answer.get("tokens", []))

    def complete(
        self,
        prompt: str,
        n_predict: int,
        temperature: float = 0.2,
        json_schema: dict | None = None,
    ) -> Completion:
        payload: dict = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "cache_prompt": True,
        }
        if json_schema is not None:
            payload["json_schema"] = json_schema
        answer = self._request("/completion", payload, self.timeout)
        timings = answer.get("timings", {})
        return Completion(
            content=answer.get("content", "").strip(),
            tokens_in=int(timings.get("prompt_n", 0)),
            tokens_out=int(timings.get("predicted_n", 0)),
        )
