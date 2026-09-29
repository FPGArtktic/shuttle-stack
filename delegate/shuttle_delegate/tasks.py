# SPDX-License-Identifier: GPL-3.0-only
"""The work itself, once a backend is in hand.

Every task takes text that has already been read from disk and returns
a plain dictionary, so the MCP layer above adds nothing but argument
checking and error wording.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .backend import Backend, Completion
from .chunking import chunk

# Qwen3 averages above this on prose and near it on dense code, so the
# estimate errs towards chunks that are smaller than they need to be.
CHARS_PER_TOKEN = 3

RESERVE_TOKENS = 600
MIN_BUDGET_TOKENS = 256
FIT_ATTEMPTS = 3
MAX_ROUNDS = 4
LABEL_TOKENS = 64

SUMMARY_MAP = """\
Summarise the passage below in at most {words} words.{focus}
Write plain prose. Add nothing the passage does not say.

PASSAGE:
{text}"""

SUMMARY_REDUCE = """\
Below are summaries of consecutive parts of one document. Write one
summary of the whole document in at most {words} words.{focus}
Do not mention that it arrived in parts.

PARTS:
{text}"""

CLASSIFY = """\
Assign exactly one of these labels to the text below: {labels}.{question}
Answer with the label only.

TEXT:
{text}"""

EXTRACT = """\
Extract the requested fields from the text below.{instructions}
Use only what the text says. Omit a field the text does not support.

TEXT:
{text}"""


class TaskError(RuntimeError):
    """The work cannot be done as asked, and why."""


@dataclass
class Work:
    """What the local servers spent, so the caller can see the saving."""

    calls: int = 0
    tokens: int = 0
    truncated: bool = False
    _answers: list[str] = field(default_factory=list)

    def record(self, answer: Completion) -> str:
        self.calls += 1
        self.tokens += answer.tokens
        self.truncated = self.truncated or answer.truncated
        return answer.content

    def report(self) -> dict:
        return {
            "local_calls": self.calls,
            "local_tokens": self.tokens,
            "truncated": self.truncated,
        }


def read_text(path: str) -> str:
    file = Path(path).expanduser()
    if not file.is_file():
        raise TaskError(f"{file}: not a readable file")
    text = file.read_text(errors="replace")
    if not text.strip():
        raise TaskError(f"{file}: holds no text to work on")
    return text


def fit(backend: Backend, text: str, n_predict: int) -> list[str]:
    """Cut the text into pieces the server will accept.

    The character estimate is checked against the server's own
    tokeniser and tightened if it was optimistic, so a file of dense
    tokens is not silently truncated by llama-server.
    """
    allowed = backend.context_size() - n_predict - RESERVE_TOKENS
    if allowed < MIN_BUDGET_TOKENS:
        raise TaskError(
            f"shuttle-{backend.role} has a context of "
            f"{backend.context_size()}, which leaves no room for the text "
            f"once {n_predict} tokens are reserved for the answer"
        )
    limit = allowed * CHARS_PER_TOKEN
    for _ in range(FIT_ATTEMPTS):
        chunks = chunk(text, limit)
        if not chunks:
            return []
        used = backend.count_tokens(max(chunks, key=len))
        if used <= allowed:
            return chunks
        limit = max(1, limit * allowed // used * 95 // 100)
    raise TaskError(
        "the text does not divide into pieces this server can hold; "
        "it is probably not text"
    )


def _clause(prefix: str, value: str) -> str:
    return f" {prefix} {value.strip()}" if value.strip() else ""


def summarise(
    backend: Backend, text: str, words: int = 200, focus: str = ""
) -> dict:
    """Summarise, folding the parts together until one remains."""
    if words < 10:
        raise TaskError(f"words must be at least 10, got {words}")
    n_predict = max(96, words * 3)
    focus_clause = _clause("Concentrate on:", focus)
    work = Work()
    chunks = fit(backend, text, n_predict)
    if not chunks:
        raise TaskError("nothing to summarise")
    rounds = 0
    while len(chunks) > 1:
        rounds += 1
        if rounds > MAX_ROUNDS:
            raise TaskError(
                f"still {len(chunks)} parts after {MAX_ROUNDS} rounds of "
                "folding; ask for a longer summary or a smaller file"
            )
        parts = [
            work.record(
                backend.chat(
                    SUMMARY_MAP.format(
                        words=words, focus=focus_clause, text=piece
                    ),
                    n_predict,
                )
            )
            for piece in chunks
        ]
        folded = fit(backend, "\n\n".join(parts), n_predict)
        if len(folded) >= len(chunks):
            raise TaskError(
                "the parts are not getting shorter; ask for a shorter "
                "summary with a smaller words value"
            )
        chunks = folded
    template = SUMMARY_MAP if rounds == 0 else SUMMARY_REDUCE
    summary = work.record(
        backend.chat(
            template.format(words=words, focus=focus_clause, text=chunks[0]),
            n_predict,
        )
    )
    return {
        "summary": summary,
        "rounds": rounds,
        "server": f"shuttle-{backend.role}",
    } | work.report()


def _decode(content: str, what: str) -> dict:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise TaskError(
            f"{what} did not come back as JSON: {content[:200]}"
        ) from error
    if not isinstance(value, dict):
        raise TaskError(f"{what} came back as {type(value).__name__}")
    return value


def classify(
    backend: Backend, text: str, labels: list[str], question: str = ""
) -> dict:
    """Put the text in one of the given labels, voting across chunks."""
    clean = [label.strip() for label in labels if label.strip()]
    if len(clean) < 2:
        raise TaskError(f"classify needs at least two labels, got {clean}")
    schema = {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": clean}},
        "required": ["label"],
    }
    prompt = CLASSIFY.replace("{labels}", ", ".join(clean))
    work = Work()
    votes: Counter[str] = Counter()
    chunks = fit(backend, text, LABEL_TOKENS)
    for piece in chunks:
        answer = work.record(
            backend.chat(
                prompt.format(
                    question=_clause("Consider:", question), text=piece
                ),
                LABEL_TOKENS,
                temperature=0.0,
                schema=schema,
            )
        )
        votes[_decode(answer, "the label")["label"]] += 1
    if not votes:
        raise TaskError("nothing to classify")
    label, count = votes.most_common(1)[0]
    return {
        "label": label,
        "agreement": round(count / sum(votes.values()), 3),
        "votes": dict(votes),
        "server": f"shuttle-{backend.role}",
    } | work.report()


def extract(
    backend: Backend, text: str, schema: dict, instructions: str = ""
) -> dict:
    """Pull structured fields out of a text that fits in one go."""
    if schema.get("type") != "object":
        raise TaskError("schema must be a JSON schema of type 'object'")
    n_predict = max(256, backend.context_size() // 8)
    chunks = fit(backend, text, n_predict)
    if not chunks:
        raise TaskError("nothing to extract from")
    if len(chunks) > 1:
        raise TaskError(
            f"the text needs {len(chunks)} pieces to fit shuttle-"
            f"{backend.role}, and fields cannot be merged across pieces "
            "without inventing precedence; summarise it first, or pass a "
            "smaller file"
        )
    work = Work()
    answer = work.record(
        backend.chat(
            EXTRACT.format(
                instructions=_clause("", instructions), text=chunks[0]
            ),
            n_predict,
            temperature=0.0,
            schema=schema,
        )
    )
    return {
        "fields": _decode(answer, "the fields"),
        "server": f"shuttle-{backend.role}",
    } | work.report()
