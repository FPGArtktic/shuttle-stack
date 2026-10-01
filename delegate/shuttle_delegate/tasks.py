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
from .grounding import check
from .retrieval import CONTEXT_LINES, narrow

# Qwen3 averages above this on prose and near it on dense code, so the
# estimate errs towards chunks that are smaller than they need to be.
CHARS_PER_TOKEN = 3

RESERVE_TOKENS = 600
MIN_BUDGET_TOKENS = 256
FIT_ATTEMPTS = 3
MAX_ROUNDS = 4
LABEL_TOKENS = 64
MISSING = "NOT IN THIS TEXT"

# llama-server reuses the KV cache of a prompt prefix it has already
# seen, so every template puts the text first and the instruction last.
# A second question about the same text then skips re-reading it:
# measured over 12 KB of the installer, 13.6 s against 7.1 s.
SUMMARY_MAP = """\
PASSAGE:
{text}

---
Summarise the passage above in at most {words} words.{focus}
Write plain prose. Add nothing the passage does not say."""

SUMMARY_REDUCE = """\
PARTS:
{text}

---
Above are summaries of consecutive parts of one document. Write one
summary of the whole document in at most {words} words.{focus}
Do not mention that it arrived in parts."""

ASK_ONE = """\
TEXT:
{text}

---
Answer the question below using only the text above.{words}
Quote the words from the text that support the answer.

If the text does not contain the answer, reply with exactly this and
nothing else: NOT IN THIS TEXT
Do not answer from anything you know outside the text.

QUESTION: {question}"""

ASK_JOIN = """\
ANSWERS:
{text}

---
Above are answers to one question, each taken from a different part of
one document. Write a single answer from them.{words}
Do not mention that they arrived in parts.

QUESTION: {question}"""

CLASSIFY = """\
TEXT:
{text}

---
Assign exactly one of these labels to the text above: {labels}.{question}
Answer with the label only."""

EXTRACT = """\
TEXT:
{text}

---
Extract the requested fields from the text above.{instructions}
Use only what the text says. Omit a field the text does not support."""


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


def narrowed(text: str, pattern: str, context: int) -> tuple[str, int]:
    """The regions of the text matching the pattern, or the text.

    An empty pattern means the whole text. A pattern matching nothing
    is an error rather than an empty read: a caller that mistyped a
    pattern should hear so, not receive an answer drawn from nowhere.
    """
    if not pattern:
        return text, 0
    found, matched = narrow(text, pattern, context)
    if not matched:
        raise TaskError(
            f"nothing in the file matches {pattern!r}; widen the "
            "pattern, or leave it out to read the whole file"
        )
    return found, matched


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


def _answered(text: str) -> bool:
    """Whether a part claims to hold the answer.

    The sentinel is asked for verbatim, and any appearance of it is
    read as a refusal: a part that mentions it while also answering is
    a part whose answer cannot be trusted either way.
    """
    return MISSING not in text.upper()


def _join(
    backend: Backend,
    work: Work,
    answers: list[str],
    question: str,
    words: str,
    n_predict: int,
) -> str:
    """Fold several part-answers into one, in bounded rounds."""
    pieces = answers
    rounds = 0
    while len(pieces) > 1:
        cut = fit(backend, "\n\n".join(pieces), n_predict)
        folded = [
            work.record(
                backend.chat(
                    ASK_JOIN.format(
                        question=question, words=words, text=piece
                    ),
                    n_predict,
                )
            )
            for piece in cut
        ]
        rounds += 1
        if len(folded) >= len(pieces) or rounds > MAX_ROUNDS:
            raise TaskError(
                "the part answers are not combining into one; ask a "
                "narrower question or a shorter answer"
            )
        pieces = folded
    return pieces[0]


def ask(
    backend: Backend,
    text: str,
    question: str,
    words: int = 200,
    pattern: str = "",
    context: int = CONTEXT_LINES,
) -> dict:
    """Answer a question from a file, part by part if it is long.

    A pattern narrows the file first. Locating a passage is exact work
    at which a model is poor: asked over a whole 35 KB script the
    answer came back wrong with an invented quote, and asked over the
    one matching function it came back right, for a thirtieth of the
    tokens and in a fifteenth of the time.
    """
    if not question.strip():
        raise TaskError("ask needs a question")
    if words < 10:
        raise TaskError(f"words must be at least 10, got {words}")
    text, matched = narrowed(text, pattern, context)
    n_predict = max(96, words * 3)
    limit = f" Answer in at most {words} words."
    work = Work()
    chunks = fit(backend, text, n_predict)
    if not chunks:
        raise TaskError("nothing to read")
    answers = [
        answer
        for answer in (
            work.record(
                backend.chat(
                    ASK_ONE.format(question=question, words=limit, text=piece),
                    n_predict,
                )
            )
            for piece in chunks
        )
        if _answered(answer)
    ]
    report = {
        "parts": len(chunks),
        "parts_answering": len(answers),
        "server": f"shuttle-{backend.role}",
    } | work.report()
    if pattern:
        report["matched_regions"] = matched
    if not answers:
        return {
            "answer": "",
            "found": False,
            "note": "no part of the file answers this question",
        } | report
    answer = _join(backend, work, answers, question, limit, n_predict)
    # The model saw `text` and nothing else, so that is what a
    # quotation in its answer has to have come from.
    result = (
        {"answer": answer, "found": True}
        | (report | work.report())
        | check(answer, text).report()
    )
    if len(answers) > 1:
        # A part that holds nothing may answer anyway rather than use
        # the sentinel, and the fold then blends it with a real answer.
        # The parts are returned so the caller can see that happen.
        result["said_by_part"] = answers
    return result


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
    backend: Backend,
    text: str,
    schema: dict,
    instructions: str = "",
    pattern: str = "",
    context: int = CONTEXT_LINES,
) -> dict:
    """Pull structured fields out of a text that fits in one go.

    A pattern narrows the file first, which is what makes this usable
    on a file larger than the context: the fields are read from the
    matching regions rather than from a refusal.
    """
    if schema.get("type") != "object":
        raise TaskError("schema must be a JSON schema of type 'object'")
    text, matched = narrowed(text, pattern, context)
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
            schema=schema,
        )
    )
    result = {
        "fields": _decode(answer, "the fields"),
        "server": f"shuttle-{backend.role}",
    } | work.report()
    if pattern:
        result["matched_regions"] = matched
    return result
