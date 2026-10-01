# SPDX-License-Identifier: GPL-3.0-only
"""The work itself, once a backend is in hand.

Every task takes text that has already been read from disk and returns
a plain dictionary, so the MCP layer above adds nothing but argument
checking and error wording.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backend import Completion, Server
from .chunking import chunk
from .documents import is_document
from .documents import read as read_document
from .grounding import Grounding, check, fields_in_source
from .retrieval import CONTEXT_LINES, narrow

# Qwen3 averages above this on prose and near it on dense code, so the
# estimate errs towards chunks that are smaller than they need to be.
CHARS_PER_TOKEN = 3

RESERVE_TOKENS = 600
MIN_BUDGET_TOKENS = 256
FIT_ATTEMPTS = 3
MAX_ROUNDS = 4
LABEL_TOKENS = 64
# A retry is only worth making if it can come out differently, so the
# samples after the first are drawn warmer than the profile asks for.
# The first attempt keeps the profile's own temperature, which for
# schema-bound work is zero, so nothing is paid when it works.
RETRY_TEMPERATURE = 0.7
MAX_ATTEMPTS = 5
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

BRAINSTORM_WITH = """\
TEXT:
{text}

---
{request}"""

BRAINSTORM_ASK = """\
Suggest {count} different options for the request below. Draw on the
text above where it bears on the question and on your own judgement
where it does not. One short, concrete line each, no repetition.

REQUEST: {question}"""

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

    def record(self, answer: Completion) -> str:
        self.calls += 1
        self.tokens += answer.tokens
        self.truncated = self.truncated or answer.truncated
        return answer.content

    def report(self) -> dict[str, Any]:
        return {
            "local_calls": self.calls,
            "local_tokens": self.tokens,
            "truncated": self.truncated,
        }


def read_text(path: str) -> str:
    """A file as text, extracting it first if it is a document.

    Every tool here reads its input through this, so a PDF is accepted
    anywhere a path is, and arrives with its page markers intact.
    """
    file = Path(path).expanduser()
    if not file.is_file():
        raise TaskError(f"{file}: not a readable file")
    if is_document(file):
        return read_document(str(file))
    text = file.read_text(errors="replace")
    if not text.strip():
        raise TaskError(f"{file}: holds no text to work on")
    return text


def fit(backend: Server, text: str, n_predict: int) -> list[str]:
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
        # The longest chunk in characters is not the heaviest in
        # tokens: a table of numbers out-tokenises prose of twice its
        # length, and checking only the longest let the dense one
        # through to be truncated by the server.
        used = max(backend.count_tokens(piece) for piece in chunks)
        if used <= allowed:
            return chunks
        limit = max(1, limit * allowed // used * 95 // 100)
    raise TaskError(
        "the text does not divide into pieces this server can hold; "
        "it is probably not text"
    )


def narrowed(
    text: str, pattern: str, context: int, until: str = ""
) -> tuple[str, int]:
    """The regions of the text matching the pattern, or the text.

    An empty pattern means the whole text. A pattern matching nothing
    is an error rather than an empty read: a caller that mistyped a
    pattern should hear so, not receive an answer drawn from nowhere.
    """
    if not pattern:
        return text, 0
    found, matched = narrow(text, pattern, context, until)
    if not matched:
        raise TaskError(
            f"nothing in the file matches {pattern!r}; widen the "
            "pattern, or leave it out to read the whole file"
        )
    return found, matched


def temperatures(attempts: int) -> list[float | None]:
    """The temperature of each attempt, first one as configured."""
    if not 1 <= attempts <= MAX_ATTEMPTS:
        raise TaskError(
            f"attempts must be 1 to {MAX_ATTEMPTS}, got {attempts}"
        )
    later = [
        min(1.2, RETRY_TEMPERATURE + 0.2 * k) for k in range(attempts - 1)
    ]
    return [None, *later]


def _clause(prefix: str, value: str) -> str:
    return f" {prefix} {value.strip()}" if value.strip() else ""


def summarise(
    backend: Server, text: str, words: int = 200, focus: str = ""
) -> dict[str, Any]:
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
        if not folded:
            raise TaskError(
                "summarising the parts produced nothing; the server "
                "returned empty completions"
            )
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
    a part whose answer cannot be trusted either way. An empty
    completion is not an answer either, and counting one as an answer
    produced a found and verified result with nothing in it.
    """
    return bool(text.strip()) and MISSING not in text.upper()


def _parts(
    backend: Server,
    work: Work,
    chunks: list[str],
    question: str,
    words: str,
    n_predict: int,
    temperature: float | None,
) -> list[tuple[int, str]]:
    """What each part says, keeping which part said it.

    The index is kept because a retry must ask only the parts that
    answered: re-asking a part that refused, warmer, is exactly the
    sampling-until-something-comes-back this code refuses to do.
    """
    said = []
    for index, piece in enumerate(chunks):
        answer = work.record(
            backend.chat(
                ASK_ONE.format(question=question, words=words, text=piece),
                n_predict,
                temperature=temperature,
            )
        )
        if _answered(answer):
            said.append((index, answer))
    return said


def _join(
    backend: Server,
    work: Work,
    answers: list[str],
    question: str,
    words: str,
    n_predict: int,
    temperature: float | None = None,
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
                    temperature=temperature,
                )
            )
            for piece in cut
        ]
        rounds += 1
        kept = [piece for piece in folded if piece.strip()]
        if not kept:
            raise TaskError(
                "folding the part answers produced nothing; the server "
                "returned empty completions"
            )
        if len(kept) >= len(pieces) or rounds > MAX_ROUNDS:
            raise TaskError(
                "the part answers are not combining into one; ask a "
                "narrower question or a shorter answer"
            )
        pieces = kept
    return pieces[0]


def ask(
    backend: Server,
    text: str,
    question: str,
    words: int = 200,
    pattern: str = "",
    context: int = CONTEXT_LINES,
    until: str = "",
    attempts: int = 2,
) -> dict[str, Any]:
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
    text, matched = narrowed(text, pattern, context, until)
    n_predict = max(96, words * 3)
    limit = f" Answer in at most {words} words."
    work = Work()
    chunks = fit(backend, text, n_predict)
    if not chunks:
        raise TaskError("nothing to read")
    # The model saw `text` and nothing else, so that is what a
    # quotation in its answer has to have come from. An answer quoting
    # something that is not there is drawn again, warmer, because the
    # citation is what makes the answer checkable at all.
    #
    # Which parts refused is decided once, on the first and coldest
    # draw, and a retry asks only the parts that answered. Re-asking a
    # part that refused would be drawing again until something came
    # back, which rewards the model for inventing an answer; and a
    # warmer draw that refuses must not be able to throw away the
    # answer an earlier one produced.
    said: dict[int, str] = {}
    best: tuple[str, Grounding] | None = None
    tried = 0
    for temperature in temperatures(attempts):
        tried += 1
        asking = sorted(said) if said else list(range(len(chunks)))
        fresh = _parts(
            backend,
            work,
            [chunks[i] for i in asking],
            question,
            limit,
            n_predict,
            temperature,
        )
        said.update({asking[position]: answer for position, answer in fresh})
        if not said:
            break
        answer = _join(
            backend,
            work,
            [said[i] for i in sorted(said)],
            question,
            limit,
            n_predict,
            temperature,
        )
        grounded = check(answer, text)
        # The best sample is kept rather than the last: a warmer draw
        # can quote worse than the one before it.
        if best is None or len(grounded.missing) < len(best[1].missing):
            best = (answer, grounded)
        if grounded.ok:
            break
    report = {
        "parts": len(chunks),
        "parts_answering": len(said),
        "attempts": tried,
        "server": f"shuttle-{backend.role}",
    } | work.report()
    if pattern:
        report["matched_regions"] = matched
    if best is None:
        return {
            "answer": "",
            "found": False,
            "verified": True,
            "note": "no part of the file answers this question",
        } | report
    answer, grounded = best
    result = (
        {"answer": answer, "found": True, "verified": grounded.ok}
        | report
        | grounded.report()
    )
    if len(said) > 1:
        # A part that holds nothing may answer anyway rather than use
        # the sentinel, and the fold then blends it with a real answer.
        # The parts are returned so the caller can see that happen.
        result["said_by_part"] = [said[i] for i in sorted(said)]
    return result


def brainstorm(
    backend: Server,
    text: str,
    question: str,
    count: int = 5,
    pattern: str = "",
    context: int = CONTEXT_LINES,
    until: str = "",
) -> dict[str, Any]:
    """Ask for several options, and say that they are only options.

    This is the one task here with nothing to verify against. The
    schema fixes the shape of the answer, so a caller always gets a
    list of strings, and the result says plainly that the strings are
    suggestions rather than findings. Nothing in them has been checked
    against anything, and the caller is the verifier.
    """
    if not question.strip():
        raise TaskError("brainstorm needs a request")
    if not 1 <= count <= 20:
        raise TaskError(f"count must be 1 to 20, got {count}")
    matched = 0
    if text.strip():
        text, matched = narrowed(text, pattern, context, until)
    ask = BRAINSTORM_ASK.format(count=count, question=question.strip())
    prompt = (
        BRAINSTORM_WITH.format(text=text, request=ask) if text.strip() else ask
    )
    schema = {
        "type": "object",
        "properties": {
            "ideas": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["ideas"],
    }
    work = Work()
    answer = work.record(
        backend.chat(prompt, max(256, count * 80), schema=schema)
    )
    ideas = _decode(answer, "the ideas").get("ideas", [])
    if not isinstance(ideas, list) or not ideas:
        raise TaskError(f"no ideas came back: {answer[:200]}")
    result = {
        "ideas": [str(idea).strip() for idea in ideas],
        "grounded": False,
        "note": "suggestions, not findings; nothing here was verified",
        "server": f"shuttle-{backend.role}",
    } | work.report()
    if pattern:
        result["matched_regions"] = matched
    return result


def _decode(content: str, what: str) -> dict[str, Any]:
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
    backend: Server, text: str, labels: list[str], question: str = ""
) -> dict[str, Any]:
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
    backend: Server,
    text: str,
    schema: dict[str, Any],
    instructions: str = "",
    pattern: str = "",
    context: int = CONTEXT_LINES,
    until: str = "",
    attempts: int = 2,
) -> dict[str, Any]:
    """Pull structured fields out of a text that fits in one go.

    A pattern narrows the file first, which is what makes this usable
    on a file larger than the context: the fields are read from the
    matching regions rather than from a refusal.

    The server holds the answer to the schema, so its shape is never
    wrong; its content can be. A string value the text does not
    contain was not read out of it, and that is the verifier here. An
    answer that fails it is drawn again, warmer, up to `attempts`
    times. The first attempt uses the profile's own temperature, so
    nothing is paid when it passes, which is the usual case.
    """
    if schema.get("type") != "object":
        raise TaskError("schema must be a JSON schema of type 'object'")
    text, matched = narrowed(text, pattern, context, until)
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
    prompt = EXTRACT.format(
        instructions=_clause("", instructions), text=chunks[0]
    )
    work = Work()
    fields: dict[str, Any] = {}
    missing: list[str] = []
    tried = 0
    for temperature in temperatures(attempts):
        tried += 1
        answer = work.record(
            backend.chat(
                prompt, n_predict, temperature=temperature, schema=schema
            )
        )
        fields = _decode(answer, "the fields")
        missing = fields_in_source(fields, chunks[0])
        if not missing:
            break
    result = {
        "fields": fields,
        "attempts": tried,
        "verified": not missing,
        "server": f"shuttle-{backend.role}",
    } | work.report()
    if missing:
        result["values_not_in_source"] = missing
    if pattern:
        result["matched_regions"] = matched
    return result
