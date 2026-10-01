# SPDX-License-Identifier: GPL-3.0-only
"""The MCP surface: argument checking, wording, and nothing else."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import (
    answers,
    audit,
    cascade,
    code,
    digest,
    indexing,
    profiles,
    runs,
    sessions,
    tasks,
)
from .backend import Backend, BackendError
from .cascade import CascadeError
from .config import ConfigError, load_endpoints
from .documents import DocumentError
from .indexing import IndexingError
from .jobs import JobError, Queue
from .profiles import ProfileError
from .retrieval import PatternError
from .sessions import SessionError
from .tasks import TaskError

DEFAULT_PROFILE = "long"

# Failures the caller can do something about: a path that is not there,
# a pattern that matches nothing, a server that is down.  They reach the
# model as a message rather than as a crash it cannot read.
#
# Every entry is a type something here raises deliberately. A bare
# ValueError was in this list and had to come out: a programming
# mistake raises one too, so a bug in the delegate reached the model
# dressed as a failure it could do something about.
EXPECTED = (
    TaskError,
    IndexingError,
    DocumentError,
    CascadeError,
    SessionError,
    JobError,
    ProfileError,
    BackendError,
    ConfigError,
    PatternError,
    OSError,
)

INSTRUCTIONS = """\
SHUTTLE hands bulk text work to two llama-servers on this machine, so
that the text itself never enters your context.

A path may be a PDF. It is extracted in a container that has no network
and sees only that document's directory, using its text layer where it
has one and OCR only for the pages without, and the text arrives marked
with page numbers so an answer can say which page it came from.

Pass a file path, not the file's contents. The delegate reads the file,
splits it if it does not fit the server's context, and returns only the
result. Reading a file yourself and pasting it into a tool call spends
the tokens these tools exist to save.

shuttle-long holds the larger model and answers better; shuttle-fast
holds a smaller one and answers sooner. Every answer reports the tokens
the local servers spent, which are the tokens you did not.

Give ask_file a `pattern` whenever you can name what you are looking
for. It does not change what the model can answer, but it changes the
price: the same question over one matching function instead of a whole
35 KB file costs 408 tokens rather than 13079, and three seconds rather
than twenty-two. Repeated questions about the same text are cheaper
again, because the server keeps the cache of a prefix it has seen.
"""

mcp = MCPServer(name="shuttle", instructions=INSTRUCTIONS)

_backends: dict[str, Backend] = {}


def servers() -> dict[str, Backend]:
    """One connection per role, made on first use and then kept."""
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


# The caller gets a report under a hard token limit and the name of the
# file holding the rest; every call leaves a line in the audit log,
# whether it answered or failed.
def perform(
    name: str, call: Callable[..., dict[str, Any]], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Do the work, cap the report, write the run, log the line.

    One place, because a job has to be recorded the same way as the
    call that could afford to wait for its answer.
    """
    started = time.time()
    answered = False
    try:
        result = call(**kwargs)
        answered = True
        # The profile that answered, not the one asked for: a cascade
        # climbs, and capping the report on a server that did no work
        # contacted the wrong tokeniser and applied the wrong limit.
        chosen = profiles.get(
            result.get("profile") or kwargs.get("profile") or DEFAULT_PROFILE
        )
        counted = runs.report(
            servers()[chosen.server].count_tokens,
            name,
            kwargs,
            result,
            chosen.report_tokens,
        )
    except Exception as error:
        # Writing the run file and counting the report both happen after
        # the model has worked, and both can fail. Without this the call
        # left no audit line at all, which the house rule forbids.
        audit.log(
            {
                "tool": name,
                "args": kwargs,
                "ok": False,
                "answered": answered,
                "seconds": round(time.time() - started, 2),
                "error": str(error),
            }
        )
        raise
    audit.log(
        {
            "tool": name,
            "args": kwargs,
            "ok": True,
            "profile": chosen.name,
            "seconds": round(time.time() - started, 2),
            "local_tokens": result.get("local_tokens"),
            "run": counted["run"],
        }
    )
    return counted


# Anything recorded can also be run as a job: the two paths differ only
# in who waits.
JOBABLE: dict[str, Callable[..., dict[str, Any]]] = {}
QUEUE = Queue()


def recorded(
    fn: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    JOBABLE[fn.__name__] = fn

    @functools.wraps(fn)
    def wrapper(**kwargs: Any) -> dict[str, Any]:
        return perform(fn.__name__, fn, kwargs)

    return wrapper


def as_job(tool: str) -> Callable[..., dict[str, Any]]:
    inner = JOBABLE[tool]

    def run(**kwargs: Any) -> dict[str, Any]:
        return perform(tool, inner, kwargs)

    return run


def backend(profile: str = DEFAULT_PROFILE) -> Backend:
    """The server a profile names, sampled the way it asks for."""
    chosen = profiles.get(profile)
    server = servers()[chosen.server]
    if server.temperature == chosen.temperature:
        return server
    return Backend(server.endpoint, server.timeout, chosen.temperature)


def _describe(server: Backend) -> dict[str, Any]:
    entry: dict[str, Any] = {
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
def status() -> dict[str, Any]:
    # The one tool that reports a failure instead of raising it. Its
    # job is to say what is wrong, and an uninstalled stack is the
    # answer to that question rather than an obstacle to answering it.
    try:
        return {"servers": [_describe(b) for b in servers().values()]}
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
@recorded
def summarize_file(
    path: str,
    words: int = 200,
    focus: str = "",
    profile: str = "long",
) -> dict[str, Any]:
    return tasks.summarise(
        backend(profile), tasks.read_text(path), words, focus
    )


@mcp.tool(
    description="Answer a question about a file on this machine. Give "
    "the path, not the contents. Pass a regexp as `pattern` whenever "
    "you can name what you are looking for, and widen `context` until "
    "the region covers a whole function or section: the same question "
    "over one matching function rather than a whole 35 KB file costs "
    "408 tokens instead of 13079 and answers in three seconds instead "
    "of twenty-two. Give `until` as well whenever the passage has an "
    "end you can name, such as `^[a-z_]+\\(\\)` for the next shell "
    "function or `^## ` for the next section: the region then stops "
    "there instead of running on for `context` lines into whatever "
    "follows, which otherwise gets you an answer about the neighbour. "
    "Asking several questions about the same region is cheaper still. "
    "Without a pattern the whole file is read in parts "
    "and the parts that answer are combined. If nothing in the file "
    "answers, the reply says so rather than inventing one."
)
@anticipated
@recorded
def ask_file(
    path: str,
    question: str,
    pattern: str = "",
    until: str = "",
    context: int = 12,
    words: int = 200,
    attempts: int = 2,
    cascade_profiles: str = "",
    profile: str = "long",
) -> dict[str, Any]:
    # A question asked again is answered from the index without the
    # server, which here is the difference between tens of seconds and
    # a tenth of one. Only a verified answer is ever kept, and only a
    # near-exact repeat is served; see answers.MAX_DISTANCE for what
    # that cost and why it is not looser.
    kept = answers.held(path, question)
    if kept is not None:
        return kept
    text = tasks.read_text(path)

    def once(chosen: str) -> dict[str, Any]:
        return tasks.ask(
            backend(chosen),
            text,
            question,
            words,
            pattern,
            context,
            until,
            attempts,
        )

    if cascade_profiles:
        result = cascade.climb(cascade.ladder(cascade_profiles), once)
    else:
        result = once(profile)
    # A narrowed answer is not an answer to the question on its own:
    # it was read from the region the pattern matched, and the next
    # caller may pass a different one.
    if not pattern:
        answers.keep(path, question, result)
    return result


@mcp.tool(
    description="Put a file into one of the labels you give. The label "
    "is constrained by a schema, so the answer is always one of them. "
    "A file longer than the context is classified in parts and the "
    "parts vote; the answer carries the agreement between them. "
    "shuttle-fast is usually enough for this."
)
@anticipated
@recorded
def classify_file(
    path: str,
    labels: list[str],
    question: str = "",
    profile: str = "extract",
) -> dict[str, Any]:
    return tasks.classify(
        backend(profile), tasks.read_text(path), labels, question
    )


@mcp.tool(
    description="Pull structured fields out of a file, following a JSON "
    "schema you supply; the server is held to the schema, so the shape "
    "of the answer is guaranteed, though its content is not: a string "
    "value the text does not contain was not read out of it, and such "
    "an answer is drawn again, warmer, up to `attempts` times. What "
    "still fails is listed in `values_not_in_source` rather than "
    "passed off as read. Pass a `pattern` to read the fields "
    "from one region of a large file; without one the file must fit "
    "the context in a single piece, because fields cannot be merged "
    "across parts without inventing a precedence, and a file too "
    "large is refused rather than guessed at."
)
@anticipated
@recorded
def extract(
    path: str,
    schema: dict[str, Any],
    instructions: str = "",
    pattern: str = "",
    until: str = "",
    context: int = 12,
    attempts: int = 2,
    cascade_profiles: str = "",
    profile: str = "extract",
) -> dict[str, Any]:
    text = tasks.read_text(path)

    def once(chosen: str) -> dict[str, Any]:
        return tasks.extract(
            backend(chosen),
            text,
            schema,
            instructions,
            pattern,
            context,
            until,
            attempts,
        )

    if not cascade_profiles:
        return once(profile)
    return cascade.climb(cascade.ladder(cascade_profiles), once)


@mcp.tool(
    description="Run one of the file tools in the background and return "
    "a job id at once. Use this when the file is large: a summary of a "
    "35 KB script took four minutes here, and a client that waits that "
    "long gives up before the server answers. `tool` is the name of the "
    "tool to run and `arguments` the dictionary you would have passed "
    "it. Poll with get_status and collect with get_result."
)
@anticipated
def start_job(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if tool not in JOBABLE:
        raise JobError(
            f"{tool!r} cannot be run as a job; the ones that can are "
            f"{sorted(JOBABLE)}"
        )
    return QUEUE.start(tool, as_job(tool), arguments).report()


@mcp.tool(
    description="Say whether a job is queued, running, done or failed, "
    "and how long it has waited and run. A job that failed carries the "
    "reason here; get_result would only raise it again."
)
@anticipated
def get_status(job: str) -> dict[str, Any]:
    return QUEUE.status(job)


@mcp.tool(
    description="Collect a finished job's answer, in the same shape the "
    "tool would have returned had you waited for it, with the run file "
    "and the tokens it spent. A job still queued or running is an error "
    "rather than an empty answer; ask get_status first."
)
@anticipated
def get_result(job: str) -> dict[str, Any]:
    return QUEUE.collect(job)


@mcp.tool(
    description="Ask the local model for several options rather than "
    "for a fact. Give a `path` to have a file considered, or leave it "
    "empty to ask from nothing. Everything else here answers from a "
    "text and can be checked against it; this cannot, and says so: the "
    "schema fixes the shape of the answer, the content is unverified, "
    "and you are the one who judges it. Do not treat an idea from here "
    "as something the file says."
)
@anticipated
@recorded
def brainstorm(
    question: str,
    path: str = "",
    count: int = 5,
    pattern: str = "",
    until: str = "",
    context: int = 12,
    profile: str = "brainstorm",
) -> dict[str, Any]:
    return tasks.brainstorm(
        backend(profile),
        tasks.read_text(path) if path else "",
        question,
        count,
        pattern,
        context,
        until,
    )


@mcp.tool(
    description="Read a document once and keep its cache under a name, "
    "so later questions about it answer quickly. Give the same `path`, "
    "`pattern` and `until` you would have given ask_file, and a `words` "
    "limit that holds for every answer in the session. Worth it when "
    "you expect more than one or two questions about the same text: on "
    "this machine a 4000-token document costs ten seconds to read and "
    "the questions after it answer in six rather than eleven."
)
@anticipated
def session_open(
    path: str,
    words: int = 200,
    pattern: str = "",
    until: str = "",
    context: int = 12,
    profile: str = "long",
) -> dict[str, Any]:
    return sessions.open_session(
        backend(profile), profile, path, words, pattern, until, context
    )


@mcp.tool(
    description="Ask an open session a question. The server is the one "
    "the session was opened on; you do not choose it here. The document "
    "is not "
    "sent again; its cache is restored instead. If the file has changed "
    "since the session was opened the cache is thrown away rather than "
    "trusted, the answer comes from the new text, and the reply says so "
    "in `source_changed`. Quotations are checked against the document "
    "exactly as in ask_file."
)
@anticipated
def session_ask(session: str, question: str) -> dict[str, Any]:
    return sessions.ask(backend, session, question)


@mcp.tool(
    description="Drop a session's cache and keep its transcript. The "
    "dump is 305 MiB on this machine, so a session left open is disk "
    "spent on a document nobody is asking about. The transcript under "
    ".shuttle/sessions survives and holds every question and answer."
)
@anticipated
def session_close(session: str) -> dict[str, Any]:
    return sessions.close(backend, session)


@mcp.tool(
    description="Index a file so later questions can find the right "
    "part of it without anyone naming a pattern. A PDF or text "
    "document is cut into sections, found by heading — a numbered "
    "clause or a Markdown heading — and by page where it has none. "
    "Source is cut into the units its language defines: a module, an "
    "entity, a do_install, a device tree node, a function. The reply "
    "says which happened and how the cutting went. Indexing the same "
    "file again replaces what it contributed rather than doubling it."
)
@anticipated
def index_path(
    path: str,
    heading: str = indexing.HEADING,
) -> dict[str, Any]:
    name = Path(path).name
    if code.is_source(path):
        cut = code.units(path)
        lines = len(cut.units) and max(u.last_line for u in cut.units)
        with closing(code.connect()) as db:
            stored = code.store(db, name, cut, lines)
        answer = {
            "file": name,
            "index": str(indexing.index_file()),
            "language": cut.language,
            "units": stored,
            "cut_by": cut.how,
        }
        if cut.why:
            answer["why"] = cut.why
        return answer
    text = tasks.read_text(path)
    sections, how = indexing.split(text, heading)
    with closing(indexing.connect()) as db:
        stored = indexing.store(db, name, sections)
    return {
        "file": name,
        "index": str(indexing.index_file()),
        "sections": stored,
        "sections_from": how,
        "pages": max((s.page for s in sections), default=0),
    }


@mcp.tool(
    description="Search the indexed documents and get back a few "
    "sections with the file, the page and the heading each came from. "
    "Both halves of the index are asked: BM25 for a register name or "
    "a clause number, vectors for a question phrased in words the "
    "document may not use, and the two rankings are fused. The reply "
    "says which half found each hit. The citation is the point: an "
    "answer that names its page can be checked, one that does not "
    "cannot."
)
@anticipated
def search_docs(
    question: str,
    limit: int = 3,
    profile: str = DEFAULT_PROFILE,
) -> dict[str, Any]:
    return indexing.search(question, limit, backend(profile).count_tokens)


@mcp.tool(
    description="Search the indexed source and get back a few units "
    "with the file and the lines each occupies. Use it for a name — a "
    "module, a task, a function, a node — and for a question about "
    "what something does; both halves of the index are asked and the "
    "reply says which found each hit. The lines are the point: the "
    "answer is a place to open, not a file to read."
)
@anticipated
def search_code(
    question: str,
    limit: int = 3,
    profile: str = DEFAULT_PROFILE,
) -> dict[str, Any]:
    return code.search(question, limit, backend(profile).count_tokens)


@mcp.tool(
    description="List what is indexed: documents with how many "
    "sections and pages each contributed, source with its language "
    "and how it was cut, and the files that have kept answers. Call "
    "it to find out whether something is worth indexing again, or at "
    "all."
)
@anticipated
def list_indexed() -> dict[str, Any]:
    with closing(code.connect()) as db:
        answers.prepare(db)
        said = digest.of_documents(db)
        held = indexing.indexed(db)
        for one in held["documents"]:
            if one["file"] in said:
                one["summary"] = said[one["file"]]
        if said:
            held["summaries"] = digest.CAVEAT
        return (
            held | {"sources": code.indexed(db)["sources"]} | answers.kept(db)
        )
