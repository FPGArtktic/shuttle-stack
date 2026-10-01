# SPDX-License-Identifier: GPL-3.0-only
"""A bounded loop: the model asks for tools until something verifies.

The rule the design puts above everything else here is that a local
agent runs only where a verifier exists. A loop that decides for
itself when it is finished is a loop that reports success, and an 8B
model is confident in exactly the way that makes that dangerous. So a
preset without a verifier is refused rather than run: either a schema
the report has to satisfy, or a command that has to exit zero, or
both.

The loop is bounded three ways, because each runs out first in a
different kind of failure. Steps catch a model asking for the same
tool forever. Tokens catch one writing an essay at every step. Seconds
catch a verifier that hangs. Whichever is hit, the run stops and says
which — a budget that is silently exceeded is not a budget.

What the model may do is an allowlist, not a denylist. It reads the
index and it writes into one directory; it cannot reach the network,
cannot see a path nobody indexed, and cannot run a command that the
preset did not name. The verifier runs in a container of its own,
because a verifier command is arbitrary code from a configuration
file and running it on the host would undo the rest.

Every step is written to a transcript as it happens, so a run that is
killed is still a record of how far it got.
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import subprocess
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import code, grounding, indexing
from .backend import Call, Conversational, Server
from .runs import home, new_id

NAME = re.compile(r"^[a-z][a-z0-9-]{1,31}$")
# Where presets beyond the built-in ones are read from.
FILE = "agents.toml"
# What a step may be told, so that one tool's answer cannot fill the
# context on its own. Measured against the report limit rather than
# chosen: an answer longer than the report the whole run is allowed
# cannot be a step towards it.
STEP_CHARS = 4000
# The reply the loop gives back. The run directory holds everything.
REPORT_TOKENS = 1000
# Lines one read may return. A model asked for 1 to 40 and meant "the
# file"; one asking for 1 to 40000 has lost track and should be told.
MAX_LINES = 200
# How many times the same call may fail with the same arguments before
# the run is stopped. A model told "the report is missing places" and
# handed back the identical report will hand it back forever: five of
# ten steps went that way, and a loop that cannot move is better
# stopped than left to spend its budget.
MAX_REPEATS = 2


class AgentError(RuntimeError):
    """The run cannot be made as asked, and why."""


@dataclass(frozen=True)
class Preset:
    """One bounded way of working, named in configuration."""

    name: str
    profile: str = "long"
    tools: tuple[str, ...] = ()
    task: str = ""
    max_steps: int = 12
    max_tokens: int = 60000
    max_seconds: float = 900.0
    schema: dict[str, Any] | None = None
    verifier: tuple[str, ...] = ()
    image: str = ""
    # The report fields that must have come from a tool answer. Named
    # rather than all of them: grounding the whole report meant the
    # prose field had to appear verbatim in the tool output, which it
    # never will, so a run that had found the right lines was told it
    # had not and spent the rest of its budget resubmitting them.
    ground: tuple[str, ...] = ()

    def verified(self) -> None:
        """Refuse a preset that cannot tell success from failure."""
        if not (self.schema or self.verifier or self.ground):
            raise AgentError(
                f"{self.name}: no verifier. A preset needs a report "
                "schema, a citation-grounding check or a verifier "
                "command; without one the loop would decide for itself "
                "that it had finished, and the task belongs with Claude"
            )
        if self.verifier and not self.image:
            raise AgentError(
                f"{self.name}: a verifier command needs an image to run "
                "in. It is arbitrary code from a configuration file and "
                "is not run on the host"
            )


@dataclass
class Step:
    """One turn of the loop, as the transcript records it."""

    at: int
    tool: str
    arguments: dict[str, Any]
    answer: str
    seconds: float
    ok: bool = True
    # Whether the model called the tool or wrote the call out as its
    # message. Both are accepted; the difference is recorded.
    written: bool = False


@dataclass
class Run:
    """One agent run, and what it spent."""

    id: str
    preset: str
    task: str
    steps: list[Step] = field(default_factory=list)
    tokens: int = 0
    seconds: float = 0.0
    report: dict[str, Any] = field(default_factory=dict)
    verified: bool = False
    stopped: str = ""
    checks: list[str] = field(default_factory=list)

    @property
    def directory(self) -> Path:
        return home() / "agents" / self.id

    def summary(self) -> dict[str, Any]:
        return {
            "run": self.id,
            "preset": self.preset,
            "verified": self.verified,
            "stopped": self.stopped,
            "steps": len(self.steps),
            "local_tokens": self.tokens,
            "seconds": round(self.seconds, 1),
            "tools_used": sorted({one.tool for one in self.steps}),
            "checks": self.checks,
            "report": self.report,
            "output": str(self.directory),
        }


# --- the tools an agent may ask for ----------------------------------
#
# A subset of what the delegate offers, shaped for a model rather than
# for Claude: no profiles, no budgets, no cascade. Each one answers
# with text, because a tool result is a message and the model reads it
# as one.

SCHEMAS: dict[str, dict[str, Any]] = {
    "search_docs": {
        "description": "Search the indexed documents. Returns sections "
        "with the file and page each came from.",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
    "search_code": {
        "description": "Search the indexed source. Returns units with "
        "the file and the lines each occupies.",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
    "find_references": {
        "description": "Every place in the indexed source that names "
        "something, as file:line. Use it for a module, a task, a "
        "function or a variable.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "relation": {
                    "type": "string",
                    "description": "optional: instantiates, includes, "
                    "calls, uses, depends, provides, assigns",
                },
            },
            "required": ["name"],
        },
    },
    "expand_symbol": {
        "description": "One step further than find_references: where a "
        "name is defined, who names it, and what those places refer to "
        "in turn, each with a file:line. Use it when the answer is one "
        "hop past the name itself.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "hops": {
                    "type": "integer",
                    "description": "1 for the name alone, 2 to follow "
                    "what it reaches. Default 2.",
                },
            },
            "required": ["name"],
        },
    },
    "read_lines": {
        "description": "Read a range of lines from an indexed source "
        "file. Name the file as find_references reported it.",
        "parameters": {
            "type": "object",
            "properties": {
                "file": {"type": "string"},
                "first": {"type": "integer"},
                "last": {"type": "integer"},
            },
            "required": ["file", "first", "last"],
        },
    },
    "write_file": {
        "description": "Write a file into this run's output directory. "
        "The only place anything can be written.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["name", "text"],
        },
    },
    "run_verifier": {
        "description": "Run the verifier over the output directory and "
        "read what it said. Do this before finishing.",
        "parameters": {"type": "object", "properties": {}},
    },
    "finish": {
        "description": "End the run with the report. Call this once, "
        "when the work is done and the verifier agrees.",
        "parameters": {
            "type": "object",
            "properties": {"report": {"type": "object"}},
            "required": ["report"],
        },
    },
}


def describe(preset: Preset) -> list[dict[str, Any]]:
    """The allowlist as the server's tool format.

    The preset's report schema goes inside the `finish` tool rather
    than only being checked after the fact, so the shape is held by
    the decoding grammar. Measured on the scout questions: asked for
    `{answer, places}` with places as strings and told so only in
    prose, one run reported everything correctly under field names of
    its own invention and another returned places as objects. In the
    tool, the shape comes out right by construction, and the check
    after it becomes the second line of defence rather than the only
    one.
    """
    out = []
    for name in preset.tools:
        if name not in SCHEMAS:
            raise AgentError(
                f"{name!r} is not a tool; the ones there are: "
                + ", ".join(sorted(SCHEMAS))
            )
        described = dict(SCHEMAS[name])
        if name == "finish" and preset.schema is not None:
            described["parameters"] = {
                "type": "object",
                "properties": {"report": preset.schema},
                "required": ["report"],
            }
        out.append(
            {
                "type": "function",
                "function": {"name": name, **described},
            }
        )
    return out


BUILT_IN: dict[str, Preset] = {
    # Answers the scout question in M5. Everything it needs is a fact
    # the parse established, so the verifier is the grounding check:
    # a file:line it reports has to be one the index holds.
    "repo-scout": Preset(
        name="repo-scout",
        tools=(
            "find_references",
            "expand_symbol",
            "search_code",
            "read_lines",
            "finish",
        ),
        task="Answer the question about the indexed source. Use "
        "find_references for a name, expand_symbol when the answer is "
        "a step past the name, and read_lines to see the code around "
        "it. Every place you name must be a file and line the tools "
        "gave you; do not guess one.",
        schema={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "places": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "each as file:line",
                },
            },
            "required": ["answer", "places"],
        },
        ground=("places",),
        max_steps=10,
    ),
    # Answers from the documents, with the citation checked against
    # the sections it was shown.
    "doc-qa": Preset(
        name="doc-qa",
        tools=("search_docs", "finish"),
        task="Answer the question from the indexed documents. Quote "
        "the sentence you answered from, in double quotes, and name "
        "the file and page it came from.",
        schema={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "cited": {"type": "string"},
            },
            "required": ["answer", "cited"],
        },
        ground=("cited",),
        max_steps=8,
    ),
    # Generate, verify, fix, up to the step budget. The verifier is an
    # external command and is the whole point: the loop ends when
    # something other than the model says it is right.
    "verify-loop": Preset(
        name="verify-loop",
        tools=(
            "read_lines",
            "write_file",
            "run_verifier",
            "finish",
        ),
        task="Fix the file so that the verifier passes. Read it, write "
        "a corrected copy under the same name with write_file, then "
        "call run_verifier. If it still complains, read what it said "
        "and write the file again. Change only what the verifier "
        "objects to.",
        verifier=("bash", "-n"),
        image="localhost/shuttle-docs",
        max_steps=14,
        max_seconds=1200.0,
    ),
}


def settings(path: Path | None = None) -> dict[str, Preset]:
    """The built-in presets, with the file's own on top of them."""
    held = dict(BUILT_IN)
    where = path or home() / FILE
    if not where.is_file():
        return held
    try:
        written = tomllib.loads(where.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError) as error:
        raise AgentError(f"{where}: {error}") from error
    for name, values in written.items():
        if not isinstance(values, dict):
            raise AgentError(f"{where}: [{name}] is not a table")
        if not NAME.match(name):
            raise AgentError(f"{where}: {name!r} is not a preset name")
        held[name] = _build(name, values, held.get(name))
    return held


def _build(name: str, values: dict[str, Any], over: Preset | None) -> Preset:
    base = over or Preset(name=name)
    return Preset(
        name=name,
        profile=str(values.get("profile", base.profile)),
        tools=tuple(values.get("tools", base.tools)),
        task=str(values.get("task", base.task)),
        max_steps=int(values.get("max_steps", base.max_steps)),
        max_tokens=int(values.get("max_tokens", base.max_tokens)),
        max_seconds=float(values.get("max_seconds", base.max_seconds)),
        schema=values.get("schema", base.schema),
        verifier=tuple(values.get("verifier", base.verifier)),
        image=str(values.get("image", base.image)),
        ground=tuple(values.get("ground", base.ground)),
    )


# --- running one -----------------------------------------------------

Resolve = Callable[[str], Conversational]


def _bounded(run: Run, preset: Preset) -> str:
    """Which budget is spent, if any. Named so the reply can say."""
    if len(run.steps) >= preset.max_steps:
        return f"the step budget of {preset.max_steps} is spent"
    if run.tokens >= preset.max_tokens:
        return f"the token budget of {preset.max_tokens} is spent"
    if run.seconds >= preset.max_seconds:
        return f"the time budget of {preset.max_seconds:.0f}s is spent"
    return ""


def _short(text: str, room: int = STEP_CHARS) -> str:
    if len(text) <= room:
        return text
    return text[:room] + f"\n[cut: {len(text) - room} more characters]"


def _read_lines(file: str, first: int, last: int) -> str:
    """A range of an indexed file, read from the file itself.

    Through the index and then from disk, which is two different jobs.
    The index decides whether the file may be read at all — a name it
    does not hold does not resolve, so nothing reaches a path nobody
    indexed. The lines then come from the file, because they were
    asked for by number: reading them out of the units answered a
    question about line 54 with "no indexed unit covers it", the
    instantiation on that line having fallen between two units.
    """
    if first < 1 or last < first:
        return f"lines {first}-{last} is not a range"
    with code.connect() as db:
        where = code.located(db, file)
        if where is None:
            held = [
                str(one["path"])
                for one in db.execute(
                    "SELECT path FROM sources ORDER BY path LIMIT 20"
                )
            ]
            return (
                f"{Path(file).name} is not an indexed file, or has moved "
                "since. Indexed: " + ", ".join(held)
            )
    try:
        lines = where.read_text(errors="replace").splitlines()
    except OSError as error:
        return f"{where.name}: {error}"
    held = lines[first - 1 : min(last, first - 1 + MAX_LINES)]
    if not held:
        return f"{where.name} has {len(lines)} lines; {first} is past it"
    numbered = "\n".join(
        f"{number:>5}  {text}" for number, text in enumerate(held, start=first)
    )
    more = ""
    if last - first + 1 > MAX_LINES:
        more = f"\n[cut at {MAX_LINES} lines; ask for a smaller range]"
    return numbered + more


def _verify(run: Run, preset: Preset) -> tuple[bool, str]:
    """Run the preset's command over the output, in its own container.

    The command comes from a configuration file and is therefore
    arbitrary code. It gets no network, the output directory and
    nothing else, and it is never run on the host.
    """
    if not preset.verifier:
        return True, "no verifier command in this preset"
    podman = shutil.which("podman")
    if podman is None:
        return False, (
            "podman is not on the path, and the verifier runs in a container"
        )
    produced = sorted(
        one.name for one in run.directory.iterdir() if one.is_file()
    )
    if not produced:
        return False, "nothing has been written yet; use write_file first"
    # The entrypoint is overridden on purpose. An image that has one
    # takes the verifier as arguments to it: shuttle-docs answered
    # `bash -n five.sh` with its own usage text and exit 2, which the
    # loop then spent its whole budget trying to satisfy.
    argv = [
        podman,
        "run",
        "--rm",
        "--network=none",
        "--read-only",
        "-v",
        f"{run.directory}:/out:ro",
        "-w",
        "/out",
        "--entrypoint",
        preset.verifier[0],
        preset.image,
        *preset.verifier[1:],
        *produced,
    ]
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell
            argv,
            capture_output=True,
            text=True,
            timeout=min(preset.max_seconds, 300.0),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "the verifier did not finish in time"
    said = (done.stdout + done.stderr).strip()
    if done.returncode == 0:
        return True, f"the verifier passed. {said}".strip()
    return False, f"the verifier failed (exit {done.returncode}):\n{said}"


def _write_file(run: Run, name: str, text: str) -> tuple[str, bool]:
    """Into this run's directory, and nowhere else.

    A refusal comes back as a failed step and not as a message the
    loop treats as success: the repeat detector watches failures, and
    a write that was quietly refused would let a model ask for the
    same impossible name until its budget ran out.
    """
    safe = Path(name).name
    if not safe or safe.startswith("."):
        return f"{name!r} is not a file name I will write", False
    run.directory.mkdir(parents=True, exist_ok=True)
    (run.directory / safe).write_text(text, encoding="utf-8")
    return f"wrote {safe}, {len(text)} characters", True


def _perform(
    call: Call, run: Run, preset: Preset, server: Server
) -> tuple[str, bool]:
    """One tool, as the loop sees it: an answer and whether it worked.

    A tool that fails answers with why instead of raising. The loop is
    the thing being bounded, and a model that asked for a page that is
    not there should be told so and given another step, not have the
    run end underneath it.
    """
    args = call.arguments
    try:
        if call.name == "search_docs":
            found = indexing.search(
                str(args["question"]), 3, server.count_tokens
            )
            return json.dumps(found, ensure_ascii=False), True
        if call.name == "search_code":
            found = code.search(str(args["question"]), 3, server.count_tokens)
            return json.dumps(found, ensure_ascii=False), True
        if call.name == "find_references":
            with code.connect() as db:
                found = code.references(
                    db, str(args["name"]), str(args.get("relation", ""))
                )
            return json.dumps(found, ensure_ascii=False), True
        if call.name == "expand_symbol":
            with code.connect() as db:
                found = code.expand(
                    db, str(args["name"]), int(args.get("hops", 2))
                )
            return json.dumps(found, ensure_ascii=False), True
        if call.name == "read_lines":
            return (
                _read_lines(
                    str(args["file"]),
                    int(args["first"]),
                    int(args["last"]),
                ),
                True,
            )
        if call.name == "write_file":
            return _write_file(run, str(args["name"]), str(args["text"]))
        if call.name == "run_verifier":
            passed, said = _verify(run, preset)
            run.checks.append("passed" if passed else "failed")
            run.verified = passed
            return said, True
    except KeyError as error:
        return f"the call is missing the argument {error}", False
    except (indexing.IndexingError, OSError, ValueError) as error:
        return f"{type(error).__name__}: {error}", False
    return f"{call.name!r} is not a tool I have", False


def _check_report(
    run: Run, preset: Preset, report: dict[str, Any], shown: str
) -> tuple[bool, list[str]]:
    """Whether the report satisfies what the preset asked of it.

    Three kinds of verifier, each able to refuse on its own. The
    schema says the shape is right, the grounding says the quotations
    and the places were really in what the tools returned, and the
    command says something outside the model agrees.
    """
    said: list[str] = []
    ok = True
    if preset.schema is not None:
        missing = [
            key
            for key in preset.schema.get("required", [])
            if key not in report
        ]
        if missing:
            said.append("the report is missing " + ", ".join(missing))
            ok = False
        else:
            said.append("the report has the fields the preset asks for")
    if preset.ground:
        cited = {key: report[key] for key in preset.ground if key in report}
        absent = grounding.fields_in_source(cited, shown)
        # ensure_ascii=False or the check cannot pass. The default
        # turns every non-ASCII character into a \uXXXX escape, and
        # the escape is what then gets looked for in the source: a
        # correct answer holding a µ, an Ω, a °C or the curly quotes
        # gcc writes its messages with was refused for that alone.
        # Measured on a real make log — the model reported
        # `expected ‘;’ before ‘return’`, which is the line the log
        # has, and the run was refused three times and gave up.
        quoted = grounding.check(
            json.dumps(cited, ensure_ascii=False, default=str), shown
        )
        if absent or quoted.missing:
            said.append(
                "these are not in what the tools returned, so they were "
                "not read out of them: "
                + ", ".join((absent + quoted.missing)[:4])
            )
            ok = False
        else:
            said.append(
                "the "
                + " and ".join(preset.ground)
                + " came from a tool answer"
            )
    if preset.verifier:
        if not run.checks:
            said.append("the verifier was never run")
            ok = False
        elif run.checks[-1] != "passed":
            said.append("the verifier did not pass")
            ok = False
        else:
            said.append("the verifier passed")
    return ok, said


def _in_content(text: str, allowed: tuple[str, ...]) -> Call | None:
    """A tool call the model wrote as text instead of calling.

    Qwen3 does this: it decides on the tool, gets the arguments right,
    and emits `{"name": "finish", "arguments": {...}}` as its message.
    Measured on the scout questions, a run that had already found the
    right two lines then spent six of its ten steps writing that same
    object into the content and being told it had not called anything.

    Refusing it is not integrity, it is a formatting argument the loop
    loses. It is accepted and recorded as having come from the
    content, so the transcript says which channel was used.
    """
    guess = text.strip()
    if not guess.startswith("{"):
        return None
    try:
        parsed = json.loads(guess)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    name = parsed.get("name") or parsed.get("tool")
    args = parsed.get("arguments")
    if not isinstance(name, str) or name not in allowed:
        return None
    return Call(
        id="from-content",
        name=name,
        arguments=args if isinstance(args, dict) else {},
    )


SYSTEM = (
    "You work by calling tools. Call one tool at a time, read what it "
    "answers, and call the next. When the work is done call finish "
    "with the report. Do not describe what you would do; do it. Never "
    "state a file, a page or a line that a tool did not give you.\n\n"
    "{task}"
)


def start(
    resolve: Resolve,
    preset: Preset,
    task: str,
    transcript: bool = True,
) -> Run:
    """Run the loop until something verifies or a budget is spent."""
    preset.verified()
    if not task.strip():
        raise AgentError("an agent run needs a task")
    tools = describe(preset)
    server = resolve(preset.profile)
    run = Run(id=new_id(f"agent-{preset.name}"), preset=preset.name, task=task)
    run.directory.mkdir(parents=True, exist_ok=True)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM.format(task=preset.task)},
        {"role": "user", "content": task},
    ]
    # What the tools have said, for the grounding check: a report may
    # only name what a tool returned, and this is that record.
    shown: list[str] = []
    # The last failing call, to notice a loop that cannot move.
    stuck: tuple[str, str] = ("", "")
    repeats = 0
    began = time.monotonic()
    n_predict = max(512, server.context_size() // 4)
    while True:
        run.seconds = time.monotonic() - began
        spent = _bounded(run, preset)
        if spent:
            run.stopped = spent
            break
        answer = server.converse(messages, n_predict, schema=None, tools=tools)
        run.tokens += answer.tokens
        calls = answer.calls
        if not calls:
            written = _in_content(answer.content, preset.tools)
            if written is not None:
                calls = (written,)
        if not calls:
            # A model that answered in prose has not finished: the
            # report is the only thing that counts as finishing, and
            # saying so is cheaper than another whole attempt.
            messages.append({"role": "assistant", "content": answer.content})
            messages.append(
                {
                    "role": "user",
                    "content": "Call a tool. When the work is done, call "
                    "finish with the report.",
                }
            )
            prose = ("(prose)", answer.content[:200])
            repeats = repeats + 1 if prose == stuck else 1
            stuck = prose
            _record(
                run,
                Step(
                    len(run.steps) + 1,
                    "(prose)",
                    {},
                    _short(answer.content, 400),
                    round(time.monotonic() - began, 1),
                    False,
                ),
                transcript,
            )
            if repeats > MAX_REPEATS:
                run.stopped = (
                    f"the same prose came back {repeats} times instead "
                    "of a tool call"
                )
                break
            continue
        messages.append(
            {
                "role": "assistant",
                "content": answer.content,
                "tool_calls": [
                    {
                        "id": one.id,
                        "type": "function",
                        "function": {
                            "name": one.name,
                            "arguments": json.dumps(
                                one.arguments, ensure_ascii=False
                            ),
                        },
                    }
                    for one in calls
                ],
            }
        )
        for call in calls:
            at = time.monotonic()
            same = (
                call.name,
                json.dumps(call.arguments, sort_keys=True, default=str),
            )
            if call.broken:
                said, ok = call.broken, False
            elif call.name == "finish":
                report = call.arguments.get("report") or {}
                if not isinstance(report, dict):
                    said, ok = "the report must be an object", False
                else:
                    passed, why = _check_report(
                        run, preset, report, "\n".join(shown)
                    )
                    run.report = report
                    run.checks.extend(why)
                    if passed:
                        run.verified = True
                        run.stopped = "finished and verified"
                        _record(
                            run,
                            Step(
                                len(run.steps) + 1,
                                "finish",
                                {},
                                "; ".join(why),
                                round(time.monotonic() - at, 1),
                            ),
                            transcript,
                        )
                        return _close(run, transcript)
                    said, ok = (
                        "not finished: "
                        + "; ".join(why)
                        + ". Fix it and call finish again.",
                        False,
                    )
            else:
                said, ok = _perform(call, run, preset, server)
                if ok:
                    shown.append(said)
            if ok:
                stuck, repeats = ("", ""), 0
            elif same == stuck:
                repeats += 1
            else:
                stuck, repeats = same, 1
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": _short(said),
                }
            )
            _record(
                run,
                Step(
                    len(run.steps) + 1,
                    call.name,
                    call.arguments,
                    _short(said, 600),
                    round(time.monotonic() - at, 1),
                    ok,
                    call.id == "from-content",
                ),
                transcript,
            )
            if repeats > MAX_REPEATS:
                run.stopped = (
                    f"{call.name} failed {repeats} times with the same "
                    "arguments; the loop is not moving"
                )
                return _close(run, transcript)
    return _close(run, transcript)


def _record(run: Run, step: Step, transcript: bool) -> None:
    """Keep the step, and write it down as it happens.

    As it happens rather than at the end: a run killed by its time
    budget, or by the operator, is still a record of how far it got.
    """
    run.steps.append(step)
    if not transcript:
        return
    line = json.dumps(
        {
            "at": step.at,
            "tool": step.tool,
            "arguments": step.arguments,
            "answer": step.answer,
            "seconds": step.seconds,
            "ok": step.ok,
            "written_not_called": step.written,
        },
        ensure_ascii=False,
        default=str,
    )
    try:
        run.directory.mkdir(parents=True, exist_ok=True)
        with (run.directory / "steps.jsonl").open(
            "a", encoding="utf-8"
        ) as sink:
            sink.write(line + "\n")
    except OSError:
        pass


def _close(run: Run, transcript: bool) -> Run:
    if not run.stopped:
        run.stopped = "the loop ended without finishing"
    if transcript:
        with contextlib.suppress(OSError):
            (run.directory / "report.json").write_text(
                json.dumps(run.summary(), indent=1, ensure_ascii=False),
                encoding="utf-8",
            )
    return run
