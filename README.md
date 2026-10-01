<!-- SPDX-License-Identifier: GPL-3.0-only -->

# SHUTTLE

[![ci](https://github.com/FPGArtktic/shuttle-stack/actions/workflows/ci.yml/badge.svg)](https://github.com/FPGArtktic/shuttle-stack/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](delegate/pyproject.toml)
[![latest tag](https://img.shields.io/github/v/tag/FPGArtktic/shuttle-stack)](https://github.com/FPGArtktic/shuttle-stack/tags)
[![licence: GPL-3.0-only](https://img.shields.io/badge/licence-GPL--3.0--only-blue.svg)](LICENSE)

SHUTTLE runs two `llama-server` instances under rootless Podman on a network
without egress, so that a planning agent can hand bulk text work to a model on
your own machine. The name expands to *SHUTTLE Hauls Unwieldy Text To Local
Engines*.

## Status

M0 is the inference layer, M1 the delegate that puts it in front of an agent,
and M2 the evaluation that says how far either can be trusted.

| Component | State |
|---|---|
| `install.sh` phases | working — `detect plan host gpu quadlets models verify bench docs delegate sweep audit status`, each runnable on its own |
| `shuttle-long` | working — Qwen3-8B Q4_K_M, partial GPU offload through CDI, one slot |
| `shuttle-fast` | working — Qwen3-1.7B Q8_0, CPU only, four parallel slots |
| Speculative decoding | working — Qwen3-0.6B draft, 2.15x on generation where it fits in VRAM |
| Internal network | working — `10.89.7.0/24`, `Internal=true`, no route out |
| Direct host ports | working — `127.0.0.1:8081` and `:8082`, measured to reach an internal network |
| Model download | working — size and sha256 from the Hugging Face tree API, resumable |
| Benchmarks | working — `bench` writes JSON with the configuration it measured |
| KV slot dumps | working — a document's cache survives a restart; 1.7x on the next question, 305 MiB per document |
| `status` | working — which servers answer, which model each holds, how large its context is |
| `summarize_file` | working — folds a file larger than the context into one summary |
| `ask_file` | working — answers from a file, or says the file does not answer; narrows by regexp |
| `classify_file` | working — labels enforced by a schema, votes across parts, reports agreement |
| `extract` | working — fields enforced by a schema, narrows by regexp; refuses a file that needs more than one part, and an answer cut off by the output cap |
| `brainstorm` | working — several options, shape fixed by a schema, content explicitly unverified |
| N-gram lookup | working, off by default — measured no gain, and slightly slower with no draft model |
| Cascade | working, and not worth switching on here — the ladder verified on the first profile every time, and cost 3.2x the seconds |
| Best-of-n | working — a sample failing its verifier is drawn again, warmer; 1.00 attempts per case over the evaluation set |
| Document index | working — one SQLite file, BM25 and vectors fused, sections cited by file, page and clause, answers under 300 tokens |
| Source index | working — Tree-sitter units for fourteen languages, cited by file and line range; falls back to blank-line blocks and says why |
| Bounded agents | working — six presets, `agent_start` runs one until a verifier agrees or a budget is spent; a preset with no verifier is refused; `doc-extract` scores 78% against M5's reference set |
| Code graph | working — includes, calls, instantiation, inherit, DEPENDS and every variable assignment, read off the parse; `find_references` answers with file:line and no model, `expand_symbol` one hop further |
| Network audit | working — `install.sh audit` proves no egress with two probes, and checks what each server publishes and where |
| Audit log rotation | working — eight megabytes a generation, four kept, rotated before the write |
| Backup | working — `shuttle-backup` writes one tar holding the index, the transcripts and the log; the index copied through SQLite, not byte for byte |
| Distillation set | working, and empty until used — `grade_run` records a verdict and a correction, `shuttle-dataset` exports JSONL |
| Search cache | working — a repeated search served from the file, 15x faster, invalidated by the index changing rather than by the clock |
| Overnight pass | working — a timer indexes and digests watched directories and leaves a thousand-token report; the KV warming is in and off, worth a second against 305 MiB |
| Answer cache | working, deliberately narrow — a reworded repeat of a question answered without the model, 46x faster; a paraphrase in different words is not, and the measurement below says why |
| PDF documents | working — text layer first, OCR only for the pages without one, in a container with no network |
| Sessions | working — a document read once and asked repeatedly; the transcript is the record, the KV dump only a cache |
| `start_job` / `get_status` / `get_result` | working — a file tool run in the background, polled and collected |
| Evaluation set | working — twenty cases with known answers, two of them refusals; 20/20 on the long server, 18/20 on the fast one |
| Profiles | working — `long`, `fast`, `extract`; endpoint, sampling and report limit, overridable in `profiles.toml` |
| Report limit and run files | working — a thousand tokens back, the whole output to `.shuttle/runs/<id>.md` |
| Audit log | working — one JSONL line per call in `.shuttle/audit.jsonl`, success or failure |
| Citation grounding | working — a quoted span absent from the source is listed rather than trusted; compared without markup, wrapping or spacing |
| Ubuntu 24.04 | not yet on hardware — the dry runs pass in CI, the system-changing phases have only been run on Arch |

Neither milestone includes any of the following, and no placeholders are left
for them:

- the DeepSeek Harness,
- BitNet,
- vision-language models,
- integration with [WEFT](https://github.com/FPGArtktic/weft-mcp).

## Why

Work that is bulky but not hard — summarising, classifying, extracting,
rewriting — consumes a large share of a planning agent's token budget while
asking little of the model that does it. SHUTTLE draws the boundary there: the
agent decides what needs doing and a local model does it, so the tokens spent
are the ones spent on judgement. The server is `llama-server` from llama.cpp
rather than Ollama because delegation needs three things Ollama does not
expose. Structured output is reliable only with a schema the server itself
enforces, so that a label can only be one of the labels offered and a field
can only have the shape asked for; a caller can size its own requests only if
`n_ctx` per slot is stated rather than inferred; and a document read once can
be asked about after a restart only if its KV cache can be written to disk
and read back. Those three are the whole argument; everything else Ollama
does, it does well.

## Requirements

- Arch Linux, or Ubuntu 24.04 or newer.
- Podman 4.9 or newer, rootless, on cgroup v2. The installer adds the package,
  the subuid and subgid ranges and `loginctl enable-linger` if they are absent.
- `bash` 5, `curl` and `jq`. No Python on the host.
- A regular user account. The installer refuses to run as root.
- An NVIDIA GPU is optional. With one, `shuttle-long` offloads as many layers
  as the VRAM budget allows through CDI, which needs
  `nvidia-container-toolkit`; without one, both servers run on the CPU.
- `uv`, for the delegate only. Nothing the stack runs needs it, and the
  installer does not fetch it: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- Disk: **7.0 GiB** for the three GGUF files and **5.2 GB** for the two
  container images, so about **12.5 GB** in total. The `models` phase refuses
  to start a download that would leave under 1 GiB free.

## Quick start

```
git clone https://github.com/FPGArtktic/shuttle-stack && cd shuttle-stack
./install.sh detect plan --dry-run
./install.sh
```

The first command brings the repository. The second reports what the installer
found and what it intends to do, changing nothing. The third runs every phase
except `bench`, asking before each command that needs root.

The last phase installs the delegate; telling an MCP client about it is a
separate step, described under *Delegating* below.

## How it works

```
            MCP client (Claude Code, Claude Desktop, ...)
                               |
                         stdio: 16 tools
                               |
                   shuttle-delegate, on the host
                               |
                               +--> .shuttle/documents.sqlite
                               |      FTS5 for BM25, sqlite-vec
                               |      for the vectors, one file
                               |
                               +--> shuttle-docs container
                               |      Poppler, Tesseract,
                               |      Tree-sitter; --network=none
                               |      and one directory read-only
                               |
              127.0.0.1:8081   |   127.0.0.1:8082
            +------------------+------------------+
            |                                     |
            v                                     v
  +---------------------+             +---------------------+
  | shuttle-long        |             | shuttle-fast        |
  | .service            |             | .service            |
  |                     |             |                     |
  | llama.cpp:server-   |             | llama.cpp:server    |
  |   cuda              |             |                     |
  | Qwen3-8B Q4_K_M     |             | Qwen3-1.7B Q8_0     |
  | + Qwen3-0.6B draft  |             | 4 slots, CPU only   |
  | GPU via CDI         |             |                     |
  |                     |             |                     |
  | 10.89.7.10:8080     |             | 10.89.7.11:8080     |
  +----------+----------+             +----------+----------+
             |                                   |
             +-----------------+-----------------+
                               |
                   shuttle.network  10.89.7.0/24
                   Internal=true, gateway 10.89.7.1
                   no route out of the network
```

Both servers are systemd user units generated by Quadlet. They reach each other
by container name over the internal network, and the host reaches them through
the two published ports. Nothing inside the network can reach the internet,
which is why the models are downloaded during installation rather than at run
time.

The installer writes these files, and nothing else:

| Path | Contents |
|---|---|
| `~/.config/shuttle/long.env` | `LLAMA_ARG_*` for `shuttle-long` |
| `~/.config/shuttle/fast.env` | `LLAMA_ARG_*` for `shuttle-fast` |
| `~/.config/shuttle/stack.env` | URLs, addresses and ports, for callers |
| `.shuttle/runs/<id>.md` | the whole output of each call, beside the working directory |
| `.shuttle/audit.jsonl` | one line per call |
| `.shuttle/documents.sqlite` | the index: sections, source units, vectors, and the kept searches |
| `~/.config/containers/systemd/shuttle.network` | the internal network unit |
| `~/.config/containers/systemd/shuttle-long.container` | the long server unit |
| `~/.config/containers/systemd/shuttle-fast.container` | the fast server unit |
| `~/.local/share/shuttle/models/*.gguf` | the three model files |
| `~/.local/share/shuttle/cache/{long,fast}/` | KV slot dumps |
| `~/.local/state/shuttle/install.log` | every command that changed the system |
| `~/.local/state/shuttle/bench-*.json` | benchmark results with their configuration |
| `/etc/cdi/nvidia.yaml` | the CDI specification, written as root by the `gpu` phase |
| `~/.local/bin/shuttle-delegate` | the delegate launcher, written by the `delegate` phase |

A generated file is rewritten only when its content changes, and the previous
version is kept as `<name>.bak.<timestamp>`.

## Delegating

The delegate is an MCP server. It reads `stack.env`, talks to the two servers
over their published ports, and exposes four tools.

### What it saves, on one real task

The task: one sentence on what each of the installer's ten phases does.
Reading `install.sh` to answer it costs 12844 tokens of context, counted with
the server's own tokeniser. Ten narrow questions through `ask_file` come back
as 468.

```
reading the file               12844 tokens of context
ten answers instead              468 tokens of context
saved                          12376 tokens (97%)
spent on this machine           2557 tokens
quotations not in the file         3
```

`python -m evals.reduction` repeats it. The last line is there on purpose: a
saving bought with a wrong answer is not a saving, so the quotations that are
not in the file are counted beside it.

The first attempt at this measurement gave the same 97% and four of the ten
answers were about the wrong phase, because a pattern landing on a six-line
function pulled thirty lines after it and the model read the next function
too. That is what `until` is for.

### Registering it with a client

The `delegate` phase puts the launcher on PATH. It stops there, because an
installer should not edit configuration files belonging to a program it did
not install.

Claude Code:

```
claude mcp add --scope user shuttle -- ~/.local/bin/shuttle-delegate
```

Claude Desktop, in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "shuttle": {
      "command": "/home/you/.local/bin/shuttle-delegate",
      "args": ["--transport", "stdio"]
    }
  }
}
```

Any other MCP client is told the same two things: run that binary, speak stdio.

`stdio` is the default and is what a client on this machine wants. The
launcher also accepts `sse` and `streamable-http` for a client elsewhere on
the network, but neither carries authentication of its own, so anything
exposed that way needs something in front of it. M0 publishes the servers on
`127.0.0.1` for the same reason.

### Teaching a client when to reach for it

Registering the server makes the tools available; it does not make a client
prefer them. An agent with its own file reading will keep using it, because
that is the habit, and a sentence in one message wears off after a few turns.

The delegate ships instructions of its own, which a client reads along with
the tool list, and those set the defaults. A standing rule goes in
`CLAUDE.md`: `~/.claude/CLAUDE.md` applies to every project, a `CLAUDE.md` in
a project root applies to that one. This repository deliberately ships
neither, because the rule belongs to whoever runs the stack.

```markdown
## Delegating to SHUTTLE

Two local llama-servers are reachable through the `shuttle` MCP server.
Use them to keep bulk text out of context, not to avoid deciding things.

- Before reading a long file only to establish a fact from it, call
  `ask_file` with a `pattern` that lands on the passage and an `until`
  that says where the passage ends: `^[a-z_]+\(\)` for the next shell
  function, `^## ` for the next section. Without `until` the region
  runs on and the answer comes back about the neighbour.
- To pull known fields out of a file, or the same fields out of many,
  call `extract` with a JSON schema.
- Find things yourself with grep; hand the local model a region, never
  a search. It is worse at searching and the search is free.
- Pass a path. Never paste file contents into a tool call.
- Do not delegate a file you could read in a few hundred tokens: the
  call costs seconds and the read costs almost nothing.
- Do not ask about anything the file does not contain. Asked without a
  text to read, these models invent an answer and sound sure of it.
- Prefer `profile: "long"`. It is both more accurate and faster here;
  `fast` is for answering several requests at once.
- Use `start_job` when the file is large: a call can take minutes, and
  `get_status` and `get_result` let you do something else meanwhile.
- `brainstorm` returns suggestions, never findings. Nothing in them was
  verified against anything; judge them yourself.
- `local_tokens` in each reply is the context you saved. If it is
  small, the call was not worth making. `quotes_not_in_source` means
  the answer was reconstructed rather than read.
```

Keep it short. A rule competing with twenty other rules is a suggestion.

### What the tools do

Every tool takes a **path**, not text. The file is read on this machine, split
if it does not fit the server's context, and only the result crosses back. An
agent that reads a file itself and pastes the contents into a tool call spends
exactly the tokens the project exists to save, so the tool descriptions say so
and the instructions repeat it.

| Tool | Takes | Returns |
|---|---|---|
| `status` | nothing | which servers answer, their models and contexts |
| `summarize_file` | path, words, focus, profile | one summary, however many parts the file needed |
| `ask_file` | path, question, pattern, until, context, words, attempts, cascade_profiles, profile | the answer, or that the file does not answer |
| `classify_file` | path, labels, question, profile | one of the labels, the vote and the agreement |
| `extract` | path, JSON schema, instructions, pattern, until, context, attempts, cascade_profiles, profile | the fields, in the shape asked for |
| `brainstorm` | a request, optionally a path | several options, marked unverified |
| `index_path` | path, heading | for a document, how many sections and whether by heading or page; for source, the language, the units and how they were cut |
| `search_docs` | a question, limit, file | a few sections with file, page and heading, under 300 tokens; a file keeps the search inside that document |
| `search_code` | a question, limit | a few units with file and line range, under 300 tokens |
| `find_references` | a name, a relation, limit | every file:line that names it, with no model asked |
| `expand_symbol` | a name, hops, limit | where it is defined with the head of the definition, who names it, and what those places reach, each as file:line |
| `agent_start` | a task, a preset | the report, which budget stopped it, and the directory holding every step |
| `agent_presets` | nothing | the presets, with each one's budget and what verifies it |
| `grade_run` | a run, a verdict, a correction | the verdict recorded, and how many cases the set now holds |
| `list_indexed` | nothing | the documents with their sections and pages, the source files with their language, the files with kept answers |
| `session_open` | path, words, pattern, until, context, profile | a session id, the document read and cached |
| `session_ask` | a session id and a question | the answer, with the cache restored rather than the document resent |
| `session_close` | a session id | the cache dropped, the transcript kept |
| `start_job` | a tool name and its arguments | a job id, at once |
| `get_status` | a job id | queued, running, done or failed, with the time |
| `get_result` | a job id | the finished answer, as the tool would have returned it |

Labels and field shapes are enforced by llama-server through a response
format, so an answer is valid by construction rather than by parsing hope.
Every answer reports `local_calls` and `local_tokens`: the work the machine
did, which is the work the caller's context did not have to hold.

A file longer than the context is summarised in parts and the parts folded
together, repeatedly, until one remains. Classification votes across the parts
and reports how far they agreed. Extraction refuses such a file outright,
because merging fields across parts means inventing a precedence the caller
never gave.

Measured on the reference machine: `install.sh`, 35 KB, summarised in two
parts and one fold for 13260 local tokens; `CONTRIBUTING.md` yielded its
subject-length limit, linter and indent style in a single call of 1410.

Qwen3 reasons before answering by default, and that reasoning is spent from
the same budget as the answer. The delegate turns it off. Left on, a request
with 200 tokens to spend returns an empty string and a finish reason of
`length`.

### The index, for a question you cannot turn into a pattern

`pattern` needs you to know the wording. `index_path` reads a document into
the index so `search_docs` can find the right part of it without one, and
every hit says which file, page and clause it came from. The citation is the
point: an answer that names its page can be checked, and one that does not
cannot.

```
                                    tokens  cited
absolute maximum ratings               296  07447_datasheet.pdf page 3
what limits the output current         284  07447_datasheet.pdf page 3
how long may a commit subject be       262  CONTRIBUTING.md "## Commits"
what package types are available       279  07447_datasheet.pdf page 11
```

**The index is one file.** `~/.shuttle/documents.sqlite`, or wherever
`SHUTTLE_INDEX` points. FTS5 holds the lexical half and sqlite-vec the
vectors, in the tables [WEFT](https://github.com/FPGArtktic/weft-mcp)'s OCR
module already writes, so a document indexed there is searchable here and a
copy for a machine with no network is one `cp`. Nothing else has to be
installed or kept running.

**Both halves are asked, because they fail differently.** A register name or
a clause number is a lexical question: FTS5 finds `TIMER0` or `4.2.1`, and a
vector search is as likely to return the paragraph beside it. "what resets
the peripheral" is the opposite, and BM25 cannot see the word the document
used instead. Measured on the 37-page datasheet cut into 44 sections, over
40 questions that are a term appearing in exactly one section and 22 written
by `shuttle-long` from the section each answers:

| | exact term | phrased question |
|---|---|---|
| BM25 alone | 39/40 | 14/22 |
| vectors alone | 13/40 | 17/22 |
| the two fused | 39/40 | 16/22 |

Neither half is usable alone. The rankings are fused by reciprocal rank
rather than by score, because a BM25 score and a cosine distance share no
scale and normalising them would mean inventing one; the reply says which
half found each hit. MMR then drops the second copy of a section the
document repeats, which a datasheet does on every page.

Dropping the common terms from the lexical half looked obviously right and
earned nothing: across those 62 questions, thresholds from a quarter of the
index to all of it gave identical recall, and a tenth lost one. A
44-section index is too small for document frequency to tell "before" from
"absolute"; both appear once.

**The budget is in tokens, not characters.** 1066 characters of a datasheet's
numbers come to 549 tokens — half a token a character against a third for
prose — so the excerpts are trimmed by measurement until the whole answer
fits 300. When the citations alone fill the budget, a hit is dropped and the
reply says so: three source units cost 277 tokens before a character of
their text, and two units somebody can read beat three nobody can.

**Sections come from headings, or from pages, and the reply says which.** A
heading is a numbered clause or a Markdown heading whose number is followed
by something that looks like a word, and three of them are wanted before the
document is believed to have a structure. Both halves of that were learned
the hard way: the first pattern turned every row of a timing table into a
heading, so citations read "8" and "15 5"; and then a single numbered
footnote put a whole datasheet into heading mode, with every page of it cited
under that footnote. Pass your own `heading` pattern when you know the
document's shape.

**A section never spans pages.** One running from page 3 to page 37 was
stored as page 3, which sends a reader to the wrong page and is worse than no
citation at all.

The embeddings are bge-m3 through Ollama, on the CPU. On the GPU it is 43 ms
a section against 62, and it takes 289 MiB of the card `shuttle-long` is
using: nineteen milliseconds is not worth taking memory from the server that
answers the questions. If Ollama cannot be reached the lexical half still
answers and the reply carries a `degraded` note, because a search that had
quietly become a grep is worse than one that failed.

Indexing is not quick: the 37-page datasheet takes 96 seconds, of which 25 is
reading it and 71 is embedding 44 sections. Do it once per document, with
`start_job` if you would rather not wait. Asking is quick — 0.13 s, and 8.9 ms
for a question already asked, which is 15 times faster and the reason the
cache exists. A cached answer says how old it is, and is thrown away the
moment anything is indexed rather than when an hour is up.

### Source, cut into the units its language defines

`index_path` on a `.sv`, `.vhd`, `.bb`, `.dts`, `.c`, `Kconfig`, `Makefile`,
`.tcl` or `.sh` file sends it to `shuttle-docs` to be parsed instead of read,
and `search_code` then answers with a place to open rather than a file to
read:

```
do_install                        passt_git.bb       lines 29-31  function_definition do_install
debouncer                         debouncer.sv       lines 19-35  module_declaration header debouncer
which module divides the clock    clk_tick.v         lines 20-29  module_declaration header clk_tick
the clock period constraint       counter.sdc        line 5
```

The units are the ones an engineer would name a file by: C functions,
structs, enums and typedefs; Bash and Python functions; Verilog and
SystemVerilog modules, packages, interfaces, always and initial blocks,
functions and tasks; VHDL entities, architectures, packages and processes;
device tree nodes; Kconfig entries; Make rules; Tcl procs; BitBake tasks and
python functions; Rust items and impl blocks; Go functions, methods and
types. A run of nodes that is no unit of its own becomes one,
which is what makes a recipe's SUMMARY, LICENSE, SRC_URI and DEPENDS a unit
without a rule for it. A module yields what comes before its first inner
unit rather than itself, so its ports are findable without indexing the whole
module twice over.

**The grammars are checked, not trusted.** They are community-maintained and
the design says so. The test is what came out, not how many error nodes the
parse holds: `tree-sitter-c` leaves 84 % of a preprocessor-heavy
`tclAppInit.c` inside error nodes and still finds `main` and `Tcl_AppInit` at
the right lines, while `tree-sitter-tcl` fails on git-gui's 1370-line
`blame.tcl` so completely that the tree holds one unnamed blob of the whole
file. The first is kept; the second falls back to blank-line blocks and the
reply says why. Over 20 files of VHDL, SystemVerilog, Verilog, BitBake, Make
and Tcl, 15 parsed into units and 5 fell back — four recipes whose shell
bodies the BitBake grammar cannot follow, and one constraint file.

The grammars are fetched when the image is built, never at runtime: the
container runs with `--network=none` and could not fetch one if it wanted to.

Indexing source costs what indexing prose costs, and for the same reason: 20
files and 113 units took 190 seconds, nearly all of it bge-m3 on the CPU.

### Asking the same thing twice

`ask_file` is the expensive tool. It reads the file, answers, and is drawn
again when the answer does not check out: tens of seconds, sometimes minutes.
Asking it the same question twice in one conversation is the ordinary case,
so the answer is kept in the index and served from there.

```
                                       time   local tokens
how long may a commit subject be      5.95s           1457
how long may a commit subject be      0.13s              0   from cache
How long may a commit subject be?     0.14s              0   from cache, 0.023 away
which linter does the project use     6.97s           1485
which linter does the project use     0.12s              0   from cache
```

**It is narrower than it looks, on purpose.** The plan asks for a paraphrase
to be served too, and that cannot be done safely with a distance. Over 22
questions written from a datasheet and 22 rewrites of them by `shuttle-long`,
the rewrites sat between 0.018 and 0.173 apart and every other pairing of the
462 began at 0.087 — which reads like a clean threshold near 0.2 until you
notice that a model asked to reword a question keeps most of its vocabulary.
Six pairs written by hand to mean the same thing in different words:

```
0.705  which leg is ground               / what pin is VSS on
0.623  how do I keep the chip cool       / thermal impedance of the PW package
0.542  how long does a signal take       / what is the propagation delay
0.402  how hot can it get before it dies / maximum junction temperature
0.265  what voltage on the supply pin    / absolute maximum supply voltage
0.177  how much current can one sink     / output low sink current limit
```

and against them, two questions whose answers differ by a sign:

```
0.213  how much current can one output sink / ... can one output source
```

The same question reaches 0.705 and an opposite one 0.213. No threshold lies
between them, so the cut is below the dangerous pair rather than above the
useful ones: a reworded repeat is served, a paraphrase in a different
vocabulary goes to the model, and the datasheet is never answered about
source when it was asked about sink. Widening this needs something other
than a distance.

**What it refuses matters more than what it serves.** An answer that failed
its own grounding check is never kept — the argument for serving a stored
answer is that it was checked once. An entry belongs to one file at one size
and modification time, so editing the file throws its answers away. An
answer narrowed by a `pattern` is not kept at all, having been read from
whatever region that pattern matched. A cache that cannot be reached is a
miss and never a failed call. Every hit says what it was originally asked,
how far that was, and that it cost nothing.

### The night, and what the morning starts with

`shuttle-sweep` runs on a systemd timer over the directories named in
`sweep.toml`, and the pass is idempotent: what was indexed is remembered by
path, size and modification time, so a directory of four hundred datasheets
costs nothing on the second night and an edited file is read again. A
document dropped in during the evening is searchable with citations in the
morning, which is what M6 asks for.

```
roots  = ["~/datasheets", "~/work/bsp"]
digest = true        # section summaries, about as long again as indexing
warm   = 0           # documents to leave in a KV cache; see below
```

The night leaves `.shuttle/report.md`, held to the same thousand tokens
every other answer here is. Failures are listed before successes and the
list is cut before the report is, so what went wrong survives the trimming.
One unreadable file does not end the pass, and a document that was indexed
but not digested is reported as indexed with a note, not as failed — the
alternative has the next pass read it all over again.

**The KV warming is implemented and off by default, because it was measured
to be worth about a second.** The pass can leave the most-asked documents in
a server's KV cache, chosen from the audit log rather than guessed, so a
morning question restores a dump instead of reprocessing the document.
Measured on a 5709-token document, alternating with another document served
between every pair so neither run reads a warm slot:

```
answer length    session_ask on a dump    ask_file read fresh
20 words          4.7  5.3  5.3 s          6.3  6.1  5.3 s
200 words        18.6 19.8 20.6 s         18.2 18.4 19.3 s
```

Restoring is worth under a second on a short answer and costs about a second
on a long one, where the restore is more work than the prompt processing it
avoids. Generation dominates either way. A dump is 305 MiB whatever the
document — it is sized by the context, not by the text — so `warm = 2` buys
610 MiB of disk for a second that the answer's own length can hide. It is
left in because the mechanism is sound and the arithmetic is a property of
this machine: a host whose disk is much faster than its prompt processing,
or one with a context large enough for documents several times this size,
would see the other result. On this one, set it only if you have measured
your own.

What it chooses is not a guess: the audit log records every call with the
path it was about, so the count of successful `ask_file`, `extract`,
`classify_file`, `summarize_file` and `session_open` calls per document is
the ranking. A search does not count — it reads the index rather than the
file. A document re-indexed that same night is read again rather than kept,
because a cache about the text as it was is worse than no cache. A document
still wanted keeps the session it had, so the id the report names survives
from one night to the next.

### PDFs

A path may be a PDF, anywhere a path is taken. It is extracted in the
`shuttle-docs` image — Poppler, Tesseract and Tree-sitter and nothing else —
run with
`--network=none`, `--read-only`, and only that document's own directory
mounted read only. A document is data and never a program: one that tries
something cannot reach the network, cannot write, and cannot see another
directory.

The text layer is used wherever there is one and OCR only for the pages
without. On a 37-page datasheet that is the difference between a third of a
second and half a minute:

```
4 pages, all with a text layer     0.3 s
37 pages, 16 of them needing OCR  25.1 s
```

The text arrives marked with page numbers, so an answer can say which page it
came from. Asked about the absolute maximum supply voltage with a `pattern`
landing on the rating table, the long server answered `20V` from page 3 in ten
seconds, and `extract` pulled `{"supply_voltage_absolute_maximum": "20V",
"part_numbers": "CD4049UB, CD4050B"}` in five and a half.

A whole datasheet is 24000 tokens against a context of 8192, so `pattern` is
not optional here, it is the only way in. A pattern that matches nothing says
so and names the pattern, which is how both of those examples were found: the
first guess at the wording matched nothing at all.

`./install.sh docs` builds the image. It is built rather than pulled because
it is nine lines of apt-get and a shell script, and a reader can see all of
it.

### The code graph, for a question with one right answer

Some questions are not searches. "Who instantiates `debouncer`" has an answer
the parse already established, and ranking guesses at it would be worse than
looking it up. `find_references` is that lookup: no model, no embedding, no
score.

```
debouncer / instantiates      counter_top.sv:54   counter_top.sv:65
clk_tick / instantiates       counter_top.sv:97   counter_top.sv:122
RDEPENDS                      openvswitch.inc:16  RDEPENDS:${PN}
                              openvswitch.inc:20  RDEPENDS:${PN}-switch
autotools / uses              openvswitch.inc:75  slirp4netns_git.bb:21
```

The relations are read off the same tree as the units: C and C++ includes and
calls, `include and package imports and module instantiation in Verilog and
SystemVerilog, use clauses and component and entity instantiation in VHDL,
`inherit`, `require`, `DEPENDS`, `RDEPENDS` and `PROVIDES` in BitBake, `use`
and `mod` and calls in Rust, `import` and calls in Go, `source` in shell and
Tcl. Every variable assignment is an edge too, not only the dependency
lists, because "which layer overrides this" needs the variables nobody
thought to enumerate.

A name matches whatever follows it after a separator, which is one rule for
four languages: `RDEPENDS` finds `RDEPENDS:${PN}`, `std` finds
`use std::io::Write`, `fmt` finds the call to `fmt.Println`. The separator
has to be there, so `counter` is not answered about `counterweight`.

Two things it deliberately does not do. A shell or Tcl call is recorded only
for a name the file defines itself — everything in a shell script is a
command, and `ls` is not an edge in anybody's graph. And C calls are all
recorded, including library ones, which makes a large C file produce a lot of
edges; that is what "includes and calls" asks for.

A `${...}` is cut out of a value before it is read as a list, nesting and
all. Skipping a word holding a `${` is not enough, because an expansion can
hold whitespace and only its first word holds the brace:
`${@bb.utils.contains('XEN_TARGET_ARCH', 'x86_64', 'dev86-native', '', d)}`
put `'x86_64',` and `'dev86-native',` into the graph as recipes. Cutting the
expansion removes 16 of `xen-tools.inc`'s 183 edges and 5 of `xen.inc`'s 75,
all of them that or a trailing line continuation, and loses nothing else.
Resolved values are the separate question the design notes answer with
`bitbake -e` as an indexed document; this graph is the static
approximation.

### Two hops, for a question whose subject is not the name

`find_references` answers "who instantiates `counter_top`". It does not
answer "which layer overrides `EXTRA_OECONF`", although it looks as though
it should: the answer there is the recipe that assigns it and what that
recipe inherits, and the inherit is a second hop away. `expand_symbol` is
that hop. Still a join and not a search — every edge carries its line and
every unit its span, so the unit enclosing an edge is a comparison.

```
counter_top    defined   counter_top.sv:24   module_declaration header (24-41)
                         | module counter_top #(
                         |     parameter int CLK_HZ    = 50_000_000,
                         |     parameter int TICK_HZ   = 4,
                         | ) (
                         |     input  logic       clk,
               named     counter_top_tb.sv:35   instantiate
               reaches   debouncer       counter_top.sv:54
                                           -> debouncer.sv:19
                         clk_tick        counter_top.sv:97
                                           -> clk_tick.v:20
                         updown_counter  counter_top.sv:108
                                           -> updown_counter.sv:16
```

The head of a definition is free. The chunker already emits a container's
head as its own unit, so for HDL those lines are the ports and the
parameters and for C they are the signature; nothing parses an interface.
What makes the body reachable is the naming: a container's inner units are
named after it, so the separator match that finds `RDEPENDS:${PN}` from
`RDEPENDS` also gives a module's whole extent from its header.

A `defined_at` of `-` means the index holds no definition of that name,
which is what a call into a library looks like. It is not a failure and
nothing guesses at one.

Three things the measurement changed, over `meta-virtualization` and the
WEFT counter example. Ordering what a place reaches by line filled all
twelve slots of `RDEPENDS` with `SUMMARY`, `DESCRIPTION`, `HOMEPAGE`,
`SECTION`, `LICENSE` and `DEPENDS` from the first fourteen lines of
`openvswitch.inc`, and the `inherit autotools` on line 75 — the one
structural fact in the file — did not make the budget; the order is now a
fixed precedence over relation kinds, what a file pulls in first and a
variable assignment last. One ranked list then spent the whole budget on
the file with the most edges and said nothing about the other three, so
each place answers before any place answers twice. And a `.bbappend`
inherits at the top and overrides at the bottom, so the unit around the
override holds neither: the rest of the file is read for the structural
kinds alone.

Through the scout, with the four-layer question, three steps and 18 seconds:

```
which layer overrides EXTRA_OECONF
  find_references -> expand_symbol -> finish        verified, 3 steps, 17.7s
  places  libxcrypt_%.bbappend:4  openvswitch.inc:36
          xen-tools.inc:743       xen.inc:123
```

All four are right, and the loop reached for the second hop without being
told to. The prose beside them is weaker than the places: one run wrote
"the layers that inherit from EXTRA_OECONF", which has the direction
backwards. Only `places` is grounded, and grounding gates what it checks —
the same limit a verifier has.

### Grading an answer, and what it is for

Nothing here records a judgement except `grade_run`. Delegate, judge the
answer as you were going to anyway, and say whether it was good, wrong or a
refusal — with a correction when you know what it should have said. The
verdict lands beside the run it is about, and `shuttle-dataset` exports the
triples as JSONL: the input, what the local model produced, and what you made
of it.

That file is the whole of M7 that can exist before the stack has been used.
Training an adapter on two cases would measure nothing, and the milestone
asks for one that beats the base model on the golden set by a measurable
margin. What the call cost is stripped from the exported output: a token
count is a fact about the delegation, and a model has no business learning to
emit one.

### Jobs, when waiting is not an option

Summarising a 35 KB file takes about a minute on the reference machine, and
asking for four hundred words of it took four. A client that blocks on an
answer that long gives up before the server produces one.

`start_job` takes the name of one of the four file tools and the arguments
you would have passed it, and returns a job id in a hundredth of a second.
`get_status` says whether it is queued, running, done or failed, and how long
it has waited and run. `get_result` returns exactly what the tool would have
returned, run file and token count included; a job still running is an error
saying to poll again rather than an empty answer, and a failed one carries
its reason in the status.

There is one worker. The long server answers one request at a time, so a
queue that pretended otherwise would only move the waiting somewhere less
visible. The newest sixty-four settled jobs are kept and the rest forgotten.

Everything a job does is recorded exactly as a direct call is: the same
report limit, the same run file, the same line in the audit log. The two
paths differ in who waits and in nothing else.

### Drawing again when the answer does not check out

Two of the tools can verify their own answers, and both report `verified`
and `attempts`.

`ask_file` compares every quotation against the text the model was shown.
`extract` compares every string value it returns against the same text: the
server holds the answer to your schema so its shape can never be wrong, but a
value the text does not contain was not read out of it. Numbers and booleans
are the model's reading rather than a span of the text and are left alone, as
are values too short to be a quotation.

A sample that fails is drawn again, warmer, up to `attempts` times, two by
default. The first attempt uses the profile's own temperature, so a correct
answer costs exactly what it did before: over the evaluation set this comes to
**1.00 attempts per case**. What still fails after the last attempt is
reported in `values_not_in_source` or `quotes_not_in_source` rather than
passed off as read.

A refusal is never drawn again. Sampling until something comes back would
reward the model for inventing an answer to a question the file does not
address, which is the failure the checking exists to prevent.

### The n-gram lookup, and why it is off too

`--ngram` adds a second speculator ahead of the draft model: a lookup for
the last few tokens of the prompt, drafting whatever followed them there. It
needs no model and no VRAM, which on a 4 GiB card promised speculation even
with `--no-draft`, where the draft model does not fit.

It gains nothing here. Comparing within the same number of offloaded layers:

| Speculators | GPU layers | extract | summarise |
|---|---|---|---|
| `ngram-simple,draft-simple` | 11 | 4.7 / 4.0 s | 12.6 / 9.5 s |
| `draft-simple` | 11 | 4.5 / 4.2 s | 14.2 / 9.4 s |
| `ngram-simple` | 19 | 7.1 / 6.6 s | 12.9 / 11.5 s |
| none | 19 | 6.8 / 6.3 s | 13.3 / 9.1 s |

Two runs of each. With a draft model the lookup makes no difference; without
one it is marginally the slower choice. Narrowing by pattern leaves a prompt
of a few hundred tokens and an answer of two fields or eighty words, so there
is little for a lookup to copy and few chances to copy it.

The same table says something else worth having: eleven layers with a draft
beat nineteen without one, 4.0 seconds against 6.3. **A draft model is worth
more than eight layers of GPU offload on this card.**

### The cascade, and why it is off here

`ask_file` and `extract` take `cascade_profiles`, a comma-separated ladder
such as `fast,extract`. Each profile is asked in turn until one verifies its
own answer. If none does, the last attempt comes back with `escalate` set and
the whole ladder recorded, which is the third rung: the question is handed
back to you with what was tried, rather than answered with something that did
not check out.

Measured over the eleven verifiable cases of the evaluation set:

```
direct on long    75s   15901 local tokens
cascade fast,long 243s  17617 local tokens

0 of 11 climbed past the first profile, 0 came back unverified
```

The mechanism does what it should: the small server verified its own answer
every time, so nothing escalated, which satisfies the plan's threshold of
fewer than one in five by a wide margin. The ladder is still the wrong choice
on this machine, because it is 3.2 times slower. A cascade saves time when
the cheap profile is cheap, and here the small server is CPU-only while the
large one has layers on the GPU and a draft model in front of it. One case
shows it plainly: the question `LICENSE` does not answer takes 1.8 seconds
directly and 116.5 through the ladder, because the small server has to read
7600 tokens at ninety-odd a second first.

So it stays off unless asked for. It earns its place on a machine where the
small model is genuinely the faster one, or when the GPU is busy with
something else and `shuttle-long` is the server you cannot have.

`python -m evals.cascade` repeats the comparison.

### Sessions, for a document you will ask more than once

Each of the file tools reads its file from scratch. Asking the same document
five questions reads it five times, and on this machine reading 1404 tokens
costs two and a half seconds before any thinking starts.

`session_open` reads it once and keeps its KV cache under a name.
`session_ask` restores that cache instead of sending the document again.
`session_close` drops the cache and keeps the record. Measured on
`CONTRIBUTING.md`:

```
session_open     2.4s  1404 tokens read and cached
session_ask      3.2s  1404 restored
session_ask      2.8s  1404 restored
session_ask      1.7s  1404 restored
```

**The transcript is the source of truth and the dump is only a cache.** Every
exchange is appended to `.shuttle/sessions/<id>.jsonl`, and if the dump is
gone the document is simply read again: the answers are the same, only
slower.

If the file has changed since the session opened, the dump is thrown away
rather than trusted, the answer comes from the new text, and the reply says
so in `source_changed`. A cached answer about an older version of a file is
worse than a slow answer about the current one. The cache is rebuilt once the
new text has been read, so only the first question after a change pays for
it.

A document the server cannot hold is refused when the session opens, with
both numbers and the three ways out, rather than reaching you as the HTTP 400
llama-server answers an oversized prompt with. A session is opened once and
asked many times, so that is where the check belongs.

The dump is 305 MiB. A session left open is disk spent on a document nobody
is asking about, which is what `session_close` is for.

### Profiles

A profile names one way of asking: which server answers, how it is sampled,
and how much of the answer comes back. Every tool takes one.

| Profile | Server | Temperature | Used by default for |
|---|---|---|---|
| `long` | `shuttle-long` | 0.2 | `summarize_file`, `ask_file` |
| `fast` | `shuttle-fast` | 0.2 | nothing; ask for it when concurrency matters |
| `extract` | `shuttle-long` | 0.0 | `classify_file`, `extract` |
| `brainstorm` | `shuttle-long` | 0.8 | `brainstorm` |

Schema-bound work is not sampled: there is one right shape for the answer and
temperature can only move it away from that. Brainstorming is the one case
where sampling is the point rather than a hazard, which is why it has a
profile of its own.

`~/.config/shuttle/profiles.toml` overrides a key or adds a profile; what it
does not mention keeps the built-in value.

```toml
[long]
temperature = 0.4

[strict]
server = "fast"
temperature = 0.0
report_tokens = 400
```

A key no profile has, a server that is not one of the two, a temperature
outside 0.0 to 2.0, or a report of no tokens is an error naming the profile
rather than a value quietly ignored.

### What comes back, and where the rest goes

A tool returns a report of at most `report_tokens`, a thousand by default,
measured by the server's own tokeniser rather than estimated. The whole of
what the model produced goes to `.shuttle/runs/<id>.md` next to the working
directory, together with the request that produced it, and the report carries
that path. A cut report says so in `report_cut`. Structured fields are never
cut: a schema bounds them already, and half a JSON object is worth less than
none.

Every call appends one line to `.shuttle/audit.jsonl`, whether it answered or
failed, with the arguments, the seconds, the tokens spent and the run file.

`brainstorm` is the exception to all of this and says so in every reply:
`grounded` is false and the note reads "suggestions, not findings; nothing
here was verified". The schema fixes the shape, so a caller always receives a
list of strings, but nothing in them was checked against anything. The rule
of the project is that a local model is used where a verifier exists; here
the caller is the verifier, and an idea from it must not be treated as
something a file says.

Given a file it is noticeably better than given nothing. Asked what could go
wrong in `probe_publish`, it named an incompatible Podman or netavark, a
timeout on a misconfigured network, and Podman missing from PATH. Asked
without a file how to test the installer on a machine with no GPU, it
suggested GPU emulators, when the answer is the `--no-gpu` flag the installer
already has.

`ask_file` also reports `quotes` and `quotes_grounded`. Every quoted or
backticked span in the answer is compared against the text the model was
shown, and any that is not there is listed in `quotes_not_in_source`. This
costs nothing and catches the worst failure available: on the first day of
this project the 8B server answered a question about the installer wrongly
and supported it with a sentence that appears nowhere in the file. A
fabricated citation reads exactly like a real one, so it is checked rather
than trusted.

### Using it well

Delegate volume, not judgement. The local models are good when the material
comes from you and the shape of the answer is fixed by you; they are poor at
deciding what matters. Keep the choosing, hand over the reading.

- **Give a path, never contents.** Reading a file yourself and pasting it into
  a tool call spends exactly the tokens the tool exists to save.
- **Pass a `pattern` whenever you can name what you are after**, and give
  `until` as well whenever the passage has an end you can name:
  `^[a-z_]+\(\)` for the next shell function, `^## ` for the next section,
  `^\d+\.\d+ ` for the next clause. The same answer costs 408 tokens
  instead of 13079, and the region stops before the neighbour it would
  otherwise be confused with.
- **Ask several questions of one region rather than one question of a file.**
  The server keeps the cache of a prefix it has already read, so the second
  question about the same text is about twice as fast.
- **Prefer `extract` and `classify_file` when you know the shape of
  the answer.** The server is held to your schema, so the reply cannot be a
  label you did not offer or a field you did not ask for. `ask_file` is
  free-form and carries no such guarantee.
- **Do not ask about anything the file does not contain.** Asked what Quadlet
  is, with no text to read, `shuttle-fast` answered that it is a character
  from an anime series by the author of *Sword Art Online*. Given a file, the
  same server answers from the file or says the file does not say.
- **Do not delegate a small file.** A call costs three to twenty seconds
  against a fraction of one for reading it directly. Delegation pays from the
  point where the file would otherwise fill thousands of tokens of context.
- **Use `shuttle-long` unless you have a reason not to.** On this machine it
  is both better and faster; see the evaluation below.
- **Read `local_tokens` in every reply.** It is the work the machine did, and
  therefore the work your context did not have to hold.
- **Read `quotes_not_in_source` when `ask_file` answers.** A quotation the
  source does not contain means the answer was reconstructed rather than
  read, and nothing else in the reply is more trustworthy than that.
- **Read the run file when the report was cut.** `report_cut` says when the
  thousand-token limit bit, and `run` says where the rest is.

A worked pair, both measured on this machine: `install.sh` at 35 KB summarised
for 13260 local tokens, and `CONTRIBUTING.md` yielding its subject-length
limit, linter and indent style for 1410. Neither file entered the caller's
context.

### How often it is right

`delegate/evals/` holds twenty cases, each a file already in this tree, a
tool, its arguments and an expectation taken from the source. Two of them
expect a refusal, which is the half that matters: a model that answers
everything scores full marks on questions that have answers, and the failure
worth catching is the confident answer to a question the file does not
address. It needs a running stack, so it is not part of CI.

```
python -m evals.run --server both
```

| Tool | `shuttle-long` | `shuttle-fast` |
|---|---|---|
| `extract` | 5 / 5 | 5 / 5 |
| `ask` | 6 / 6 | 6 / 6 |
| `summarise` | 2 / 2 | 2 / 2 |
| `classify` | 7 / 7 | **5 / 7** |
| total | **20 / 20** | 18 / 20 |
| seconds | **72** | 631 |

Those seconds were taken on an otherwise idle machine, and they are the
part of the table that does not travel. Re-run during a working desktop
session — a browser, an editor, Steam, load average between eight and ten —
the same suite takes 186 seconds on `shuttle-long` and 1008 on
`shuttle-fast`, 2.6 and 1.6 times longer, with the GPU sitting at nought
per cent because the work that is left is on the CPU. The pass counts held:
20 of 20 and 19 of 20, the one that differed being classification of a long
document, which is where both of `shuttle-fast`'s failures were. **Read the
counts as the baseline and the seconds as a property of the machine at the
time.**

Two things in that table are worth reading twice.

`shuttle-fast` is not faster. It took 631 seconds against 72, nearly nine
times longer, because `shuttle-long` has thirteen layers on the GPU and a
draft model in front of it while the fast server is pure CPU. The name
describes the model, not the latency, and on this hardware the smaller model
is the slower choice. It earns its place by answering four requests at once,
not by answering one sooner.

The small model's two failures were both `classify`, and both on a long
document: it called `README.md` licence text, and a GitHub workflow
documentation. Given a narrow region it was right every time, and it refused
both unanswerable questions as cleanly as the large one. So the small model
is usable for what it is given in a few hundred tokens and unreliable when
asked to judge a whole file, which is the same shape as everything else
measured here.

### Why the text comes before the question

Every prompt the delegate builds puts the text first and the instruction
last. That ordering decides two things.

llama-server keeps the KV cache of a prompt prefix it has already processed,
so a second question about the same text skips re-reading it. Over 12 KB of
`install.sh`, two questions took 12.3 s and 13.6 s with the question first,
and 12.1 s and 7.1 s with the text first.

It also decides whether the answer is right at all. Asked about one function
in the 35 KB installer, with the question ahead of the text, the model
answered wrongly and supported it with a quotation that appears nowhere in
the file; narrowing to fragments made it refuse instead. With the text first
and the question after it, every one of those cases answers correctly:

| What the model was given | Tokens | Time | Result |
|---|---|---|---|
| the whole file, no pattern | 13079 | 22 s | correct |
| `pattern`, six lines of context | 563 | 4 s | correct |
| `pattern`, twenty-five lines | 1634 | 6 s | correct |
| `pattern`, forty lines | 2312 | 4 s | correct |
| the one matching function | 408 | 3 s | correct |

So `pattern` is about price rather than about capability: thirty-two times
fewer tokens and seven times faster for the same answer. Use it whenever you
can name what you are looking for, and ask several questions of one region
rather than one question of a whole file.

## Configuration

Every flag is shown by `./install.sh --help`.

| Flag | Effect |
|---|---|
| `--dry-run` | print the commands and file differences, change nothing |
| `--yes` | do not ask before commands that run as root |
| `--ctx N` | context of `shuttle-long`; the default comes from RAM |
| `--ngl N` | layers of `shuttle-long` on the GPU, instead of the estimate |
| `--threads N` | CPU threads for both servers; the default is the physical core count |
| `--layers N` | layers in the long model, for the VRAM estimate (default 36) |
| `--draft-layers N` | layers in the draft model, for the same estimate (default 28) |
| `--no-gpu` | run `shuttle-long` on the CPU as well |
| `--no-draft` | disable the draft model |
| `--no-expose-direct` | do not publish the host ports; reach the servers only over the network |
| `--long-model REPO FILE` | a different Hugging Face GGUF for `shuttle-long` |
| `--fast-model REPO FILE` | the same for `shuttle-fast` |
| `--draft-model REPO FILE` | the same for the draft model |
| `--bench-tokens N` | prompt size for `bench`; repeat the flag for several |
| `--prefix DIR` | where the models and the KV cache live |
| `--force-distro arch\|ubuntu` | skip detection; for tests only |

`HF_TOKEN` in the environment is sent to Hugging Face. It is written to a
header file under a temporary directory, so it appears neither in the printed
commands nor in `install.log`.

The two `.env` files are plain `KEY=value` and are read by the container at
start. They are regenerated by the `quadlets` phase, so edit the flags rather
than the files. `quadlets/` holds the three units this machine produced, as
examples to read rather than to copy.

The delegate takes a handful of environment variables, all with defaults
that work on a machine the installer set up:

| Variable | Effect |
|---|---|
| `SHUTTLE_HOME` | where runs, sessions and the audit log go; `./.shuttle` by default |
| `SHUTTLE_INDEX` | the index file; `$SHUTTLE_HOME/documents.sqlite` by default. Point it at WEFT's `documents.sqlite` to share one |
| `SHUTTLE_SEARCH_TTL` | seconds a search result is served from the cache, 3600 by default; 0 switches the cache off |
| `SHUTTLE_ANSWER_TTL` | seconds an `ask_file` answer is served from the cache, 86400 by default; 0 switches it off |
| `SHUTTLE_AUDIT_BYTES` | size at which the audit log rotates, 8 MB by default; 0 never rotates |
| `SHUTTLE_AUDIT_KEEP` | generations of the log kept beside the live one, 4 by default |
| `SHUTTLE_EMBED_URL` | Ollama, for the embeddings; `http://127.0.0.1:11434` |
| `SHUTTLE_EMBED_MODEL` | the embedding model, `bge-m3` |
| `SHUTTLE_DOCS_IMAGE` | the extraction and parsing image, `localhost/shuttle-docs` |

To change a model, pass the repository and the file name and rerun the two
phases that depend on them:

```
./install.sh quadlets models verify \
    --long-model unsloth/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf --layers 36
```

The file name is checked against the Hugging Face tree API before anything is
downloaded. A name that does not exist is an error; the installer does not
guess a similar one. Pass `--layers` to match the new model, or the VRAM
estimate will be wrong.

## Verification and benchmarks

`verify` starts both units, waits for `/health`, sends one completion to each
server and saves and restores a KV slot on `shuttle-long`. It answers the
question "is this installation working at all". The `pp` and `tg` columns it
prints come from a 32-token completion and are too short to be a measurement.

`audit` is the hardening check, and it reads rather than changes: podman's
own view of `Internal=true`, a container on the shuttle network trying to
leave it, what each server publishes and where, and whether anything that is
not a `shuttle-` container shares the subnet. It exits non-zero on a FAIL.

Two egress probes, because one proves less than it looks. A name that does
not resolve is what a broken resolver says on a network that routes perfectly
well, so the second probe is a bare address and asks about the route. On this
machine:

```
audit   network                ok     shuttle has Internal=true
audit   egress-huggingface.co  ok     (6) Could not resolve host
audit   egress-1.1.1.1         ok     (7) Failed to connect to 1.1.1.1:443
audit   shuttle-long           ok     127.0.0.1:8081, loopback only
audit   shuttle-fast           ok     127.0.0.1:8082, loopback only
audit   members                ok     only shuttle- containers on shuttle
```

The same probe on the default network returns HTTP 200. Without that control
an "ok" here could have meant the probe was broken rather than the network
closed.

`bench` is the measurement. For each server and each prompt size it builds a
prompt of exactly that many tokens, sends it with `cache_prompt: false` so that
nothing is reused between runs, and records the server's own timings. A prompt
that would not fit in the slot context is skipped and named. The result is
written to `~/.local/state/shuttle/bench-<timestamp>.json` together with the
full `LLAMA_ARG_*` configuration it was measured under, so two runs can be
compared without remembering which flags were in force.

Two columns matter:

- **pp** — prompt processing, in tokens per second. The prompt is processed in
  parallel, so this is bound by compute and is high.
- **tg** — token generation, in tokens per second. Tokens are produced one at a
  time and each one reads the whole model, so this is bound by memory
  bandwidth and is one to two orders of magnitude lower. This is the number a
  user feels.

### Measured on the reference machine

NVIDIA RTX 3050 Mobile, 4 GB VRAM; Intel i5-12500H, 12 physical cores; 64 GB
RAM; Arch Linux, Podman 6.1.2, driver 615.71.09, llama.cpp build b11243.

`shuttle-long`, Qwen3-8B Q4_K_M:

| ctx | GPU layers | draft | prompt | pp tok/s | tg tok/s |
|---|---|---|---|---|---|
| 4096 | 15 / 36 | Qwen3-0.6B | 2048 | 650.5 | **19.6** |
| 4096 | 15 / 36 | off | 2048 | 739.1 | 9.1 |
| 32768 | 16 / 36 | off | 2048 | 653.8 | 7.9 |
| 32768 | 16 / 36 | off | 8192 | 621.2 | 5.5 |

`shuttle-fast`, Qwen3-1.7B Q8_0, CPU only, context 16384 over four slots:

| prompt | pp tok/s | tg tok/s |
|---|---|---|
| 2048 | 94.6 – 110.0 | 13.0 – 16.0 |
| 8192 | 73.2 – 74.8 | 7.0 – 8.3 |

The ranges are four runs of the same configuration. Both servers share the same
twelve cores, and `shuttle-long` computes 21 of its 36 layers there, so the
figure depends on what the other server was doing.

The first two rows differ only in the draft model: same context, same number of
GPU layers. Speculative decoding is worth **2.15x** on generation, at a cost of
12% on prompt processing, with 72.4% of drafted tokens accepted and a mean
accepted run of 3.10 tokens. The gain is large because 21 layers run on the CPU,
where generation is bandwidth-bound and verifying three tokens costs almost what
verifying one costs.

Rows one and three are the two configurations worth running on 4 GB of VRAM. A
larger context cannot be combined with a draft model on this card; see the next
section for why.

## Troubleshooting

**`nvidia-ctk` is installed but containers cannot see the GPU.** The CDI
specification names the driver version in every library path, 108 times on this
machine, so an upgraded driver leaves it pointing at files that were deleted.
The error surfaces as a runtime failure rather than a missing file. `detect`
reports the file as `stale` when it no longer matches `nvidia-smi`, and `gpu`
rewrites it; running `./install.sh detect gpu` after a driver upgrade is enough.

**`no terminal to confirm '...'`.** The installer asks before every command that
runs as root and there is no terminal to ask on. This appears in a script, a
CI job or an editor-hosted shell. Either run the phase from a terminal or pass
`--yes`. On a machine that is already prepared the phase needs no root at all:
packages are installed only when one is missing, and the CDI specification is
written only when it is stale.

**`estimate_ngl` offloads fewer layers on every run.** Fixed, but worth knowing
if you see an old installation behaving this way. The `quadlets` phase replans
while the servers it is about to replace are still holding VRAM, so their
weights and KV cache were counted as memory in use. A machine planned for
sixteen layers dropped to four over three reruns with nothing reporting it.
`detect` now prints the servers' own usage on a separate row and leaves it out
of the budget.

**The draft model is loaded and never used.** Setting
`LLAMA_ARG_SPEC_DRAFT_MODEL` alone does not turn speculative decoding on. The
speculative types default to `none`, and the only inference llama.cpp performs
reads the draft GGUF for an MTP head, which a plain Qwen3-0.6B does not carry.
The server starts, answers correctly and holds the draft's VRAM for nothing.
`LLAMA_ARG_SPEC_TYPE=draft-simple` is required and the installer sets it; the
symptom of it missing is that `--no-draft` changes nothing in `bench`. The
variable for the draft model is `LLAMA_ARG_SPEC_DRAFT_MODEL`, not
`LLAMA_ARG_MODEL_DRAFT`; the old name is ignored silently.

**Only three layers reach the GPU at a large context.** The draft context is
taken from the target context and there is no option to cap it, so the draft's
KV cache grows with `--ctx`. A 28-layer draft at q8_0 costs 56 KiB per token:
224 MiB at `--ctx 4096` and 1792 MiB at `--ctx 32768`. On a 4 GB card the
second leaves room for three of the long model's 36 layers, which is slower
than every other setting available. The default context is derived from RAM,
which says nothing about the card, so `plan` prints a warning when under half
the layers fit. It does not lower the context on its own. Either reduce
`--ctx` or pass `--no-draft`.

**`shuttle-long.service` times out while starting.** Loading a 4.7 GiB model and
allocating its KV cache takes time, and on a cold page cache it takes longer.
`TimeoutStartSec` and the health start period are both 900 s for this reason.
If a unit is still killed, `journalctl --user -u shuttle-long.service` shows how
far it got; an allocation failure at the end of loading means the VRAM budget
was wrong for the machine, not that the timeout was short.

**Rootless Podman fails before any container starts.** Check
`grep "^$USER:" /etc/subuid /etc/subgid`. Without a range there, the user
namespace a rootless container needs cannot be built. The `host` phase adds
`100000-165535` and runs `podman system migrate`, unless part of that range is
already taken by another user, in which case it stops and says so rather than
overlapping two users' container UIDs.

**The published ports do not answer.** Whether a rootless `PublishPort` reaches
a container on an `Internal=true` network depends on the Podman and netavark
versions, so `quadlets` measures it instead of assuming: it creates a temporary
internal network, starts a listener on it and tries the port from the host. On
the reference machine, Podman 6.1.2, the probe succeeded and the units were
written without a second network. Where the probe fails, the installer adds
`Network=podman` to both units and says so. That makes the ports work, and it
also **gives the containers egress** — the isolation the internal network
provides is gone, and a model processing untrusted text can reach the network.
Use `--no-expose-direct` and talk to the servers over `shuttle.network` if that
trade is not acceptable.

## Roadmap

The order of work is M0 to M1 to M2, then M3 and M5, then M6 and M7. M4 runs
alongside from M2. M8 is independent of all of it.

- **M0 — inference layer.** Done: the two servers, the internal network and
  the installer that produces them.
- **M1 — delegate.** Done: the MCP server, its tools, and the measured
  reduction above that completes it.
- **M2 — sessions, prefix cache and reliability.** Partly done. The
  evaluation set, the sessions over the KV cache, the schema-constrained
  output, best-of-n with a verifier, the cascade and the n-gram lookup are
  all in. The last two are measured and left off by default, for the reasons
  given above. Nothing of M2 is outstanding.
- **M3 — documents and code.** Done: the `shuttle-docs` container, the
  reading, the Tree-sitter chunking for fourteen languages, and the hybrid
  index with its tools (see *PDFs*, *The index* and *Source* above). A
  register table comes out of a datasheet as valid JSON with the values the
  page does not contain named rather than returned. The digestion the plan
  asks for was overnight work and arrived with M6.
- **M4 — isolation and hardening.** Partly done: `install.sh audit` proves
  there is no egress, the audit log rotates, and `shuttle-backup` writes the
  index and the transcripts into one archive. The remaining two parts need a
  decision rather than code. The milestone's criterion is that only the
  delegate on 127.0.0.1 with a token is reachable, and the delegate reaches
  the servers through those same loopback ports from the host, so closing
  them would cut it off; the token assumes an HTTP delegate where this one
  speaks stdio, on which a bearer token means nothing.
- **M5 — code graphs and constrained agents.** The graph, the two-hop
  expansion, the scout, the bounded loop and all six presets the design
  names are in. Two of the three criteria are met: `repo-scout` answers both
  of its questions with the right file and line, and `verify-loop` converges
  on five of five seeded defects. The third is measured and **not** met —
  `doc-extract` reaches 21 of 27 fields, 78 %, over five real reports whose
  expected values were parsed from their own tables, against the 90 % asked
  for. The one document it fails is diagnosed above; the fix is in what the
  loop sends as its query, not in the index or the model.
- **M6 — batch work.** Partly done: the timer, the watched directories, the
  overnight indexing of both documents and source, a morning report short
  enough to read in a thousand tokens, and the answer cache — narrower than
  the plan asks for, for the reason given under *Asking the same thing
  twice*. The KV dump refresh is in and off by default, measured to be worth
  about a second against 305 MiB a document; see *The night* above. The
  criterion that the report is in place before 07:00 for five consecutive
  days is a matter of five days passing rather than of code.
- **M7 — distillation.** The set exists and fills as the stack is used; see
  *Grading an answer* above. The training waits on data rather than on code:
  QLoRA on Qwen3-1.7B, evaluated against the golden set, the adapter exported
  as GGUF with the base model's hash pinned to it.
- **M8 — scaling.** The cluster works and was measured on one machine;
  `containers/Containerfile.shuttle-rpc` builds it, because no official
  llama.cpp image has RPC compiled in. Showing it is worth anything needs a
  second machine, for the reason under *The cluster* below. The backend swap
  is dropped: the delegate needs five llama-server endpoints and another
  server offers one of them.

## Agents, and the one rule they are built around

A loop that decides for itself when it has finished reports success. An 8B
model is confident in exactly the way that makes that dangerous, so a preset
without a verifier is refused rather than run — the design's own rule, and
the reason this is a short list of presets rather than a general agent.

Three kinds of verifier count, and each can refuse on its own: a schema the
report has to satisfy, a grounding check that every citation came from a tool
answer, and a command that has to exit zero. `agent_presets` says which a
preset has.

```
repo-scout    steps=10  a report schema; grounding places
doc-qa        steps=8   a report schema; grounding cited
doc-extract   steps=12  a report schema; grounding the field values
doc-compare   steps=12  a report schema; grounding the quotations
log-triage    steps=12  a report schema; grounding the error lines
verify-loop   steps=14  bash -n
```

The agent's own tool list is smaller than the delegate's and shaped for a
model: `search_docs`, `search_code`, `find_references`, `expand_symbol`,
`read_section`, `read_lines`, `write_file`, `run_verifier`, `finish`.
`read_section` is the one a document preset cannot do without — a search
says which section holds the answer and spends three hundred tokens on the
hit, which is not enough to read a register table or an error line out of.

**`verify-loop` on five seeded defects.** Five shell scripts, one syntax
error each, `bash -n` as the verifier, which M5 asks to converge on at least
three of:

```
one.sh    verified  7 steps  26s        four.sh   verified  4 steps  16s
two.sh    verified  7 steps  26s        five.sh   verified  8 steps  29s
three.sh  verified  4 steps  15s
```

Five of five, every one leaving a file `bash -n` accepts, and every fix the
minimal one: the missing `fi`, `}`, `done`, `done`, and for the `case` an
`esac` with a default branch. Three of the five leave no newline at the end
of the file.

**These numbers replace an earlier set that was not measuring anything.**
The transcript used to live in the directory the verifier is handed, so
SHUTTLE's own `steps.jsonl` was passed to the verifier as a file to check —
and `bash -n` reads its first file argument and treats the rest as positional
parameters. Whether the agent's work was checked depended on its name:
`one.sh`, `four.sh` and `five.sh` sort before `steps.jsonl`, `two.sh` and
`three.sh` sort after it. `two.sh` came back byte for byte as it went in and
the run called itself verified in four steps. The agent now writes into
`<run>/out` and the verifier is mounted there; the transcript stays beside it
and is nobody's input. The extra steps above are the loop doing the thing it
exists for — being refused and trying again.

The earlier text here also reported `four.sh` rewritten as an `if/elif` and
`two.sh` losing its shebang, as evidence that a verifier gates only what it
checks. That evidence was an artefact of the same bug: with nothing refusing
it, the model rewrote freely. **The principle stands and the examples do
not.** A syntax verifier still accepts a file whose logic is wrong, which is
the thing to remember before putting one in a preset.

**`repo-scout` on the milestone's two questions.** Every place it reports is
correct; on the variable question it found 8 of the 9 assignments the index
holds, so it is sound rather than exhaustive.

```
who instantiates clk_tick     verified  4 steps  10s
                              counter_top.sv:97, counter_top.sv:122
which file assigns RDEPENDS   verified  4 steps  13s
                              openvswitch.inc:16, :19, :20, :21, :22
                              openvswitch_git.bb:7, :107, :108
```

With `expand_symbol` in its allowlist the scout reaches for the second hop
itself. "What does counter_top instantiate, and where is each of those
modules defined" is three steps and 19 seconds — `find_references`,
`expand_symbol`, `finish` — and the three modules come back with the right
file and line for each. See *Two hops* above for where the prose is weaker
than the places.

**The document presets, and where each stops.** `log-triage` on a real
`make` log with one error among thirteen warnings from six other files:

```
what failed in build.log      verified  3 steps  19s
                              src/ring_bad.c:10:25: error: expected ';'
                              before 'return'
```

The right line, and the first failure rather than the last. On a clean
timing report the same preset does not converge at all — three runs, twelve
steps each, nothing reported. Evidence of a failure can be found; evidence
of its absence has no stopping rule. What counts as a failure also differs
between vendors, and that is what the preset's verifier command is for.

Only the error lines are grounded, not the verdict, the advice or `first`.
The first failure is a claim about order, and on a log with nothing wrong
there is no line to quote: grounded, it refused a correct `"verdict":
"passed"` three times over the words the model had to put somewhere.

**`doc-extract` against the milestone's criterion.** M5 asks for 90 % of
fields over five reference documents. The five are real Quartus reports from
two unrelated projects, and the expected JSON is parsed out of their own
boxed summary tables — a reference a model helped write is not a reference.
Twenty-seven fields in all, scored the way grounding compares, with space
removed and case dropped:

```
counter.flow.rpt  6/6  verified  3 steps  22s
counter.map.rpt   6/6  verified  3 steps  21s
counter.fit.rpt   6/6  verified  3 steps  23s
counter.sta.rpt   3/3  verified  3 steps  20s
top.flow.rpt      0/6  refused   8 steps  51s

21 of 27 = 78 %, so the criterion is not met
```

Four documents exactly, every value as the report writes it, including a
`Fitter Status` of `Successful - Sat Aug 22 17:58:59 2026` copied whole. The
fifth fails for a reason worth naming: the loop sends the whole request as
its search query, and `From top.flow.rpt pull these fields exactly as the
report writes them: Revision Name, …` ranks `Flow OS Summary` above `Flow
Summary`. Asked with the field names alone the right section is first, at
0.0328 against 0.0141 — so the index is not the problem and neither is the
model's reading. It is what gets asked.

It is also the better failure of the two available. Before a search could be
narrowed to a document, the same case returned `counter`'s revision, entity
and device with grounding's approval; now it returns nothing and says so.

Asked to extract fields that live in two different sections of one document,
it returns the ones in the section it opened — three of six over three runs,
each value right. **Sound rather than complete, in both directions.**

Two things it is worth knowing before trusting a shape. A field asked to
hold something and holding nothing used to verify: an empty object satisfies
`required` and grounds against anything, having nothing in it to look for,
so a run that found no field at all reported itself verified. The schema's
own `minProperties` and `minItems` now say so, which tells the decoding
grammar the same thing the check is told. And taking the wrong path away
from a model works where telling it not to take it does not: told to list a
document's sections first, it called `search_docs` anyway — but removing
`search_docs` from the preset took the score from 21 of 27 to **0 of 27**,
because without it the model asked `read_section` for a section called
`fields`. That change was reverted on the measurement.

**Requiring the fields made it worse, not better.** An operator preset naming
all six as `required` did not send the loop looking for the three it had not
read. It filled them — two as `"not specified"` and one copied out of the
Fmax table as the clock's frequency — and spent fourteen steps and two and a
half minutes being refused, twice over. Under-reporting with every value
right is the better failure, which is why the built-in shape takes its fields
from the request instead.

That run also shows what grounding does not do. The two `"not specified"`
values were refused, because the report does not contain them. The clock
frequency copied from the Fmax row was **not** refused, because `154.23 MHz`
is in the source — just not in that row. **Grounding checks where a string
came from, not which field it belongs in.**

**Two things that mattered more than the loop.** Putting the report schema
inside the `finish` tool, so the shape is held by the decoding grammar rather
than only checked afterwards: before that, one run invented its own field
names and another returned places as objects, and both spent their whole
budget being refused; after, a run takes four steps instead of seven to ten.
And accepting a tool call the model writes into its message instead of
calling — refusing that is not integrity, it is a formatting argument the loop
loses, and it cost six of ten steps on a run that already had the answer. The
transcript records which channel was used.

Bounded four ways, because each runs out first in a different failure: steps
for a model that keeps finding more to do, tokens for one writing an essay
every step, seconds for a verifier that hangs, and a repeated lookup for the
simplest way to go nowhere. That last one was counting failures only, so a
successful call reset it: `doc-compare` over two corners of a timing report
called `search_docs` with identical arguments seven times in a row, every
call successful, and spent all twelve steps and 206 seconds on it. A lookup
asked the same thing a third time now ends the run and the reason names the
tool; `run_verifier` and `write_file` are exempt, because asking the
verifier again after a write is the shape of `verify-loop` rather than a
loop standing still. The reply says which bound stopped it. The
verifier itself runs in its own container with no network and the output
directory read-only, because a verifier command is arbitrary code out of a
configuration file.

## The cluster, and why localhost is the wrong place to measure it

M8 wants a pool of machines serving one model through llama.cpp's RPC. Two
things had to be found out before any of that, and both were measurements
rather than readings of documentation.

**No official image has RPC.** `ghcr.io/ggml-org/llama.cpp:server` carries
neither the `--rpc` flag nor the worker binary. `:full` documents the flag in
its own `--help` and then refuses it: `error while handling argument
"--rpc": RPC not supported in this build`. So the pool has to be built, which
`containers/Containerfile.shuttle-rpc` does — CPU only, because the GPU
belongs to `shuttle-long`. The worker's CMake target is `ggml-rpc-server`,
not the `rpc-server` that every piece of documentation calls it.

**It works, and on one machine it is slower.** Two workers on loopback with
four threads each, against the same model and image served directly with
eight:

| | pp tok/s | tg tok/s |
|---|---|---|
| no RPC, 8 threads | 144, 149 | 40.9, 41.2 |
| two RPC workers, 4 threads each | 123, 128 | 38.6, 38.7 |

15 % off prompt processing and 6 % off generation, which is what splitting
eight cores into two groups of four and putting TCP between them should cost.
The plumbing is proven and the benefit is not, because there is no benefit to
find here: a pool adds the memory of other machines, and both halves of this
one were already the same memory. A model too large for one machine, on two
machines, is the experiment — and it needs the second machine.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md). It follows the Linux kernel's process
documents, and lists the coding style, the commit format and the tests that
gate every change.

## License

GPL-3.0-only. See [LICENSE](LICENSE).
