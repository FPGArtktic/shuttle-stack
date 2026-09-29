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

M0 is the inference layer. M1 is the delegate that puts it in front of an
agent.

| Component | State |
|---|---|
| `install.sh` phases | working — `detect plan host gpu quadlets models verify bench status`, each runnable on its own |
| `shuttle-long` | working — Qwen3-8B Q4_K_M, partial GPU offload through CDI, one slot |
| `shuttle-fast` | working — Qwen3-1.7B Q8_0, CPU only, four parallel slots |
| Speculative decoding | working — Qwen3-0.6B draft, 2.15x on generation where it fits in VRAM |
| Internal network | working — `10.89.7.0/24`, `Internal=true`, no route out |
| Direct host ports | working — `127.0.0.1:8081` and `:8082`, measured to reach an internal network |
| KV slot save and restore | working — `/slots/{id}?action=save`, verified round trip |
| Model download | working — size and sha256 from the Hugging Face tree API, resumable |
| Benchmarks | working — `bench` writes JSON with the configuration it measured |
| `shuttle_status` | working — which servers answer, which model each holds, how large its context is |
| `shuttle_summarise` | working — folds a file larger than the context into one summary |
| `shuttle_classify` | working — labels enforced by a schema, votes across parts, reports agreement |
| `shuttle_extract` | working — fields enforced by a schema; refuses a file that needs more than one part |
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
expose. A session can be suspended and resumed only if the KV cache can be
written to disk and read back, structured output is reliable only with a
grammar the server enforces, and a caller can size its own requests only if
`n_ctx` per slot is stated rather than inferred. Those three are the whole
argument; everything else Ollama does, it does well.

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

To reach the servers from an agent rather than by hand, register the delegate
the last phase installed:

```
claude mcp add --scope user shuttle -- ~/.local/bin/shuttle-delegate
```

## How it works

```
            MCP client (Claude Code, Claude Desktop, ...)
                               |
                        stdio: four tools
                               |
                   shuttle-delegate, on the host
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

Every tool takes a **path**, not text. The file is read on this machine, split
if it does not fit the server's context, and only the result crosses back. An
agent that reads a file itself and pastes the contents into a tool call spends
exactly the tokens the project exists to save, so the tool descriptions say so
and the instructions repeat it.

| Tool | Takes | Returns |
|---|---|---|
| `shuttle_status` | nothing | which servers answer, their models and contexts |
| `shuttle_summarise` | path, words, focus, server | one summary, however many parts the file needed |
| `shuttle_classify` | path, labels, question, server | one of the labels, the vote and the agreement |
| `shuttle_extract` | path, JSON schema, instructions, server | the fields, in the shape asked for |

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
| `--no-draft` | disable speculative decoding |
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
than the files.

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
server and performs a KV save and restore on `shuttle-long`. It answers the
question "is this installation working at all". The `pp` and `tg` columns it
prints come from a 32-token completion and are too short to be a measurement.

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

- **M1 — delegate.** Done: see *Delegating* above.
- **M2 — sessions and KV.** Named sessions on top of the slot save and restore
  that M0 verifies, so a long context survives a restart.
- **M3 — WEFT.** Shared conventions with
  [WEFT](https://github.com/FPGArtktic/weft-mcp) so both tools can be used by
  the same agent.
- **M4 — isolation.** Per-caller limits and an audit trail for what was sent to
  the local models.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md). It follows the Linux kernel's process
documents, and lists the coding style, the commit format and the tests that
gate every change.

## License

GPL-3.0-only. See [LICENSE](LICENSE).
