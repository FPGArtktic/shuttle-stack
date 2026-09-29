<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Contributing to SHUTTLE

SHUTTLE follows the Linux kernel's process documents, adapted to bash.
Read them once; this file only lists what is specific to this repository.

- [Documentation/process/coding-style.rst](https://www.kernel.org/doc/html/latest/process/coding-style.html)
- [Documentation/process/submitting-patches.rst](https://www.kernel.org/doc/html/latest/process/submitting-patches.html)

## Code style

- Indent with tabs, 8 columns wide. Lines are at most 80 columns; URLs and
  heredoc content may be longer.
- A function does one thing and fits on a screen: about 40 lines and at
  most 10 local variables, all declared `local`. A flat `case` with one
  line per option may be longer, as the kernel allows for simple switches.
  If a function needs a comment explaining *what* it does, split it.
- Local names are short (`i`, `tmp`), global names descriptive
  (`models_dir`), constants `UPPER_CASE`. No Hungarian notation, no type
  prefixes, no puzzle abbreviations.
- Comments explain *why*, not *what*. No per-function argument headers.
- Data instead of special cases: `--dry-run` takes effect in `run()`, and
  every difference between distributions lives in the `DISTRO` table of
  `install.sh`. Do not add `if dry_run` or `if distro` elsewhere.
- Errors go through `die()`: a message on stderr and a non-zero exit
  status. No fallback that hides a problem; if something cannot be done,
  say so and stop.
- `set -Eeuo pipefail`, `[[ ]]` instead of `[ ]`, quotes around every
  expansion, no `eval`, no parsing of `ls`, temporary files in a directory
  removed by a `trap`.
- Every file starts with `# SPDX-License-Identifier: GPL-3.0-only` (in
  Markdown, as an HTML comment).
- `shellcheck` reports nothing. It is this repository's `checkpatch.pl`.
  An exception needs `# shellcheck disable=SCxxxx` with its reason on the
  same line.

## Python

The delegate is the only Python in the tree; the installer stays free of
it. The same rules apply as above, with the language's own conventions:

- `ruff check` and `ruff format --check` report nothing. Together they
  are to Python what `shellcheck` is to bash.
- Lines are at most 79 columns. Public functions carry type hints.
- Errors raise; nothing is swallowed. A tool that cannot do its work
  says why, in a message a model can act on.
- The standard library first. A dependency needs a reason in the commit
  that adds it.

## Commits

- One logical change per commit. Every commit passes `tests/smoke.sh`, so
  `git bisect` never stops on a broken tree.
- Subject: `subsystem: imperative description`, at most 72 characters, no
  trailing period. Subsystems: `install`, `plan`, `quadlet`, `models`,
  `verify`, `bench`, `delegate`, `readme`, `tests`, `build`.
- Body wrapped at 72 columns. It describes the problem and why it is
  solved this way; it does not repeat the diff. No "This patch...", no
  first person.

Example:

```
plan: count the draft model against the VRAM budget

With speculative decoding the draft model is offloaded completely, so
its weights and a fixed overhead are no longer available to the long
model. Without subtracting them the estimate put three more layers on
the GPU than fit, and llama-server failed to allocate its KV cache.

Signed-off-by: Jane Doe <jane@example.org>
```

### Developer Certificate of Origin

Every commit carries a `Signed-off-by:` line with your real name and
e-mail address, certifying the
[Developer Certificate of Origin 1.1](https://developercertificate.org/).
`git commit -s` adds it from your git configuration. Other trailers, such
as `Co-developed-by:` or a tool attribution, go after it.

## Tests

`tests/smoke.sh` is the gate for every commit:

1. `bash -n install.sh`
2. `shellcheck install.sh tests/*.sh`
3. `./install.sh detect plan --dry-run --force-distro ubuntu`, then the
   same with `--force-distro arch`

It needs bash 5, shellcheck, curl and jq, and neither root nor a GPU. On
a machine with an NVIDIA GPU the plan estimates GPU layers from the model
file sizes, which it takes from the downloaded files or, before the
models phase, from the Hugging Face API; that case needs network access.

```
tests/smoke.sh
```

`tests/commits.sh BASE..HEAD` checks the commit messages of a range
against the rules above and runs `tests/smoke.sh` on each commit in a
scratch worktree:

```
tests/commits.sh origin/main..HEAD
```

CI (`.github/workflows/ci.yml`) runs `tests/smoke.sh` on every push and
pull request, and `tests/commits.sh` on pull requests. GitHub's runner
has no GPU and no user systemd session, so the phases that change the
system (`host` to `bench`) are tested on real hardware; report the
machine and the `bench` output in the pull request when you change them.

A tag `v*` publishes a release with `install.sh` and `SHA256SUMS`
(`.github/workflows/release.yml`).
