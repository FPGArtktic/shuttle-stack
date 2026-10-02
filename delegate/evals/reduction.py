# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mateusz Okulanis
"""What delegating one real task saves, counted rather than claimed.

The task: one sentence on what each phase of install.sh does. Done
without the delegate the file enters the caller's context whole; done
through it, ten narrow questions come back as ten sentences and the
file never arrives. Both sides are counted with the server's own
tokeniser so the numbers are the same kind of number.

    python -m evals.reduction
"""

from __future__ import annotations

from pathlib import Path

from shuttle_delegate import tasks
from shuttle_delegate.server import backend

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "install.sh"
PHASES = (
    "detect",
    "plan",
    "host",
    "gpu",
    "quadlets",
    "models",
    "verify",
    "bench",
    "delegate",
    "status",
)
QUESTION = "In one sentence, what does this phase do?"
# The next shell function definition ends the region, so a six-line
# phase is not read together with the one defined after it.
BOUNDARY = r"^[a-z_]+\(\)"


def main() -> int:
    server = backend("long")
    text = tasks.read_text(str(TARGET))
    reading = server.count_tokens(text)

    answers: list[str] = []
    spent = 0
    ungrounded = 0
    for phase in PHASES:
        result = tasks.ask(
            server,
            text,
            QUESTION,
            words=40,
            pattern=rf"^phase_{phase}\(\)",
            context=2,
            until=BOUNDARY,
        )
        answers.append(f"{phase}: {result.get('answer', '')}")
        spent += int(result.get("local_tokens", 0))
        ungrounded += len(result.get("quotes_not_in_source", []))
        print(f"{phase:<10} {answers[-1][len(phase) + 2 :][:76]}")

    returned = server.count_tokens("\n".join(answers))
    print()
    print(f"{'reading the file':<28} {reading:>7} tokens of context")
    print(f"{'ten answers instead':<28} {returned:>7} tokens of context")
    print(
        f"{'saved':<28} {reading - returned:>7} tokens "
        f"({100 - returned * 100 // reading}%)"
    )
    print(f"{'spent on this machine':<28} {spent:>7} tokens")
    print(f"{'quotations not in the file':<28} {ungrounded:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
