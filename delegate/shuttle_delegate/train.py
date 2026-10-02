# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mateusz Okulanis
"""QLoRA on the graded set, on the card this machine has.

M7 wants an adapter trained on what Claude made of the local model's
answers. The set is `shuttle-dataset`'s JSONL; this is the other end
of it.

Two decisions are arithmetic rather than taste.

The base model is Qwen3-1.7B and not the 4B the plan also named. A
QLoRA step holds the base in 4-bit NF4, the adapter with its
gradients and two Adam states, the activations, and a CUDA context: on
this card that is about 1.9 GB for the 1.7B and about 3.7 GB for the
4B, against 4096 MiB total. The second number is not a fit, it is a
coin toss, and the first leaves room for a sequence worth training on.

Gradient checkpointing is on. It is off by default in every QLoRA
recipe and it is the usual reason a small card runs out: without it
the activations alone pass a gigabyte at a sequence of a thousand.

A training example is the prompt the tool actually sent, rebuilt from
the recorded request with the template the tool uses, and the answer
the model gave. Writing a second prompt format here is how a
fine-tune comes out useless -- the adapter would learn a shape it
never sees again. The file is read again to rebuild it, so an example
whose document has changed since is dropped: the recorded answer
quotes the text, and if the quotations no longer ground the pair no
longer belongs together.

    shuttle-train --dry-run
    shuttle-train --out ~/.local/share/shuttle/adapters/first
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import dataset, grounding, tasks
from .dataset import GradeError
from .indexing import IndexingError
from .retrieval import CONTEXT_LINES

# The base model, as Hugging Face names it. Qwen3-4B is deliberately
# not an option here; the module docstring has the arithmetic.
BASE = "Qwen/Qwen3-1.7B"
# How many graded cases are worth training on. Seven will not move a
# benchmark, and an adapter trained on seven and reported as an
# improvement is worse than no adapter. Pass --min to go below it on
# purpose, which is what proving the pipeline needs.
MIN_CASES = 32
# How long a training example may be, and the reason is the loss and
# not the weights.
#
# The first arithmetic here counted the base in NF4, the adapter with
# its gradients and Adam states, the activations under checkpointing
# and a CUDA context, called it 1.9 GB of the card's 4096 MiB and set
# this at twelve thousand characters. The run went out of memory in
# `ForCausalLMLoss`, on `logits.float()`, asking for 840 MiB it did
# not have.
#
# Qwen3's vocabulary is 151936, so one position of logits costs
# 151936 x 2 bytes in bf16 and 151936 x 4 again when the loss casts
# them, which is 0.91 MB a token with both alive. At the 1450 tokens
# that failed, that is 1.3 GB for the loss alone -- more than the
# quantised weights. On a large vocabulary and a small card the
# sequence is bounded by the loss, and the weights are the cheap part.
#
# 2048 characters is about 680 tokens, so about 0.6 GB of logits, and
# it is set from the failure rather than from a second estimate.
# Three characters to a token here, because the plan is printed on
# machines with no tokeniser loaded.
MAX_PROMPT_CHARS = 2 * 1024
# LoRA on the attention and the feed-forward projections, which is
# where a task adapter earns its parameters.
TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
RANK = 16
ALPHA = 32
DROPOUT = 0.05
# Batch one and accumulate, because the batch is what the activations
# scale with and the card decides the batch.
BATCH = 1
ACCUMULATE = 8
EPOCHS = 3
LEARNING_RATE = 2e-4


class TrainError(RuntimeError):
    """The adapter cannot be trained, and why."""


@dataclass(frozen=True)
class Example:
    """One prompt the tool sent, and the answer it was given."""

    run: str
    tool: str
    prompt: str
    completion: str

    def as_json(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "tool": self.tool,
            "prompt": self.prompt,
            "completion": self.completion,
        }


def _answer_of(case: dataset.Case) -> str:
    """What the model said, as the training target.

    The graded answer and nothing around it. `found` and the grounding
    report are the delegate's own bookkeeping; a model has no business
    learning to emit them.
    """
    said = case.answer.get("answer")
    return str(said).strip() if isinstance(said, str) else ""


def _ask_prompt(request: dict[str, Any]) -> str:
    """The prompt `ask_file` sent, rebuilt from what was recorded.

    The tool's own template and the tool's own narrowing, so the
    adapter is trained on the shape it will be asked in.
    """
    path = str(request.get("path", ""))
    question = str(request.get("question", "")).strip()
    if not path or not question:
        raise TrainError("the run recorded no path or no question")
    text = tasks.read_text(path)
    narrowed, _matched = tasks.narrowed(
        text,
        str(request.get("pattern", "")),
        int(request.get("context", CONTEXT_LINES)),
        str(request.get("until", "")),
    )
    words = int(request.get("words", 200))
    return tasks.ASK_ONE.format(
        text=narrowed,
        words=f" Answer in at most {words} words.",
        question=question,
    )


# Which tools a prompt can be rebuilt for. The others are not refused
# on principle; nobody has written their rebuild yet, and guessing at
# one is the mistake this module is arranged to avoid.
REBUILD = {"ask_file": _ask_prompt}


def example_of(case: dataset.Case) -> tuple[Example | None, str]:
    """One case as a training pair, or why it is not one."""
    if case.verdict != "good":
        return None, f"verdict is {case.verdict}"
    rebuild = REBUILD.get(case.tool)
    if rebuild is None:
        return None, f"no prompt to rebuild for {case.tool}"
    said = _answer_of(case)
    if not said:
        return None, "the recorded answer holds no text"
    try:
        prompt = rebuild(case.request)
    except (TrainError, tasks.TaskError, OSError) as error:
        return None, f"{type(error).__name__}: {error}"
    if len(prompt) > MAX_PROMPT_CHARS:
        return None, f"prompt is {len(prompt)} characters"
    # The document is read again, so it may have moved on since the
    # answer was given. The answer quotes the text; if those
    # quotations are not in it any more, the pair has come apart.
    if grounding.check(said, prompt).missing:
        return None, "the answer no longer quotes the document"
    return Example(case.run, case.tool, prompt, said), ""


def examples(
    db: sqlite3.Connection | None = None,
) -> tuple[list[Example], dict[str, int]]:
    """Every graded case that is a training pair, and what the rest were."""
    own = db is None
    db = db or dataset.connect()
    try:
        kept: list[Example] = []
        dropped: dict[str, int] = {}
        for case in dataset.cases(db):
            one, why = example_of(case)
            if one is None:
                dropped[why] = dropped.get(why, 0) + 1
            else:
                kept.append(one)
        return kept, dropped
    finally:
        if own:
            db.close()


def digest_of(*paths: Path) -> str:
    """One sha256 over these files, in the order given, read in pieces.

    M7 asks for the base model's hash beside the adapter, because an
    adapter is only meaningful against the weights it was trained on
    and nothing in a LoRA file says which those were.
    """
    if not paths:
        raise TrainError(
            "nothing to hash, and the sha256 of nothing is a pin that "
            "pins nothing"
        )
    held = hashlib.sha256()
    for path in paths:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                held.update(block)
    return held.hexdigest()


def weights_of(base: str) -> list[Path]:
    """The files the base model's weights are in.

    A base may be named three ways and only one of them is a file. A
    Hugging Face id is the usual one, and the weights behind it are
    shards in a snapshot directory -- hashing the id would pin
    nothing, and leaving the field empty would be a pin that pins
    nothing, which is the same failure wearing a hash.
    """
    local = Path(base).expanduser()
    if local.is_file():
        return [local]
    if local.is_dir():
        return sorted(local.glob("*.safetensors"))
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return []
    # One known file rather than the whole snapshot. A snapshot
    # fetched with allow_patterns is missing its LICENSE and README,
    # and `snapshot_download(local_files_only=True)` calls that
    # incomplete and refuses -- which says nothing about whether the
    # weights are there.
    try:
        anchor = hf_hub_download(base, "config.json", local_files_only=True)
    except Exception:  # noqa: BLE001 - an unresolvable id pins nothing
        return []
    return sorted(Path(anchor).parent.glob("*.safetensors"))


def plan(
    base: str = BASE,
    least: int = MIN_CASES,
    db: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """What a run would train on, without loading anything that trains.

    Printable on a machine with no torch and no card, because the
    first question is always whether there is enough to train on.
    """
    kept, dropped = examples(db)
    answer: dict[str, Any] = {
        "base": base,
        "examples": len(kept),
        "least": least,
        "dropped": dropped,
        "tools": sorted({one.tool for one in kept}),
        "longest_prompt_chars": max(
            (len(one.prompt) for one in kept), default=0
        ),
        "rank": RANK,
        "epochs": EPOCHS,
        "enough": len(kept) >= least,
    }
    if not answer["enough"]:
        answer["note"] = (
            f"{len(kept)} graded cases is not enough to train on; "
            "grade_run fills the set as answers are delegated, and "
            "--min says otherwise on purpose"
        )
    return answer


def main() -> int:
    """Print the plan, or train and write the adapter beside its hash."""
    parser = argparse.ArgumentParser(
        prog="shuttle-train",
        description="train a QLoRA adapter on the graded set",
    )
    parser.add_argument("--base", default=BASE, help="the base model")
    parser.add_argument("--out", default="", help="where to write the adapter")
    parser.add_argument(
        "--min",
        type=int,
        default=MIN_CASES,
        help="graded cases required; below it the run is refused",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the examples, train nothing",
    )
    args = parser.parse_args()
    try:
        made = plan(args.base, args.min)
    except (GradeError, IndexingError, OSError) as error:
        print(f"train: {error}", file=sys.stderr)
        return 1
    for key, value in made.items():
        print(f"{key:<22} {value}")
    if args.dry_run:
        return 0
    if not made["enough"]:
        print(f"\ntrain: {made['note']}", file=sys.stderr)
        return 1
    if not args.out:
        print("train: --out says where the adapter goes", file=sys.stderr)
        return 1
    try:
        where = fit(Path(args.out), args.base, args.epochs)
    except TrainError as error:
        print(f"train: {error}", file=sys.stderr)
        return 1
    print(f"\nadapter {where}")
    return 0


def fit(out: Path, base: str = BASE, epochs: int = EPOCHS) -> Path:
    """Train the adapter and write it beside what it was trained on.

    Everything that needs a card is in here and imported here, so the
    plan above runs on a machine that has neither. This is the one
    part of the module a test cannot exercise.
    """
    try:
        import torch
        from datasets import Dataset
        from peft import (
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise TrainError(
            f"the training dependencies are missing ({error}); "
            "uv sync --group train"
        ) from error

    kept, _dropped = examples()
    if not kept:
        raise TrainError("nothing to train on")
    began = time.monotonic()
    tokeniser = AutoTokenizer.from_pretrained(base)
    tokeniser.pad_token = tokeniser.pad_token or tokeniser.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base,
        # The training libraries are typed in part, so three calls
        # here are untyped in a strict context. Named one by one
        # rather than by relaxing the module: the rest of it is
        # ordinary code and should stay checked.
        quantization_config=BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map={"": 0},
    )
    # Before get_peft_model, which is where every recipe that runs out
    # of memory on a small card has left it out.
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)  # type: ignore[no-untyped-call]
    model = get_peft_model(  # type: ignore[assignment]
        model,
        LoraConfig(
            r=RANK,
            lora_alpha=ALPHA,
            lora_dropout=DROPOUT,
            target_modules=list(TARGETS),
            task_type="CAUSAL_LM",
        ),
    )

    def encode(one: dict[str, str]) -> dict[str, Any]:
        """The prompt masked out, so the loss is on the answer alone."""
        asked = tokeniser(one["prompt"], add_special_tokens=False)
        said = tokeniser(
            one["completion"] + tokeniser.eos_token, add_special_tokens=False
        )
        ids = asked["input_ids"] + said["input_ids"]
        labels = [-100] * len(asked["input_ids"]) + said["input_ids"]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": labels,
        }

    held = Dataset.from_list(
        [{"prompt": one.prompt, "completion": one.completion} for one in kept]
    ).map(encode, remove_columns=["prompt", "completion"])
    out.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        train_dataset=held,
        args=TrainingArguments(
            output_dir=str(out / "checkpoints"),
            per_device_train_batch_size=BATCH,
            gradient_accumulation_steps=ACCUMULATE,
            num_train_epochs=epochs,
            learning_rate=LEARNING_RATE,
            gradient_checkpointing=True,
            bf16=True,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
        ),
        data_collator=_pad(tokeniser),
    )
    trainer.train()
    model.save_pretrained(str(out))
    _written(out, base, kept, round(time.monotonic() - began, 1))
    return out


def _pad(tokeniser: Any) -> Any:
    """Pad a batch and its labels to the longest sequence in it."""
    import torch

    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        room = max(len(row["input_ids"]) for row in rows)
        out: dict[str, Any] = {}
        for key, filler in (
            ("input_ids", tokeniser.pad_token_id),
            ("attention_mask", 0),
            ("labels", -100),
        ):
            out[key] = torch.tensor(
                [row[key] + [filler] * (room - len(row[key])) for row in rows]
            )
        return out

    return collate


def _written(
    out: Path, base: str, kept: list[Example], seconds: float
) -> None:
    """Record what the adapter was trained on and against which weights."""
    held = weights_of(base)
    if not held:
        raise TrainError(
            f"the weights behind {base!r} cannot be found, so the "
            "adapter cannot be pinned to them; an adapter nobody can "
            "match to a base model is not worth writing"
        )
    made: dict[str, Any] = {
        "base": base,
        "base_weights": [one.name for one in held],
        "base_sha256": digest_of(*held),
        "examples": len(kept),
        "runs": [one.run for one in kept],
        "rank": RANK,
        "alpha": ALPHA,
        "epochs": EPOCHS,
        "seconds": seconds,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "shuttle-adapter.json").write_text(
        json.dumps(made, indent=1) + "\n", encoding="utf-8"
    )
    (out / "examples.jsonl").write_text(
        "".join(
            json.dumps(one.as_json(), ensure_ascii=False) + "\n"
            for one in kept
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())
