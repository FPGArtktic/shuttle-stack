# SPDX-License-Identifier: GPL-3.0-only
"""Work that outlives the call that asked for it.

Durations and the pruning order come from a monotonic clock: with the
wall clock, a step backwards — which a running machine does — reported
a negative elapsed time and could discard the job that had just
finished in favour of an older one.

A summary of a large file took four minutes on the reference machine,
and a client that waits for an answer that long will give up before the
server does. A job is started, polled and collected instead, so the
caller decides how long to care.

One worker: the long server answers one request at a time, and a queue
that pretended otherwise would only move the waiting.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .runs import new_id

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
KEEP = 64


class JobError(RuntimeError):
    """No such job, or not one that can answer yet."""


@dataclass
class Job:
    """One piece of work and whatever is known about it."""

    id: str
    tool: str
    arguments: dict
    state: str = QUEUED
    queued_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    result: dict | None = None
    error: str = ""

    def report(self) -> dict:
        now = time.monotonic()
        entry = {
            "job": self.id,
            "tool": self.tool,
            "state": self.state,
            "waiting": round((self.started_at or now) - self.queued_at, 1),
        }
        if self.started_at:
            end = self.finished_at or now
            entry["seconds"] = round(end - self.started_at, 1)
        if self.state == FAILED:
            entry["error"] = self.error
        return entry


class Queue:
    """Jobs in flight, and the last few that finished."""

    def __init__(self, workers: int = 1, keep: int = KEEP):
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._keep = keep

    def _forget_old(self) -> None:
        """Keep the newest settled jobs; the caller holds the lock.

        Pruning happens as a job finishes rather than as one starts,
        so the number of settled jobs is bounded whatever the order
        they completed in.
        """
        settled = [j for j in self._jobs.values() if j.finished_at]
        oldest_first = sorted(settled, key=lambda j: j.finished_at)
        for job in oldest_first[: -self._keep]:
            del self._jobs[job.id]

    def _run(self, job: Job, call: Callable[..., dict]) -> None:
        with self._lock:
            job.state = RUNNING
            job.started_at = time.monotonic()
        try:
            answer = call(**job.arguments)
        except Exception as error:  # noqa: BLE001 - kept for the caller
            with self._lock:
                job.state = FAILED
                job.error = f"{type(error).__name__}: {error}"
                job.finished_at = time.monotonic()
                self._forget_old()
            return
        with self._lock:
            job.state = DONE
            job.result = answer
            job.finished_at = time.monotonic()
            self._forget_old()

    def start(
        self, tool: str, call: Callable[..., dict], arguments: dict
    ) -> Job:
        job = Job(new_id(tool), tool, dict(arguments))
        with self._lock:
            self._jobs[job.id] = job
            self._forget_old()
        self._pool.submit(self._run, job, call)
        return job

    def _locked(self, job_id: str) -> Job:
        """The caller must hold the lock."""
        if job_id not in self._jobs:
            raise JobError(
                f"no job {job_id!r}; the last {self._keep} to finish "
                "are kept and the rest are forgotten"
            )
        return self._jobs[job_id]

    def status(self, job_id: str) -> dict:
        """A snapshot taken under the lock.

        Handing out the live Job let a reader catch the worker between
        two assignments and report a failed job with no reason, or a
        finished one timed to the moment of the poll.
        """
        with self._lock:
            return self._locked(job_id).report()

    def collect(self, job_id: str) -> dict:
        with self._lock:
            job = self._locked(job_id)
            if job.state in (QUEUED, RUNNING):
                raise JobError(
                    f"job {job_id} is {job.state}; ask get_status again"
                )
            if job.state == FAILED:
                raise JobError(f"job {job_id} failed: {job.error}")
            return (job.result or {}) | job.report()

    def close(self) -> None:
        """Let go of the worker.

        The pool holds a non-daemon thread and concurrent.futures joins
        it at exit, so without this the process cannot leave while a
        job is running: a client that disconnects mid-summary left an
        orphan talking to the single-slot server for minutes.
        """
        self._pool.shutdown(wait=False, cancel_futures=True)

    def pending(self) -> int:
        with self._lock:
            return sum(
                j.state in (QUEUED, RUNNING) for j in self._jobs.values()
            )

    def wait(self, timeout: float = 30.0) -> None:
        """For tests: block until nothing is in flight."""
        deadline = time.monotonic() + timeout
        while self.pending() and time.monotonic() < deadline:
            time.sleep(0.01)
