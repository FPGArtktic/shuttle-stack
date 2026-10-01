# SPDX-License-Identifier: GPL-3.0-only
"""Work that outlives the call that asked for it.

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
        now = time.time()
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
            job.started_at = time.time()
        try:
            answer = call(**job.arguments)
        except Exception as error:  # noqa: BLE001 - kept for the caller
            with self._lock:
                job.state = FAILED
                job.error = f"{type(error).__name__}: {error}"
                job.finished_at = time.time()
                self._forget_old()
            return
        with self._lock:
            job.state = DONE
            job.result = answer
            job.finished_at = time.time()
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

    def find(self, job_id: str) -> Job:
        with self._lock:
            if job_id not in self._jobs:
                raise JobError(
                    f"no job {job_id!r}; the last {self._keep} to finish "
                    "are kept and the rest are forgotten"
                )
            return self._jobs[job_id]

    def collect(self, job_id: str) -> dict:
        job = self.find(job_id)
        with self._lock:
            if job.state in (QUEUED, RUNNING):
                raise JobError(
                    f"job {job_id} is {job.state}; ask get_status again"
                )
            if job.state == FAILED:
                raise JobError(f"job {job_id} failed: {job.error}")
            return (job.result or {}) | job.report()

    def pending(self) -> int:
        with self._lock:
            return sum(
                j.state in (QUEUED, RUNNING) for j in self._jobs.values()
            )

    def wait(self, timeout: float = 30.0) -> None:
        """For tests: block until nothing is in flight."""
        deadline = time.time() + timeout
        while self.pending() and time.time() < deadline:
            time.sleep(0.01)
