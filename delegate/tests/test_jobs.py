# SPDX-License-Identifier: GPL-3.0-only
"""Work started, polled and collected."""

from __future__ import annotations

import threading
import unittest

from shuttle_delegate.jobs import DONE, FAILED, JobError, Queue


def answer(**kwargs: object) -> dict:
    return {"answer": "done", "saw": kwargs}


def slow(gate: threading.Event) -> object:
    def call(**_: object) -> dict:
        gate.wait(timeout=5)
        return {"answer": "eventually"}

    return call


def broken(**_: object) -> dict:
    raise ValueError("it could not be done")


class QueueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.queue = Queue()

    def test_a_job_is_collected_in_the_tool_s_own_shape(self) -> None:
        job = self.queue.start("summarize_file", answer, {"path": "x"})
        self.queue.wait()
        result = self.queue.collect(job.id)
        self.assertEqual(result["answer"], "done")
        self.assertEqual(result["saw"], {"path": "x"})
        self.assertEqual(result["state"], DONE)

    def test_the_arguments_are_copied_not_shared(self) -> None:
        arguments = {"path": "x"}
        job = self.queue.start("extract", answer, arguments)
        arguments["path"] = "changed after starting"
        self.queue.wait()
        self.assertEqual(self.queue.collect(job.id)["saw"]["path"], "x")

    def test_collecting_a_running_job_says_to_poll(self) -> None:
        gate = threading.Event()
        self.addCleanup(gate.set)
        job = self.queue.start("ask_file", slow(gate), {})
        with self.assertRaises(JobError) as caught:
            self.queue.collect(job.id)
        self.assertIn("get_status", str(caught.exception))

    def test_a_failed_job_carries_its_reason(self) -> None:
        job = self.queue.start("extract", broken, {})
        self.queue.wait()
        self.assertEqual(self.queue.find(job.id).state, FAILED)
        with self.assertRaises(JobError) as caught:
            self.queue.collect(job.id)
        self.assertIn("could not be done", str(caught.exception))

    def test_an_unknown_job_is_refused(self) -> None:
        with self.assertRaises(JobError):
            self.queue.find("no-such-job")

    def test_a_status_reports_the_time_it_took(self) -> None:
        job = self.queue.start("extract", answer, {})
        self.queue.wait()
        report = self.queue.find(job.id).report()
        self.assertIn("seconds", report)
        self.assertIn("waiting", report)
        self.assertEqual(report["tool"], "extract")

    def test_only_the_last_few_finished_jobs_are_kept(self) -> None:
        queue = Queue(keep=3)
        ids = [queue.start("extract", answer, {}).id for _ in range(6)]
        queue.wait()
        kept = sum(1 for i in ids if i in queue._jobs)
        self.assertEqual(kept, 3)
        with self.assertRaises(JobError):
            queue.find(ids[0])


if __name__ == "__main__":
    unittest.main()
