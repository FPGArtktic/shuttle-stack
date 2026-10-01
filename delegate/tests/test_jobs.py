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
        self.addCleanup(self.queue.close)

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
        self.assertEqual(self.queue.status(job.id)["state"], FAILED)
        with self.assertRaises(JobError) as caught:
            self.queue.collect(job.id)
        self.assertIn("could not be done", str(caught.exception))

    def test_an_unknown_job_is_refused(self) -> None:
        with self.assertRaises(JobError):
            self.queue.status("no-such-job")

    def test_a_status_reports_the_time_it_took(self) -> None:
        job = self.queue.start("extract", answer, {})
        self.queue.wait()
        report = self.queue.status(job.id)
        self.assertIn("seconds", report)
        self.assertIn("waiting", report)
        self.assertEqual(report["tool"], "extract")

    def test_only_the_last_few_finished_jobs_are_kept(self) -> None:
        queue = Queue(keep=3)
        self.addCleanup(queue.close)
        ids = [queue.start("extract", answer, {}).id for _ in range(6)]
        queue.wait()
        for forgotten in ids[:-3]:
            with self.assertRaises(JobError):
                queue.status(forgotten)
        for kept in ids[-3:]:
            self.assertEqual(queue.status(kept)["state"], DONE)


class SnapshotTest(unittest.TestCase):
    """A status is taken under the lock, not read off a live job."""

    def test_a_status_is_a_plain_dictionary(self) -> None:
        queue = Queue()
        self.addCleanup(queue.close)
        job = queue.start("extract", answer, {})
        queue.wait()
        snapshot = queue.status(job.id)
        self.assertIsInstance(snapshot, dict)
        self.assertEqual(snapshot["state"], DONE)

    def test_a_failed_job_always_carries_its_reason(self) -> None:
        queue = Queue()
        self.addCleanup(queue.close)
        job = queue.start("extract", broken, {})
        queue.wait()
        snapshot = queue.status(job.id)
        self.assertEqual(snapshot["state"], FAILED)
        self.assertIn("could not be done", snapshot["error"])

    def test_closing_twice_is_not_an_error(self) -> None:
        queue = Queue()
        queue.close()
        queue.close()


if __name__ == "__main__":
    unittest.main()
