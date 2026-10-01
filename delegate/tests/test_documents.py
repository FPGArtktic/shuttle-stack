# SPDX-License-Identifier: GPL-3.0-only
"""Reading a PDF, without needing the image to be built."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from shuttle_delegate import documents
from shuttle_delegate.documents import DocumentError, Page

ANSWER = {
    "file": "doc.pdf",
    "pages_in_file": 3,
    "pages": [
        {"page": 1, "ocr": False, "text": "the first page"},
        {"page": 2, "ocr": True, "text": "the second, looked at"},
        {"page": 3, "ocr": False, "text": "   "},
    ],
}


def done(
    stdout: str = "", stderr: str = "", code: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["podman"], code, stdout, stderr)


class TidyTest(unittest.TestCase):
    def test_a_dot_leader_becomes_a_gap(self) -> None:
        self.assertEqual(
            documents.tidy("Supply Voltage . . . . . . 20V"),
            "Supply Voltage  20V",
        )

    def test_a_leader_ending_on_a_dot_is_taken_whole(self) -> None:
        self.assertEqual(
            documents.tidy("Temperature . . . . 125oC"),
            "Temperature  125oC",
        )

    def test_a_version_number_is_left_alone(self) -> None:
        self.assertEqual(documents.tidy("Rev 3.3.1 V"), "Rev 3.3.1 V")

    def test_an_ellipsis_is_left_alone(self) -> None:
        self.assertEqual(
            documents.tidy("the rest ... and so on"),
            "the rest ... and so on",
        )

    def test_trailing_space_goes_with_it(self) -> None:
        self.assertEqual(documents.tidy("a value 20V   "), "a value 20V")


class SuffixTest(unittest.TestCase):
    def test_a_pdf_is_a_document(self) -> None:
        self.assertTrue(documents.is_document("a/b/Datasheet.PDF"))

    def test_a_script_is_not(self) -> None:
        self.assertFalse(documents.is_document("install.sh"))


class PagesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.file = Path(self.dir.name) / "doc.pdf"
        self.file.write_bytes(b"%PDF-1.4\n")

    def run_with(self, result: subprocess.CompletedProcess[str]) -> Any:
        patch = mock.patch.object(subprocess, "run", return_value=result)
        self.addCleanup(patch.stop)
        return patch.start()

    def test_the_pages_come_back_parsed(self) -> None:
        self.run_with(done(json.dumps(ANSWER)))
        pages = documents.pages(str(self.file))
        self.assertEqual(len(pages), 3)
        self.assertEqual(pages[1], Page(2, True, "the second, looked at"))

    def test_the_container_sees_only_that_directory(self) -> None:
        runner = self.run_with(done(json.dumps(ANSWER)))
        documents.pages(str(self.file))
        argv = runner.call_args.args[0]
        self.assertIn("--network=none", argv)
        self.assertIn("--read-only", argv)
        self.assertIn(f"{self.file.parent}:{documents.MOUNT}:ro", argv)
        self.assertTrue(argv[-1].endswith("/doc.pdf"))

    def test_a_page_range_is_passed_through(self) -> None:
        runner = self.run_with(done(json.dumps(ANSWER)))
        documents.pages(str(self.file), 2, 5)
        self.assertEqual(runner.call_args.args[0][-2:], ["2", "5"])

    def test_a_missing_file_is_named(self) -> None:
        with self.assertRaises(DocumentError):
            documents.pages(str(self.file.parent / "absent.pdf"))

    def test_a_backwards_range_is_refused(self) -> None:
        with self.assertRaises(DocumentError):
            documents.pages(str(self.file), 5, 2)

    def test_a_page_before_the_first_is_refused(self) -> None:
        with self.assertRaises(DocumentError):
            documents.pages(str(self.file), 0)

    def test_the_extractor_s_own_complaint_is_carried_up(self) -> None:
        self.run_with(done("", "has 1 page(s); asked to start at 5", 2))
        with self.assertRaises(DocumentError) as caught:
            documents.pages(str(self.file), 5, 5)
        self.assertIn("asked to start at 5", str(caught.exception))

    def test_an_answer_that_is_not_json_is_an_error(self) -> None:
        self.run_with(done("this is not json at all"))
        with self.assertRaises(DocumentError) as caught:
            documents.pages(str(self.file))
        self.assertIn("did not answer with JSON", str(caught.exception))

    def test_a_timeout_suggests_a_page_range(self) -> None:
        patch = mock.patch.object(
            subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(["podman"], 1),
        )
        patch.start()
        self.addCleanup(patch.stop)
        with self.assertRaises(DocumentError) as caught:
            documents.pages(str(self.file))
        self.assertIn("page range", str(caught.exception))


class AsTextTest(unittest.TestCase):
    PAGES = [Page(1, False, "the first page"), Page(2, True, "the second")]

    def test_every_page_is_marked_with_its_number(self) -> None:
        text = documents.as_text(self.PAGES)
        self.assertIn("--- page 1", text)
        self.assertIn("--- page 2", text)

    def test_a_blank_page_is_left_out(self) -> None:
        text = documents.as_text([*self.PAGES, Page(3, False, "  \n ")])
        self.assertNotIn("--- page 3", text)

    def test_the_page_order_is_kept(self) -> None:
        text = documents.as_text(self.PAGES)
        self.assertLess(text.index("first page"), text.index("the second"))


if __name__ == "__main__":
    unittest.main()
