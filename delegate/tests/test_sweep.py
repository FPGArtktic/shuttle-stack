# SPDX-License-Identifier: GPL-3.0-only
"""The overnight pass: what it looks at, remembers and reports."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from shuttle_delegate import code, indexing, sweep
from shuttle_delegate.code import Cut, Unit
from shuttle_delegate.sweep import Done, Report, Settings, SweepError


class SettingsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name) / "docs"
        self.root.mkdir()

    def write(self, text: str) -> Path:
        path = Path(self.dir.name) / "sweep.toml"
        path.write_text(text)
        return path

    def test_the_roots_are_read_and_expanded(self) -> None:
        chosen = sweep.settings(self.write(f'roots = ["{self.root}"]\n'))
        self.assertEqual(chosen.roots, (self.root,))
        self.assertEqual(chosen.heading, indexing.HEADING)

    def test_a_missing_file_says_what_it_would_contain(self) -> None:
        with self.assertRaises(SweepError) as caught:
            sweep.settings(Path(self.dir.name) / "absent.toml")
        self.assertIn("roots =", str(caught.exception))

    def test_no_roots_is_nothing_to_do(self) -> None:
        with self.assertRaises(SweepError):
            sweep.settings(self.write("roots = []\n"))

    def test_a_root_that_is_not_a_directory_is_refused(self) -> None:
        with self.assertRaises(SweepError) as caught:
            sweep.settings(self.write('roots = ["/no/such/place"]\n'))
        self.assertIn("not a directory", str(caught.exception))

    def test_broken_toml_names_the_file(self) -> None:
        file = self.write("roots = [\n")
        with self.assertRaises(SweepError) as caught:
            sweep.settings(file)
        self.assertIn(str(file), str(caught.exception))


class DocumentsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        (self.root / "deep").mkdir()
        for name in (
            "a.pdf",
            "b.md",
            "deep/c.txt",
            "notes.log",
            "image.png",
        ):
            (self.root / name).write_text("x")

    def test_documents_are_found_below_the_root(self) -> None:
        found = [p.name for p in sweep.documents((self.root,))]
        self.assertEqual(found, ["a.pdf", "b.md", "c.txt"])

    def test_the_order_is_stable(self) -> None:
        self.assertEqual(
            sweep.documents((self.root,)), sweep.documents((self.root,))
        )


class RunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        patch = mock.patch.dict(os.environ, {"SHUTTLE_HOME": self.dir.name})
        patch.start()
        self.addCleanup(patch.stop)
        self.root = Path(self.dir.name) / "docs"
        self.root.mkdir()
        self.doc = self.root / "a.md"
        self.doc.write_text("## One\nalpha\n")
        self.chosen = Settings(roots=(self.root,))

    def indexing_that(self, sections: int = 3) -> tuple[Any, ...]:
        return (
            mock.patch.object(
                indexing, "split", return_value=([mock.Mock()], "headings")
            ),
            mock.patch.object(indexing, "connect"),
            mock.patch.object(indexing, "store", return_value=sections),
        )

    def test_source_is_swept_as_well_as_prose(self) -> None:
        (self.root / "top.sv").write_text("module top; endmodule\n")
        found = [p.name for p in sweep.documents((self.root,))]
        self.assertEqual(sorted(found), ["a.md", "top.sv"])

    def test_a_build_directory_is_not_swept(self) -> None:
        for where in ("db", "output_files", ".git"):
            made = self.root / where
            made.mkdir()
            (made / "generated.sv").write_text("module g; endmodule\n")
        found = [p.name for p in sweep.documents((self.root,))]
        self.assertEqual(found, ["a.md"])

    def test_source_goes_to_the_source_shelf(self) -> None:
        source = self.root / "top.sv"
        source.write_text("module top; endmodule\n")
        cut = Cut(
            "top.sv",
            "systemverilog",
            "tree-sitter",
            "",
            [Unit("module_declaration", "top", 1, 1, "module top")],
        )
        with (
            mock.patch.object(code, "units", return_value=cut),
            mock.patch.object(code, "connect"),
            mock.patch.object(code, "store", return_value=1) as stored,
        ):
            done = sweep.one(source, self.chosen)
        self.assertTrue(done.ok)
        self.assertEqual(done.how, "systemverilog by tree-sitter")
        stored.assert_called_once()
        self.assertEqual(stored.call_args.args[-1], str(source.resolve()))

    def test_a_new_document_is_indexed(self) -> None:
        with mock.patch.multiple(
            indexing,
            split=mock.DEFAULT,
            connect=mock.DEFAULT,
            store=mock.DEFAULT,
        ) as patched:
            patched["split"].return_value = ([mock.Mock()], "headings")
            patched["store"].return_value = 3
            report = sweep.run(self.chosen)
        self.assertEqual(len(report.indexed), 1)
        self.assertEqual(report.indexed[0].sections, 3)

    def test_an_unchanged_document_is_not_indexed_again(self) -> None:
        with mock.patch.multiple(
            indexing,
            split=mock.DEFAULT,
            connect=mock.DEFAULT,
            store=mock.DEFAULT,
        ) as patched:
            patched["split"].return_value = ([mock.Mock()], "headings")
            patched["store"].return_value = 3
            sweep.run(self.chosen)
            report = sweep.run(self.chosen)
        self.assertEqual(report.skipped, 1)
        self.assertEqual(report.indexed, [])

    def test_a_changed_document_is_read_again(self) -> None:
        with mock.patch.multiple(
            indexing,
            split=mock.DEFAULT,
            connect=mock.DEFAULT,
            store=mock.DEFAULT,
        ) as patched:
            patched["split"].return_value = ([mock.Mock()], "headings")
            patched["store"].return_value = 3
            sweep.run(self.chosen)
            self.doc.write_text("## One\nalpha\n\n## Two\nbeta\n")
            report = sweep.run(self.chosen)
        self.assertEqual(len(report.indexed), 1)

    def test_one_bad_document_does_not_end_the_night(self) -> None:
        (self.root / "b.md").write_text("## Two\nbeta\n")
        calls = {"n": 0}

        def sometimes(*_args: Any, **_kwargs: Any) -> int:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("that one is broken")
            return 2

        with mock.patch.multiple(
            indexing,
            split=mock.DEFAULT,
            connect=mock.DEFAULT,
            store=mock.DEFAULT,
        ) as patched:
            patched["split"].return_value = ([mock.Mock()], "headings")
            patched["store"].side_effect = sometimes
            report = sweep.run(self.chosen)
        self.assertEqual(len(report.failed), 1)
        self.assertEqual(len(report.indexed), 1)
        self.assertIn("that one is broken", report.failed[0].error)

    def test_a_failure_is_not_remembered_as_done(self) -> None:
        with mock.patch.multiple(
            indexing,
            split=mock.DEFAULT,
            connect=mock.DEFAULT,
            store=mock.DEFAULT,
        ) as patched:
            patched["split"].return_value = ([mock.Mock()], "headings")
            patched["store"].side_effect = RuntimeError("no")
            sweep.run(self.chosen)
        self.assertEqual(json.loads(sweep.state_path().read_text()), {})


class ReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        patch = mock.patch.dict(os.environ, {"SHUTTLE_HOME": self.dir.name})
        patch.start()
        self.addCleanup(patch.stop)

    def test_failures_come_before_successes(self) -> None:
        report = Report(
            started="now",
            indexed=[Done("a.pdf", sections=2, how="pages")],
            failed=[Done("b.pdf", error="unreadable")],
        )
        body = sweep.write_report(report).read_text()
        self.assertLess(body.index("## Failed"), body.index("## Indexed"))

    def test_the_counts_are_stated(self) -> None:
        report = Report(started="now", skipped=7)
        body = sweep.write_report(report).read_text()
        self.assertIn("unchanged: 7", body)

    def test_a_long_list_is_cut_and_says_so(self) -> None:
        report = Report(
            started="now",
            indexed=[Done(f"{n}.pdf") for n in range(sweep.MAX_LISTED + 5)],
        )
        body = sweep.write_report(report).read_text()
        self.assertIn("and 5 more", body)

    def test_the_report_is_held_to_the_token_limit(self) -> None:
        report = Report(
            started="now",
            failed=[Done(f"{n}.pdf", error="x" * 400) for n in range(20)],
        )
        body = sweep.write_report(report, lambda text: len(text)).read_text()
        self.assertLessEqual(len(body), sweep.REPORT_TOKENS + 80)


if __name__ == "__main__":
    unittest.main()
