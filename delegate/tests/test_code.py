# SPDX-License-Identifier: GPL-3.0-only
"""Source units: what is stored, what is cited, what is found.

The chunker itself runs in the container and is not started here; what
it returns is handed in directly. The index is a real file in a
temporary directory, as it is for the documents.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from shuttle_delegate import code, indexing
from shuttle_delegate.code import Cut, Edge, Unit, is_source
from shuttle_delegate.documents import DocumentError
from shuttle_delegate.indexing import DIMENSIONS, IndexingError

ANSWER = {
    "file": "counter.sv",
    "language": "systemverilog",
    "how": "tree-sitter",
    "error_share": 0.0,
    "units": [
        {
            "kind": "module_declaration header",
            "name": "counter",
            "first_line": 1,
            "last_line": 4,
            "text": "module counter (input clk, output q);",
        },
        {
            "kind": "always_construct",
            "name": "counter.lines 6-8",
            "first_line": 6,
            "last_line": 8,
            "text": "always @(posedge clk) r <= r + 1;",
        },
    ],
}

CUT = Cut(
    file="counter.sv",
    language="systemverilog",
    how="tree-sitter",
    why="",
    units=[
        Unit("module_declaration header", "counter", 1, 4, "module counter"),
        Unit("always_construct", "counter.lines 6-8", 6, 8, "r <= r + 1"),
        Unit("task_declaration", "do_reset", 10, 12, "r <= 0 reset"),
    ],
    edges=[
        Edge("instantiate", "debouncer", 3),
        Edge("instantiate", "clk_tick", 4),
        Edge("include", "defs.svh", 1),
    ],
)


def done(
    stdout: str = "", stderr: str = "", code_: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["podman"], code_, stdout, stderr)


class SuffixTest(unittest.TestCase):
    def test_the_languages_in_the_design_are_recognised(self) -> None:
        for name in (
            "main.rs",
            "server.go",
            "top.v",
            "top.sv",
            "cpu.vhd",
            "driver.c",
            "driver.h",
            "thing_git.bb",
            "thing_%.bbappend",
            "pins.xdc",
            "board.dts",
            "Kconfig.debug",
            "Makefile",
            "rules.mk",
            "build.sh",
        ):
            with self.subTest(name=name):
                self.assertTrue(is_source(name))

    def test_a_pdf_is_not_source(self) -> None:
        self.assertFalse(is_source("datasheet.pdf"))

    def test_prose_is_not_source(self) -> None:
        self.assertFalse(is_source("README.md"))


class UnitTest(unittest.TestCase):
    def test_a_range_of_one_line_is_named_as_one(self) -> None:
        self.assertEqual(Unit("k", "n", 7, 7, "t").lines, "line 7")

    def test_a_range_names_both_ends(self) -> None:
        self.assertEqual(Unit("k", "n", 7, 9, "t").lines, "lines 7-9")

    def test_an_unnamed_unit_is_labelled_by_its_kind(self) -> None:
        self.assertEqual(Unit("rule", "", 1, 2, "t").label, "rule")


class ChunkerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.file = Path(self.dir.name) / "counter.sv"
        self.file.write_text("module counter;\nendmodule\n")

    def ran(self, result: subprocess.CompletedProcess[str]) -> Any:
        patch = mock.patch.object(subprocess, "run", return_value=result)
        self.addCleanup(patch.stop)
        return patch.start()

    def test_the_container_is_asked_to_chunk_and_nothing_else(self) -> None:
        run = self.ran(done(json.dumps(ANSWER)))
        code.units(str(self.file))
        argv = run.call_args[0][0]
        self.assertIn("chunk", argv)
        self.assertIn("--network=none", argv)
        self.assertIn("--read-only", argv)

    def test_the_edges_come_back_with_their_lines(self) -> None:
        self.ran(
            done(
                json.dumps(
                    ANSWER
                    | {
                        "edges": [
                            {
                                "kind": "instantiate",
                                "name": "debouncer",
                                "line": 7,
                            }
                        ]
                    }
                )
            )
        )
        cut = code.units(str(self.file))
        self.assertEqual(cut.edges, [Edge("instantiate", "debouncer", 7)])

    def test_a_chunker_that_reports_no_edges_is_not_an_error(self) -> None:
        self.ran(done(json.dumps(ANSWER)))
        self.assertEqual(code.units(str(self.file)).edges, [])

    def test_the_units_come_back_with_their_lines(self) -> None:
        self.ran(done(json.dumps(ANSWER)))
        cut = code.units(str(self.file))
        self.assertEqual(cut.language, "systemverilog")
        self.assertEqual([u.first_line for u in cut.units], [1, 6])
        self.assertEqual(cut.units[1].lines, "lines 6-8")

    def test_a_fallback_says_why(self) -> None:
        self.ran(
            done(
                json.dumps(
                    ANSWER | {"how": "lines", "why": "the grammar gave up"}
                )
            )
        )
        cut = code.units(str(self.file))
        self.assertEqual(cut.how, "lines")
        self.assertEqual(cut.why, "the grammar gave up")

    def test_a_chunker_that_fails_says_what_it_said(self) -> None:
        self.ran(done(stderr="no grammar for that", code_=2))
        with self.assertRaises(DocumentError) as caught:
            code.units(str(self.file))
        self.assertIn("no grammar for that", str(caught.exception))

    def test_an_answer_that_is_not_json_is_an_error(self) -> None:
        self.ran(done(stdout="Traceback (most recent call last):"))
        with self.assertRaises(DocumentError):
            code.units(str(self.file))

    def test_a_file_that_is_not_there_is_refused(self) -> None:
        with self.assertRaises(DocumentError):
            code.units(str(self.file.parent / "absent.sv"))


class IndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "documents.sqlite"
        patch = mock.patch.object(
            indexing, "index_file", return_value=self.path
        )
        patch.start()
        self.addCleanup(patch.stop)
        self.db = code.connect()
        self.addCleanup(self.db.close)

    def embedding_for(self, texts: list[str]) -> list[list[float]]:
        words = ("reset", "counter", "clock")
        out = []
        for text in texts:
            low = text.lower()
            made = [0.0] * DIMENSIONS
            for at, word in enumerate(words):
                made[at] = 1.0 if word in low else 0.0
            if not any(made):
                made[len(words)] = 1.0
            out.append(made)
        return out

    def feed(self, name: str = "counter.sv", cut: Cut = CUT) -> int:
        with mock.patch.object(
            indexing, "embed", side_effect=self.embedding_for
        ):
            return code.store(self.db, name, cut, 12)

    def ask(self, question: str, limit: int = 3) -> dict[str, Any]:
        with mock.patch.object(
            indexing, "embed", side_effect=self.embedding_for
        ):
            return code.search(question, limit, db=self.db)

    def test_the_document_tables_are_left_alone(self) -> None:
        names = {
            row[0]
            for row in self.db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        self.assertLessEqual(
            {"documents", "chunks", "sources", "units"}, names
        )

    def test_a_unit_is_found_by_its_name(self) -> None:
        self.feed()
        hit = self.ask("do_reset", limit=1)["hits"][0]
        self.assertIn("do_reset", hit["section"])
        self.assertEqual(hit["clause"], "lines 10-12")

    def test_the_citation_is_a_line_range_not_a_page(self) -> None:
        self.feed()
        for hit in self.ask("counter")["hits"]:
            self.assertRegex(hit["clause"], r"^lines? \d+")

    def test_the_name_is_embedded_with_the_body(self) -> None:
        """A task's body does not contain its own name."""
        seen: list[str] = []

        def watch(texts: list[str]) -> list[list[float]]:
            seen.extend(texts)
            return self.embedding_for(texts)

        with mock.patch.object(indexing, "embed", side_effect=watch):
            code.store(self.db, "counter.sv", CUT, 12)
        self.assertTrue(any("do_reset" in text for text in seen))

    def test_indexing_again_replaces_rather_than_doubles(self) -> None:
        self.feed()
        self.feed()
        held = self.db.execute("SELECT COUNT(*) FROM units").fetchone()[0]
        self.assertEqual(held, len(CUT.units))

    def test_the_vectors_go_with_the_units_they_replace(self) -> None:
        self.feed()
        self.feed(cut=Cut("counter.sv", "sv", "lines", "", CUT.units[:1]))
        units = self.db.execute("SELECT COUNT(*) FROM units").fetchone()[0]
        vectors = self.db.execute("SELECT COUNT(*) FROM vec_units").fetchone()[
            0
        ]
        self.assertEqual((units, vectors), (1, 1))

    def test_the_lexical_index_follows_a_deletion(self) -> None:
        self.feed()
        self.feed(cut=Cut("counter.sv", "sv", "lines", "", CUT.units[:1]))
        found = self.db.execute(
            "SELECT COUNT(*) FROM units_fts WHERE units_fts MATCH ?",
            ('"do_reset"',),
        ).fetchone()[0]
        self.assertEqual(found, 0)

    def test_the_listing_says_how_each_file_was_cut(self) -> None:
        self.feed()
        self.feed(
            "fallback.tcl",
            Cut("f.tcl", "tcl", "lines", "gave up", CUT.units[:1]),
        )
        listed = code.indexed(self.db)["sources"]
        self.assertEqual(
            {row["file"]: row["cut_by"] for row in listed},
            {"counter.sv": "tree-sitter", "fallback.tcl": "lines"},
        )

    def test_the_edges_are_stored_with_the_file(self) -> None:
        self.feed()
        held = {
            (row["kind"], row["name"], row["line"])
            for row in self.db.execute(
                "SELECT kind, name, line FROM edges WHERE source = ?",
                ("counter.sv",),
            )
        }
        self.assertEqual(
            held,
            {
                ("instantiate", "debouncer", 3),
                ("instantiate", "clk_tick", 4),
                ("include", "defs.svh", 1),
            },
        )

    def test_reindexing_replaces_the_edges(self) -> None:
        self.feed()
        thinner = Cut(
            "counter.sv",
            "systemverilog",
            "tree-sitter",
            "",
            CUT.units[:1],
            [Edge("instantiate", "debouncer", 3)],
        )
        self.feed(cut=thinner)
        held = self.db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        self.assertEqual(held, 1)

    def test_who_instantiates_a_module(self) -> None:
        """One of the two questions M5 asks of a scout."""
        self.feed()
        got = code.references(self.db, "debouncer", "instantiates")
        self.assertEqual(
            [one["at"] for one in got["references"]], ["counter.sv:3"]
        )

    def test_a_relation_narrows_what_comes_back(self) -> None:
        self.feed()
        got = code.references(self.db, "defs.svh", "instantiates")
        self.assertEqual(got["references"], [])
        self.assertIn("nothing in the index", got["note"])

    def test_an_override_is_found_by_the_bare_variable(self) -> None:
        """The other question: which file overrides variable Y."""
        self.db.execute(
            "INSERT INTO edges (source, kind, name, line)"
            " VALUES ('a.bb', 'assign', 'RDEPENDS:${PN}', 16)"
        )
        self.db.commit()
        got = code.references(self.db, "RDEPENDS")
        self.assertEqual(got["references"][0]["names"], "RDEPENDS:${PN}")

    def test_a_scoped_path_is_found_by_its_head(self) -> None:
        self.db.executemany(
            "INSERT INTO edges (source, kind, name, line) VALUES (?,?,?,?)",
            [
                ("a.rs", "use", "std::io::Write", 1),
                ("a.go", "call", "fmt.Println", 18),
            ],
        )
        self.db.commit()
        self.assertEqual(
            code.references(self.db, "std")["references"][0]["names"],
            "std::io::Write",
        )
        self.assertEqual(
            code.references(self.db, "fmt")["references"][0]["names"],
            "fmt.Println",
        )

    def test_a_prefix_without_a_separator_is_not_a_match(self) -> None:
        """counter must not answer about counterweight."""
        self.db.execute(
            "INSERT INTO edges (source, kind, name, line)"
            " VALUES ('x.sv', 'instantiate', 'counterweight', 7)"
        )
        self.db.commit()
        self.assertEqual(code.references(self.db, "counter")["references"], [])

    def test_a_wildcard_in_the_name_is_text_not_a_pattern(self) -> None:
        self.db.execute(
            "INSERT INTO edges (source, kind, name, line)"
            " VALUES ('x.sv', 'instantiate', 'anything', 7)"
        )
        self.db.commit()
        self.assertEqual(code.references(self.db, "%")["references"], [])

    def test_too_many_references_say_so(self) -> None:
        self.db.executemany(
            "INSERT INTO edges (source, kind, name, line) VALUES (?,?,?,?)",
            [(f"f{n}.sv", "instantiate", "counter", n) for n in range(1, 8)],
        )
        self.db.commit()
        got = code.references(self.db, "counter", limit=3)
        self.assertEqual(len(got["references"]), 3)
        self.assertEqual(got["more_than"], 3)

    def test_an_empty_name_is_refused(self) -> None:
        with self.assertRaises(IndexingError):
            code.references(self.db, "  ")

    def test_an_unknown_relation_lists_the_ones_there_are(self) -> None:
        with self.assertRaises(IndexingError) as caught:
            code.references(self.db, "counter", "wires-up")
        self.assertIn("instantiates", str(caught.exception))

    def test_storing_nothing_is_an_error(self) -> None:
        empty = Cut("a.sv", "systemverilog", "tree-sitter", "", [])
        with self.assertRaises(IndexingError):
            code.store(self.db, "a.sv", empty, 0)

    def test_a_search_with_no_source_says_what_to_call(self) -> None:
        with self.assertRaises(IndexingError) as caught:
            self.ask("anything")
        self.assertIn("index_path", str(caught.exception))

    def test_the_same_question_twice_is_answered_from_the_file(self) -> None:
        self.feed()
        self.assertFalse(self.ask("do_reset")["cached"])
        with mock.patch.object(
            indexing, "embed", side_effect=AssertionError("asked again")
        ):
            self.assertTrue(code.search("do_reset", 3, db=self.db)["cached"])

    def test_the_two_shelves_do_not_share_a_cached_answer(self) -> None:
        """The same words asked of source and of documents differ."""
        self.feed()
        with mock.patch.object(
            indexing, "embed", side_effect=self.embedding_for
        ):
            indexing.store(
                self.db,
                "a.pdf",
                [indexing.Section(3, "1 Reset", "the reset is on page 3")],
            )
            from_code = code.search("reset", 3, db=self.db)
            from_docs = indexing.search("reset", 3, db=self.db)
        self.assertEqual(from_code["hits"][0]["file"], "counter.sv")
        self.assertEqual(from_docs["hits"][0]["file"], "a.pdf")

    def test_indexing_source_drops_the_kept_document_answers(self) -> None:
        with mock.patch.object(
            indexing, "embed", side_effect=self.embedding_for
        ):
            indexing.store(
                self.db,
                "a.pdf",
                [indexing.Section(3, "1 Reset", "the reset is on page 3")],
            )
            indexing.search("reset", 3, db=self.db)
            self.feed()
            again = indexing.search("reset", 3, db=self.db)
        self.assertFalse(again["cached"])


class LocationTest(IndexTest):
    """Where a file was read from, which a citation does not need.

    A citation is a name and a line, so the index got away with
    storing only the name for a long time. An agent that wants to
    read the lines around a reference cannot: it knows
    `counter_top.sv` is indexed and not where it is.
    """

    def test_a_file_indexed_without_a_path_does_not_resolve(self) -> None:
        self.feed()
        self.assertIsNone(code.located(self.db, "counter.sv"))

    def test_the_path_comes_back_when_the_file_is_still_there(self) -> None:
        here = Path(self.dir.name) / "counter.sv"
        here.write_text("module counter;\nendmodule\n")
        with mock.patch.object(
            indexing, "embed", side_effect=self.embedding_for
        ):
            code.store(self.db, "counter.sv", CUT, 12, str(here))
        self.assertEqual(code.located(self.db, "counter.sv"), here)

    def test_a_path_that_has_gone_away_does_not_resolve(self) -> None:
        gone = Path(self.dir.name) / "gone.sv"
        with mock.patch.object(
            indexing, "embed", side_effect=self.embedding_for
        ):
            code.store(self.db, "counter.sv", CUT, 12, str(gone))
        self.assertIsNone(code.located(self.db, "counter.sv"))


if __name__ == "__main__":
    unittest.main()
