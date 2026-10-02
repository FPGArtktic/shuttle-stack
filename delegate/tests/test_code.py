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


# A module that instantiates the one in CUT, so that the second hop
# has somewhere to go, and a bbappend whose variable has no definition
# of its own, which is the case the expansion exists for.
TOP = Cut(
    file="counter_top.sv",
    language="systemverilog",
    how="tree-sitter",
    why="",
    units=[
        Unit(
            "module_declaration header",
            "counter_top",
            1,
            5,
            "module counter_top (\n  input clk,\n  output q\n);",
        ),
        Unit(
            "declarations",
            "counter_top.lines 7-9",
            7,
            9,
            "counter u0 (.clk(clk));\n\nclk_tick u1 ();",
        ),
    ],
    edges=[
        Edge("include", "defs.svh", 2),
        Edge("instantiate", "counter", 7),
        Edge("instantiate", "clk_tick", 9),
    ],
)

APPEND = Cut(
    file="thing_1.0.bbappend",
    language="bitbake",
    how="tree-sitter",
    why="",
    units=[
        Unit("declarations", "lines 1-6", 1, 6, "RDEPENDS:${PN} += ..."),
    ],
    edges=[
        Edge("use", "systemd", 1),
        Edge("assign", "RDEPENDS:${PN}", 3),
        Edge("include", "thing-common.inc", 5),
    ],
)


OTHER = Cut(
    file="thing_%.bbappend",
    language="bitbake",
    how="tree-sitter",
    why="",
    units=[
        Unit("declarations", "lines 1-2", 1, 2, "inherit cmake"),
        Unit("declarations", "lines 40-42", 40, 42, "RDEPENDS:${PN} += x"),
    ],
    edges=[
        Edge("inherit", "cmake", 1),
        Edge("assign", "RDEPENDS:${PN}", 41),
        Edge("assign", "FILES:${PN}", 42),
    ],
)


class SameNameTest(unittest.TestCase):
    def test_a_variable_and_its_override_are_one_name(self) -> None:
        self.assertTrue(code.same_name("RDEPENDS", "RDEPENDS:${PN}"))

    def test_a_longer_name_is_a_different_name(self) -> None:
        self.assertFalse(code.same_name("counter", "counterweight"))

    def test_a_name_is_itself(self) -> None:
        self.assertTrue(code.same_name("counter", "counter"))


class ExpandTest(IndexTest):
    """Two hops: M5's expansion, over the same index as the lookup."""

    def test_a_definition_comes_back_with_its_head(self) -> None:
        self.feed()
        got = code.expand(self.db, "counter")
        self.assertEqual(
            [one["at"] for one in got["defined"]], ["counter.sv:1"]
        )
        self.assertEqual(got["defined"][0]["head"], "module counter")
        self.assertEqual(got["defined"][0]["lines"], "1-4")

    def test_the_head_is_the_ports_for_hdl(self) -> None:
        """The chunker emits a module's head as its own unit, so the
        interface falls out of the definition without parsing it."""
        self.feed("counter_top.sv", TOP)
        got = code.expand(self.db, "counter_top", hops=1)
        self.assertIn("input clk", got["defined"][0]["head"])

    def test_the_exact_name_is_preferred_over_what_is_inside_it(
        self,
    ) -> None:
        self.feed()
        got = code.expand(self.db, "counter", hops=1)
        self.assertEqual(len(got["defined"]), 1)
        self.assertEqual(got["defined"][0]["name"], "counter")

    def test_the_second_hop_leaves_the_head_and_walks_the_body(self) -> None:
        """The whole point: a module's span is its family's extent, so
        what the body instantiates is one hop from the module."""
        self.feed("counter_top.sv", TOP)
        got = code.expand(self.db, "counter_top")
        reached = {one["name"]: one for one in got["reaches"]}
        self.assertEqual(set(reached), {"defs.svh", "counter", "clk_tick"})
        self.assertEqual(reached["counter"]["named_at"], "counter_top.sv:7")
        self.assertEqual(reached["counter"]["relation"], "instantiate")

    def test_a_reached_name_says_where_it_is_defined(self) -> None:
        self.feed()
        self.feed("counter_top.sv", TOP)
        got = code.expand(self.db, "counter_top")
        reached = {one["name"]: one for one in got["reaches"]}
        self.assertEqual(reached["counter"]["defined_at"], "counter.sv:1")

    def test_a_name_the_index_does_not_define_says_so_rather_than_guess(
        self,
    ) -> None:
        self.feed("counter_top.sv", TOP)
        got = code.expand(self.db, "counter_top")
        reached = {one["name"]: one for one in got["reaches"]}
        self.assertIsNone(reached["defs.svh"]["defined_at"])

    def test_who_instantiates_it_is_still_answered(self) -> None:
        self.feed()
        self.feed("counter_top.sv", TOP)
        got = code.expand(self.db, "counter")
        self.assertEqual(
            [one["at"] for one in got["named_from"]], ["counter_top.sv:7"]
        )

    def test_a_variable_with_no_definition_expands_from_the_layer(
        self,
    ) -> None:
        """The other M5 question. A BitBake variable is an assignment
        and not a unit, so the second hop starts at the recipe that
        assigns it: what the layer is, is what the layer inherits."""
        self.feed("thing_1.0.bbappend", APPEND)
        got = code.expand(self.db, "RDEPENDS")
        self.assertEqual(got["defined"], [])
        self.assertEqual(
            [one["at"] for one in got["named_from"]],
            ["thing_1.0.bbappend:3"],
        )
        reached = {one["name"]: one["relation"] for one in got["reaches"]}
        self.assertEqual(
            reached, {"systemd": "use", "thing-common.inc": "include"}
        )
        self.assertEqual(got["reaches"][0]["through"], "names RDEPENDS")

    def test_a_name_does_not_reach_its_own_override(self) -> None:
        self.feed("thing_1.0.bbappend", APPEND)
        got = code.expand(self.db, "RDEPENDS")
        self.assertNotIn(
            "RDEPENDS:${PN}", [one["name"] for one in got["reaches"]]
        )

    def test_one_hop_stops_at_the_name(self) -> None:
        self.feed("counter_top.sv", TOP)
        got = code.expand(self.db, "counter_top", hops=1)
        self.assertNotIn("reaches", got)
        self.assertEqual(got["hops"], 1)

    def test_more_than_two_hops_is_refused(self) -> None:
        with self.assertRaises(IndexingError) as caught:
            code.expand(self.db, "counter", hops=3)
        self.assertIn("one hop or two", str(caught.exception))

    def test_an_empty_name_is_refused(self) -> None:
        with self.assertRaises(IndexingError):
            code.expand(self.db, "  ")

    def test_too_many_reached_names_say_so(self) -> None:
        self.feed("counter_top.sv", TOP)
        got = code.expand(self.db, "counter_top", limit=1)
        self.assertEqual(len(got["reaches"]), 1)
        self.assertEqual(got["more_than"], 1)

    def test_a_wildcard_in_the_name_is_text_not_a_pattern(self) -> None:
        self.feed()
        got = code.expand(self.db, "count%")
        self.assertEqual(got["defined"], [])
        self.assertIn("nothing in the index", got["note"])

    def test_a_name_nobody_knows_says_what_to_call(self) -> None:
        self.feed()
        got = code.expand(self.db, "nowhere")
        self.assertEqual(got["reaches"], [])
        self.assertIn("index_path", got["note"])

    def test_the_head_is_cut_rather_than_the_whole_definition(self) -> None:
        long = Cut(
            file="big.c",
            language="c",
            how="tree-sitter",
            why="",
            units=[
                Unit(
                    "function_definition",
                    "main",
                    1,
                    60,
                    "\n".join(f"line {i}" for i in range(1, 61)),
                )
            ],
            edges=[],
        )
        self.feed("big.c", long)
        got = code.expand(self.db, "main", hops=1)
        head = got["defined"][0]["head"].splitlines()
        self.assertEqual(len(head), code.HEAD_LINES)


class ReachOrderTest(IndexTest):
    """What makes a capped second hop, and what does not."""

    def test_what_a_file_pulls_in_comes_before_its_header(self) -> None:
        self.feed("thing_1.0.bbappend", APPEND)
        got = code.expand(self.db, "RDEPENDS", limit=1)
        self.assertEqual(got["reaches"][0]["name"], "thing-common.inc")
        self.assertEqual(got["reaches"][0]["relation"], "include")

    def test_every_layer_answers_before_any_answers_twice(self) -> None:
        """Measured: one ranked list spent the whole budget on the
        file with the most edges and said nothing about the rest."""
        self.feed("thing_1.0.bbappend", APPEND)
        self.feed("thing_%.bbappend", OTHER)
        got = code.expand(self.db, "RDEPENDS", limit=2)
        self.assertEqual(
            {one["named_at"].split(":")[0] for one in got["reaches"]},
            {"thing_1.0.bbappend", "thing_%.bbappend"},
        )

    def test_an_inherit_outside_the_unit_is_still_found(self) -> None:
        """A bbappend inherits at the top and overrides at the bottom,
        so the unit around the override holds neither."""
        self.feed("thing_%.bbappend", OTHER)
        got = code.expand(self.db, "RDEPENDS")
        reached = {one["name"]: one for one in got["reaches"]}
        self.assertEqual(reached["cmake"]["relation"], "inherit")
        self.assertEqual(
            reached["cmake"]["through"], "what thing_%.bbappend pulls in"
        )

    def test_a_variable_in_the_file_is_not_read_as_structure(self) -> None:
        self.feed("thing_%.bbappend", OTHER)
        got = code.expand(self.db, "RDEPENDS")
        through = {one["name"]: one["through"] for one in got["reaches"]}
        self.assertEqual(through["FILES:${PN}"], "names RDEPENDS")


class WithinTest(IndexTest):
    """A code search narrowed to one file.

    The same shape as the document half, and found there first: a
    question about one file had to be asked of the whole index, and
    the hit could come from another. repo-scout reaches for this
    shelf, so the gap was the same gap.
    """

    def two(self) -> None:
        self.feed()
        self.feed(
            "other.sv",
            Cut(
                "other.sv",
                "systemverilog",
                "tree-sitter",
                "",
                [
                    Unit("module_declaration", "other", 1, 4, "reset here"),
                    Unit("always_construct", "other.lines 6-8", 6, 8, "r"),
                ],
                [],
            ),
        )

    def ask_in(self, question: str, file: str) -> dict[str, Any]:
        with mock.patch.object(
            indexing, "embed", side_effect=self.embedding_for
        ):
            return code.search(question, 3, db=self.db, file=file)

    def test_a_search_within_one_file_stays_there(self) -> None:
        self.two()
        got = self.ask_in("reset", "other.sv")
        self.assertEqual({one["file"] for one in got["hits"]}, {"other.sv"})
        self.assertEqual(got["within"], "other.sv")

    def test_without_a_file_the_whole_shelf_is_searched(self) -> None:
        self.two()
        got = self.ask("reset")
        self.assertNotIn("within", got)
        self.assertNotIn("units", got)

    def test_a_narrowed_search_says_what_the_file_holds(self) -> None:
        self.two()
        got = self.ask_in("reset", "counter.sv")
        self.assertEqual(
            got["units"],
            ["counter:1", "counter.lines 6-8:6", "do_reset:10"],
        )

    def test_a_path_is_taken_by_its_name(self) -> None:
        self.two()
        got = self.ask_in("reset", "/elsewhere/other.sv")
        self.assertEqual({one["file"] for one in got["hits"]}, {"other.sv"})

    def test_a_file_nobody_indexed_is_refused(self) -> None:
        self.two()
        with self.assertRaises(IndexingError) as caught:
            self.ask_in("reset", "absent.sv")
        self.assertIn("not an indexed source file", str(caught.exception))

    def test_two_files_do_not_share_a_narrowed_answer(self) -> None:
        self.two()
        one = self.ask_in("reset", "counter.sv")
        two = self.ask_in("reset", "other.sv")
        self.assertFalse(two["cached"])
        self.assertEqual({x["file"] for x in one["hits"]}, {"counter.sv"})
        self.assertEqual({x["file"] for x in two["hits"]}, {"other.sv"})

    def test_a_narrowed_answer_is_kept_like_any_other(self) -> None:
        self.two()
        self.ask_in("reset", "other.sv")
        again = self.ask_in("reset", "other.sv")
        self.assertTrue(again["cached"])

    def test_the_broad_answer_is_not_served_to_a_narrowed_ask(self) -> None:
        self.two()
        self.ask("reset")
        narrow = self.ask_in("reset", "other.sv")
        self.assertFalse(narrow["cached"])
        self.assertEqual({x["file"] for x in narrow["hits"]}, {"other.sv"})


if __name__ == "__main__":
    unittest.main()
