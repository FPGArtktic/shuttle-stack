# SPDX-License-Identifier: GPL-3.0-only
"""Narrowing a file before a model ever sees it."""

from __future__ import annotations

import unittest

from shuttle_delegate.retrieval import PatternError, narrow, regions

TEXT = "\n".join(f"line {n}" for n in range(1, 41))


class RegionsTest(unittest.TestCase):
    def test_a_match_brings_its_neighbours(self) -> None:
        found = regions(TEXT, r"^line 20$", context=2)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].line, 18)
        self.assertEqual(found[0].text.splitlines()[0], "line 18")
        self.assertEqual(len(found[0].text.splitlines()), 5)

    def test_overlapping_matches_become_one_region(self) -> None:
        found = regions(TEXT, r"^line 2[01]$", context=3)
        self.assertEqual(len(found), 1)

    def test_distant_matches_stay_apart(self) -> None:
        found = regions(TEXT, r"^line (2|39)$", context=1)
        self.assertEqual(len(found), 2)

    def test_no_match_is_no_region_not_the_whole_file(self) -> None:
        self.assertEqual(regions(TEXT, r"^nothing here$"), [])

    def test_the_search_ignores_case(self) -> None:
        self.assertTrue(regions(TEXT, r"^LINE 7$"))

    def test_a_broken_pattern_is_named(self) -> None:
        with self.assertRaises(PatternError):
            regions(TEXT, "unclosed (")

    def test_a_negative_context_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            regions(TEXT, "line", context=-1)


class NarrowTest(unittest.TestCase):
    def test_the_result_carries_line_numbers(self) -> None:
        text, count = narrow(TEXT, r"^line 30$", context=1)
        self.assertEqual(count, 1)
        self.assertIn("--- line 29", text)
        self.assertIn("line 30", text)
        self.assertNotIn("line 20", text)

    def test_nothing_matching_yields_nothing(self) -> None:
        text, count = narrow(TEXT, r"^absent$")
        self.assertEqual((text, count), ("", 0))

    def test_narrowing_is_smaller_than_the_file(self) -> None:
        text, _ = narrow(TEXT, r"^line 5$", context=2)
        self.assertLess(len(text), len(TEXT))


if __name__ == "__main__":
    unittest.main()
