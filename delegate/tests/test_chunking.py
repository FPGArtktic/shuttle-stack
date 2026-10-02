# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mateusz Okulanis
"""Splitting, at every boundary it has to fall back through."""

from __future__ import annotations

import unittest

from shuttle_delegate.chunking import chunk

PARAGRAPHS = "one\n\ntwo\n\nthree"


class ChunkTest(unittest.TestCase):
    def test_a_short_text_is_one_chunk(self) -> None:
        self.assertEqual(chunk(PARAGRAPHS, 100), [PARAGRAPHS])

    def test_nothing_in_yields_nothing_out(self) -> None:
        self.assertEqual(chunk("", 100), [])
        self.assertEqual(chunk("   \n\n  \n", 100), [])

    def test_no_chunk_exceeds_the_limit(self) -> None:
        text = PARAGRAPHS * 40
        for limit in (8, 17, 64, 256):
            with self.subTest(limit=limit):
                for piece in chunk(text, limit):
                    self.assertLessEqual(len(piece), limit)

    def test_paragraphs_are_preferred_to_lines(self) -> None:
        self.assertEqual(chunk(PARAGRAPHS, 9), ["one\n\ntwo", "three"])

    def test_a_long_paragraph_falls_back_to_lines(self) -> None:
        text = "alpha\nbeta\ngamma"
        self.assertEqual(chunk(text, 11), ["alpha\n\nbeta", "gamma"])

    def test_a_single_long_line_is_cut_mid_line(self) -> None:
        chunks = chunk("x" * 25, 10)
        self.assertEqual(chunks, ["x" * 10, "x" * 10, "x" * 5])

    def test_every_character_of_a_word_survives(self) -> None:
        text = "".join(f"word{i} " for i in range(500))
        joined = "".join(chunk(text, 37))
        self.assertEqual(joined.replace("\n", "").count("word"), 500)

    def test_a_limit_below_one_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            chunk("anything", 0)


if __name__ == "__main__":
    unittest.main()
