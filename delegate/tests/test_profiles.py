# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mateusz Okulanis
"""Named settings, and what a bad profiles file is told."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shuttle_delegate.profiles import BUILT_IN, ProfileError, get, load


class BuiltInTest(unittest.TestCase):
    def test_the_three_the_tools_need_are_there(self) -> None:
        self.assertEqual(set(load(Path("/absent"))), set(BUILT_IN))

    def test_extraction_is_not_sampled(self) -> None:
        self.assertEqual(get("extract", Path("/absent")).temperature, 0.0)

    def test_the_fast_profile_names_the_fast_server(self) -> None:
        self.assertEqual(get("fast", Path("/absent")).server, "fast")

    def test_an_unknown_name_lists_what_there_is(self) -> None:
        with self.assertRaises(ProfileError) as caught:
            get("medium", Path("/absent"))
        self.assertIn("extract", str(caught.exception))


class FileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def write(self, text: str) -> Path:
        path = Path(self.dir.name) / "profiles.toml"
        path.write_text(text)
        return path

    def test_a_file_overrides_one_key_and_keeps_the_rest(self) -> None:
        profile = get("long", self.write("[long]\ntemperature = 0.9\n"))
        self.assertEqual(profile.temperature, 0.9)
        self.assertEqual(profile.server, "long")

    def test_a_file_may_add_a_profile(self) -> None:
        file = self.write('[strict]\nserver = "fast"\ntemperature = 0.0\n')
        self.assertEqual(get("strict", file).server, "fast")
        self.assertIn("long", load(file))

    def test_an_unknown_key_is_refused_not_ignored(self) -> None:
        with self.assertRaises(ProfileError) as caught:
            load(self.write("[long]\nmodel = 'something'\n"))
        self.assertIn("model", str(caught.exception))

    def test_an_unknown_server_is_refused(self) -> None:
        with self.assertRaises(ProfileError):
            load(self.write('[long]\nserver = "medium"\n'))

    def test_a_temperature_outside_the_range_is_refused(self) -> None:
        with self.assertRaises(ProfileError):
            load(self.write("[long]\ntemperature = 5.0\n"))

    def test_a_report_of_no_tokens_is_refused(self) -> None:
        with self.assertRaises(ProfileError):
            load(self.write("[long]\nreport_tokens = 0\n"))

    def test_a_profile_that_is_not_a_table_says_how_to_write_one(
        self,
    ) -> None:
        with self.assertRaises(ProfileError) as caught:
            load(self.write('long = "fast"\n'))
        self.assertIn("[long]", str(caught.exception))

    def test_broken_toml_names_the_file(self) -> None:
        file = self.write("[long\n")
        with self.assertRaises(ProfileError) as caught:
            load(file)
        self.assertIn(str(file), str(caught.exception))


if __name__ == "__main__":
    unittest.main()
