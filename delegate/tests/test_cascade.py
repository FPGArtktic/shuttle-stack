# SPDX-License-Identifier: GPL-3.0-only
"""Climbing from one profile to the next, and stopping out loud."""

from __future__ import annotations

import unittest
from typing import Any

from shuttle_delegate.cascade import CascadeError, climb, ladder
from shuttle_delegate.profiles import ProfileError


class answers:  # noqa: N801
    """A run that verifies on the profiles named true.

    It remembers the profiles it was asked for, which is what the
    climb is judged on.
    """

    def __init__(self, **verdicts: bool) -> None:
        self.verdicts = verdicts
        self.seen: list[str] = []

    def __call__(self, profile: str) -> dict[str, Any]:
        self.seen.append(profile)
        return {
            "answer": f"from {profile}",
            "verified": self.verdicts.get(profile, False),
            "attempts": 1,
            "local_tokens": 100,
        }


class LadderTest(unittest.TestCase):
    def test_a_list_of_profiles_is_read_in_order(self) -> None:
        self.assertEqual(ladder("fast, extract"), ["fast", "extract"])

    def test_an_empty_cascade_says_what_one_looks_like(self) -> None:
        with self.assertRaises(CascadeError) as caught:
            ladder("  ,  ")
        self.assertIn("fast,extract", str(caught.exception))

    def test_an_unknown_profile_is_caught_before_anything_runs(self) -> None:
        with self.assertRaises(ProfileError):
            ladder("fast,nonsense")


class ClimbTest(unittest.TestCase):
    def test_the_first_profile_that_verifies_ends_the_climb(self) -> None:
        run = answers(fast=True, long=True)
        result = climb(["fast", "long"], run)
        self.assertEqual(run.seen, ["fast"])
        self.assertEqual(result["answer"], "from fast")
        self.assertNotIn("escalate", result)

    def test_a_failure_moves_up_to_the_next_profile(self) -> None:
        run = answers(long=True)
        result = climb(["fast", "long"], run)
        self.assertEqual(run.seen, ["fast", "long"])
        self.assertEqual(result["answer"], "from long")
        self.assertTrue(result["verified"])

    def test_the_whole_ladder_is_reported(self) -> None:
        result = climb(["fast", "long"], answers(long=True))
        self.assertEqual(
            [step["profile"] for step in result["ladder"]], ["fast", "long"]
        )
        self.assertEqual(
            [step["verified"] for step in result["ladder"]], [False, True]
        )

    def test_the_tokens_are_the_whole_climb_not_the_last_step(self) -> None:
        result = climb(["fast", "long"], answers(long=True))
        self.assertEqual(result["local_tokens"], 200)

    def test_nothing_verifying_hands_the_question_back(self) -> None:
        result = climb(["fast", "long"], answers())
        self.assertTrue(result["escalate"])
        self.assertIn("better done by you", result["note"])
        self.assertFalse(result["verified"])

    def test_one_profile_is_a_ladder_of_one(self) -> None:
        result = climb(["long"], answers(long=True))
        self.assertEqual(len(result["ladder"]), 1)


if __name__ == "__main__":
    unittest.main()
