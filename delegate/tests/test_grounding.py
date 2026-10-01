# SPDX-License-Identifier: GPL-3.0-only
"""Catching a quotation that is not in the source."""

from __future__ import annotations

import unittest

from shuttle_delegate.grounding import check, quotes

SOURCE = """\
probe_publish()
{
	extra_network=podman
	info "PublishPort does not reach Internal=true networks here;" \\
		"adding Network=podman, which gives the containers egress"
}
"""


class QuotesTest(unittest.TestCase):
    def test_double_quoted_spans_are_found(self) -> None:
        self.assertEqual(
            quotes('He wrote "a long enough span here" in the file.'),
            ["a long enough span here"],
        )

    def test_backticked_spans_count_as_citations(self) -> None:
        self.assertEqual(
            quotes("see `extra_network=podman always`"),
            ["extra_network=podman always"],
        )

    def test_short_spans_are_words_not_citations(self) -> None:
        self.assertEqual(quotes('it said "no" and "yes"'), [])

    def test_a_repeated_quotation_is_listed_once(self) -> None:
        answer = '"the same long span" and again "the same long span"'
        self.assertEqual(len(quotes(answer)), 1)

    def test_an_answer_without_quotations_has_none(self) -> None:
        self.assertEqual(quotes("Network=podman is added."), [])


class CheckTest(unittest.TestCase):
    def test_a_real_quotation_is_grounded(self) -> None:
        quoted = "adding Network=podman, which gives the containers egress"
        answer = f'It adds "{quoted}".'
        result = check(answer, SOURCE)
        self.assertTrue(result.ok)
        self.assertEqual(result.total, 1)
        self.assertEqual(result.grounded, 1)

    def test_an_invented_quotation_is_caught(self) -> None:
        invented = (
            "If the probe fails, the container units are added with "
            "the probe failure status"
        )
        answer = f'The text states: "{invented}".'
        result = check(answer, SOURCE)
        self.assertFalse(result.ok)
        self.assertEqual(result.grounded, 0)
        self.assertIn("probe fails", result.missing[0])

    def test_whitespace_and_case_do_not_decide_it(self) -> None:
        answer = (
            '"ADDING   Network=podman,  which gives the containers egress"'
        )
        self.assertTrue(check(answer, SOURCE).ok)

    def test_an_answer_with_no_quotations_is_not_contradicted(self) -> None:
        result = check("Network=podman is added.", SOURCE)
        self.assertTrue(result.ok)
        self.assertEqual(result.total, 0)

    def test_the_report_names_only_what_is_missing(self) -> None:
        good = check(
            '"adding Network=podman, which gives the containers egress"',
            SOURCE,
        )
        self.assertNotIn("quotes_not_in_source", good.report())
        bad = check('"a sentence nobody ever wrote down here"', SOURCE)
        self.assertIn("quotes_not_in_source", bad.report())


if __name__ == "__main__":
    unittest.main()


class ContinuationTest(unittest.TestCase):
    """A line that wraps is still one line."""

    SOURCE = (
        "\troot usermod --add-subuids 100000-165535 \\\n"
        '\t\t--add-subgids 100000-165535 "$user"\n'
    )

    def test_a_quote_across_a_continuation_is_grounded(self) -> None:
        quoted = (
            "root usermod --add-subuids 100000-165535 "
            "--add-subgids 100000-165535"
        )
        self.assertTrue(check(f'"{quoted}"', self.SOURCE).ok)

    def test_the_normalisation_does_not_ground_an_invention(self) -> None:
        self.assertFalse(check('"usermod --remove-subuids 1"', self.SOURCE).ok)
