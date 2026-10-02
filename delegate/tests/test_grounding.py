# SPDX-License-Identifier: GPL-3.0-only
"""Catching a quotation that is not in the source."""

from __future__ import annotations

import unittest

from shuttle_delegate.grounding import check, fields_in_source, quotes

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

    def test_backticks_do_not_mark_a_citation(self) -> None:
        self.assertEqual(quotes("see `extra_network=podman always`"), [])

    def test_backticks_between_identifiers_invent_nothing(self) -> None:
        answer = (
            "The linter is `shellcheck`. The text states, "
            '"shellcheck reports nothing", and `checkpatch.pl` is named.'
        )
        self.assertEqual(quotes(answer), ["shellcheck reports nothing"])

    def test_trailing_punctuation_is_not_part_of_the_claim(self) -> None:
        self.assertEqual(
            quotes('it said "at most 72 characters."'),
            ["at most 72 characters"],
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


class MarkupTest(unittest.TestCase):
    """Markdown writes a thing down; it is not part of the thing."""

    SOURCE = (
        "- Subject: `subsystem: imperative description`, at most 72\n"
        "  characters, no trailing period.\n"
        "- **shellcheck** reports nothing.\n"
    )

    def test_a_quote_without_the_backticks_is_grounded(self) -> None:
        quoted = "Subject: subsystem: imperative description, at most 72"
        self.assertTrue(check(f'"{quoted}"', self.SOURCE).ok)

    def test_a_quote_without_the_emphasis_is_grounded(self) -> None:
        self.assertTrue(check('"shellcheck reports nothing"', self.SOURCE).ok)

    def test_dropping_markup_does_not_ground_an_invention(self) -> None:
        self.assertFalse(
            check('"ruff reports nothing at all"', self.SOURCE).ok
        )


class PairingTest(unittest.TestCase):
    """Two short quotations on a line must not become one long one."""

    CASES = (
        'Pass "--no-ngram" to disable the speculator, or "--no-draft".',
        'Set "-e" and then also set "-u" at the top of the script.',
        'Use "ruff" together with "shellcheck" before every commit.',
    )

    def test_the_prose_between_short_quotations_is_not_a_quotation(
        self,
    ) -> None:
        for answer in self.CASES:
            with self.subTest(answer=answer):
                self.assertEqual(quotes(answer), [])

    def test_a_long_quotation_beside_a_short_one_is_still_found(self) -> None:
        answer = 'Pass "-e" because "the script must stop on an error".'
        self.assertEqual(quotes(answer), ["the script must stop on an error"])

    def test_two_long_quotations_are_both_found(self) -> None:
        answer = (
            'It says "the first long quotation here" and also '
            '"the second long quotation here".'
        )
        self.assertEqual(len(quotes(answer)), 2)

    def test_an_unclosed_quotation_ends_at_its_line(self) -> None:
        answer = 'He wrote "an unclosed quotation here\nand the next line.'
        self.assertEqual(quotes(answer), ["an unclosed quotation here"])


class FieldShapeTest(unittest.TestCase):
    """A schema may ask for a list or an object, and those count too."""

    SOURCE = "The gateway is 10.89.7.1 and the subnet is 10.89.7.0/24.\n"

    def test_strings_inside_a_list_are_checked(self) -> None:
        fields = {"addresses": ["10.89.7.1", "192.168.254.254"]}
        missing = fields_in_source(fields, self.SOURCE)
        self.assertEqual(len(missing), 1)
        self.assertIn("addresses[1]", missing[0])

    def test_strings_inside_an_object_are_checked(self) -> None:
        fields = {"net": {"gateway": "10.89.7.1", "dns": "203.0.113.9"}}
        missing = fields_in_source(fields, self.SOURCE)
        self.assertEqual(len(missing), 1)
        self.assertIn("net.dns", missing[0])

    def test_a_nested_structure_that_holds_up_reports_nothing(self) -> None:
        fields = {"net": {"hosts": ["10.89.7.1", "10.89.7.0/24"]}}
        self.assertEqual(fields_in_source(fields, self.SOURCE), [])

    def test_numbers_are_still_left_alone(self) -> None:
        self.assertEqual(fields_in_source({"port": 8081}, self.SOURCE), [])


class TableQuoteTest(unittest.TestCase):
    """A quotation out of a table is not one span of the source.

    Measured on a timing report: every correct answer doc-qa gave was
    refused, because the column names and their values have a rule of
    dashes between them that nobody quotes. The field a preset grounds
    is compared whole, which is the path this is about.
    """

    REPORT = (
        "+------------+-----------------+------------+------+\n"
        "; Fmax       ; Restricted Fmax ; Clock Name ; Note ;\n"
        "+------------+-----------------+------------+------+\n"
        "; 154.23 MHz ; 154.23 MHz      ; clk        ;      ;\n"
        "+------------+-----------------+------------+------+\n"
    )

    HELD = (
        "; Fmax       ; Restricted Fmax ; Clock Name ; Note ;\n"
        "; 154.23 MHz ; 154.23 MHz      ; clk        ;      ;"
    )

    def test_a_label_and_its_value_ground_together(self) -> None:
        self.assertEqual(
            fields_in_source({"quote": self.HELD}, self.REPORT), []
        )

    def test_one_line_of_the_table_grounds_as_a_span(self) -> None:
        one = "; 154.23 MHz ; 154.23 MHz      ; clk        ;      ;"
        self.assertEqual(fields_in_source({"quote": one}, self.REPORT), [])

    def test_an_invented_line_is_still_refused(self) -> None:
        """Line by line is not line by invention."""
        made = (
            "; Fmax       ; Restricted Fmax ; Clock Name ; Note ;\n"
            "; 912.00 MHz ; 912.00 MHz      ; clk        ;      ;"
        )
        self.assertEqual(
            len(fields_in_source({"quote": made}, self.REPORT)), 1
        )

    def test_one_invented_line_refuses_the_whole_quotation(self) -> None:
        made = self.HELD + "\n; a line the report does not hold at all ;"
        self.assertEqual(
            len(fields_in_source({"quote": made}, self.REPORT)), 1
        )

    def test_a_single_line_that_is_absent_is_refused(self) -> None:
        """One line has no lines to fall back to, so the span rule
        stands: the relaxation is for a quotation of several."""
        self.assertEqual(
            len(
                fields_in_source(
                    {"quote": "; 912.00 MHz ; not in it ;"}, self.REPORT
                )
            ),
            1,
        )

    def test_prose_across_a_line_break_is_unaffected(self) -> None:
        source = "The reset is held for ten clock\ncycles after power-up."
        said = "The reset is held for ten clock cycles"
        self.assertEqual(fields_in_source({"quote": said}, source), [])


class QuotesByLineTest(unittest.TestCase):
    def test_an_unclosed_quotation_ends_at_its_line(self) -> None:
        """Deliberate, so a stray mark cannot swallow a paragraph --
        and the consequence is that the closing line of a quotation
        written across two lines is not extracted as one. A field a
        preset grounds does not go through this path; it is compared
        whole.
        """
        said = '"first line of it\nsecond line of it"'
        self.assertEqual(quotes(said), ["first line of it"])


if __name__ == "__main__":
    unittest.main()
