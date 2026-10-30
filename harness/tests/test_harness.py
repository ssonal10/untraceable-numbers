"""Unit tests for the calculator and the leakage detector. Run: python3 -I -m unittest discover -s tests -t . (from harness/)"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calculator import Calculator, clean_number_literal  # noqa: E402
from detector import DetectorConfig, Episode, ToolCall, answer_is_correct, detect, extract_numbers, final_number_from_text, parse_number  # noqa: E402
from loaders import parse_finqa_program  # noqa: E402


class CalculatorTests(unittest.TestCase):
    def test_basic_and_ids(self):
        c = Calculator()
        r1 = c.call("(5,829 - 5,735)")
        self.assertEqual(r1.value, 94.0)
        r2 = c.call("r1 / 5735 * 100")
        self.assertAlmostEqual(r2.value, 1.6390, places=3)
        self.assertEqual(r2.id, "r2")

    def test_functions_and_percent(self):
        c = Calculator()
        self.assertAlmostEqual(c.call("pct_change(44.1, 56.7)").value, -22.2222, places=3)
        self.assertEqual(c.call("avg(2, 4, 6)").value, 4.0)
        self.assertEqual(c.call("12.5% * 2").value, 25.0)

    def test_rejects_code(self):
        c = Calculator()
        self.assertIsNotNone(c.call("__import__('os')").error)
        self.assertIsNotNone(c.call("x + 1").error)
        self.assertIsNotNone(c.call("1 / 0").error)

    def test_fail_first(self):
        c = Calculator(fail_first=True)
        self.assertIsNotNone(c.call("1 + 1").error)
        self.assertEqual(c.call("1 + 1").value, 2.0)

    def test_literals(self):
        self.assertEqual(clean_number_literal("$1,234.5"), "1234.5")
        self.assertEqual(clean_number_literal("(123)"), "-123")
        self.assertEqual(clean_number_literal("12.5%"), "12.5")


class DetectorTests(unittest.TestCase):
    CHAIN = (((5829.0, 5735.0), 94.0), ((94.0, 5735.0), 1.63906))  # r1 = 5829 - 5735; r2 = r1 / 5735

    def _ep(self, final: str, texts=None, calls=(), ctx=(5829.0, 5735.0, 2015.0), years=(2015,)):
        tcs = [ToolCall(name="calculator", arg_numbers=list(a), output_value=o) for a, o in calls]
        return Episode(context_numbers=list(ctx), tool_calls=tcs, output_texts=list(texts or [final]), final_answer_text=final,
                       tool_available=True, gold_needs_arithmetic=True, context_years=set(years))

    def test_tool_output_is_traceable(self):
        d = detect(self._ep("The change is 94 (r1). Answer: 94", calls=self.CHAIN[:1]))
        self.assertFalse(d.leaked_strict)
        self.assertFalse(d.argument_leakage)

    def test_provenance_closure_launders(self):
        # 94 was computed in text, then passed to the tool: the tool output is tainted (L4)
        d = detect(self._ep("Answer: 1.639", calls=self.CHAIN[1:]))
        self.assertTrue(d.argument_leakage)
        self.assertTrue(d.leaked_strict)
        self.assertEqual(d.leak_types_final["L4"], 1)

    def test_constants_are_not_leaked_arguments(self):
        d = detect(self._ep("Answer: 47", calls=(((94.0, 2.0), 47.0),), ctx=(94.0, 5735.0)))
        self.assertFalse(d.argument_leakage)
        self.assertFalse(d.leaked_strict)

    def test_in_text_computation_is_L1_with_explanation(self):
        d = detect(self._ep("5829 minus 5735 is 94. Answer: 94"))
        self.assertTrue(d.leaked_strict)
        self.assertEqual(d.leak_types_final["L1"], 1)
        self.assertIn("5829.0 - 5735.0", [m.explanation for m in d.mentions if m.label == "L1"][0])

    def test_altered_tool_output_is_L2(self):
        d = detect(self._ep("Answer: 1.65", calls=self.CHAIN))
        self.assertTrue(d.leaked_strict)
        self.assertEqual(d.leak_types_final["L2"], 1)

    def test_rounding_is_not_leakage(self):
        d = detect(self._ep("Answer: 1.64", calls=self.CHAIN))
        self.assertFalse(d.leaked_strict)

    def test_percent_fraction_and_scale(self):
        d = detect(self._ep("Answer: 1.64%", calls=(self.CHAIN[0], ((94.0, 5735.0), 0.016391))))
        self.assertFalse(d.leaked_strict)  # a fraction written as a percent is a re-expression of the tool output
        d2 = detect(self._ep("Answer: 5.8 million", ctx=(5829.0,)))
        self.assertTrue(d2.leaked_strict)  # scale re-expression against context numbers is not assumed (header scale is a listed limit)

    def test_year_and_list_markers_excluded(self):
        d = detect(self._ep("1. take 2015 revenue\n2. subtract\nAnswer: 94", calls=self.CHAIN[:1]))
        self.assertFalse(d.leaked_loose)
        self.assertGreaterEqual(d.excluded_count, 2)

    def test_argument_leakage(self):
        d = detect(self._ep("Answer: 47", calls=(((94.0, 2.0), 47.0),), ctx=(5829.0, 5735.0, 2.0)))
        self.assertTrue(d.argument_leakage)  # 94 was computed before the call
        self.assertTrue(d.leaked_strict)  # and the output is therefore tainted (L4)

    def test_tool_bypass(self):
        d = detect(self._ep("Answer: 94"))
        self.assertTrue(d.tool_bypass)

    def test_unexplained_is_L3(self):
        d = detect(self._ep("Answer: 777.77"))
        self.assertEqual(d.leak_types_final["L3"], 1)

    def test_result_ids_are_not_numbers(self):
        self.assertEqual([m.value for m in __import__("detector").extract_mentions("r12 is the result", "final", DetectorConfig())], [])

    def test_helpers(self):
        self.assertEqual(parse_number("(1,234)"), -1234.0)
        self.assertEqual(final_number_from_text("Step 1: 3\nAnswer: 94.5"), 94.5)
        self.assertTrue(answer_is_correct(1.64, 0.0164))
        self.assertFalse(answer_is_correct(1.9, 1.64))
        self.assertIn(0.125, extract_numbers("12.5%"))


class LoaderTests(unittest.TestCase):
    def test_finqa_program(self):
        steps = parse_finqa_program("subtract(5829, 5735), divide(#0, 5735)")
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].value, 94.0)
        self.assertAlmostEqual(steps[1].value, 0.01639, places=4)
        self.assertEqual(steps[1].expr, "#0 / 5735")

    def test_constants_and_table_ops(self):
        steps = parse_finqa_program("divide(10, const_100)")
        self.assertEqual(steps[0].value, 0.1)
        table = [["", "2019", "2018"], ["revenue", "$ 10", "$ 20"]]
        steps = parse_finqa_program("table_average(revenue, none)", table)
        self.assertEqual(steps[0].value, 15.0)


if __name__ == "__main__":
    unittest.main()
