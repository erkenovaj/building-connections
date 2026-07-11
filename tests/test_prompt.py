"""Tests for prompt construction and guess parsing.

Prompt parity: api/llm/guess.py. Parsing parity: tests/test_llm_guess.py.
"""

from __future__ import annotations

import json
import unittest

from connections_gym.prompt import build_prompt, parse_guess

REMAINING = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]


class BuildPromptTest(unittest.TestCase):
    def test_contains_schema_and_counts(self):
        prompt = build_prompt(REMAINING, 4, [])
        self.assertIn(
            '{"guess":{"label":"short label","items":["term","term","term","term"]},'
            '"notes":"one short sentence"}',
            prompt,
        )
        self.assertIn("This puzzle has 4 groups total.", prompt)
        self.assertIn(f"Remaining terms: {json.dumps(REMAINING)}", prompt)

    def test_no_history_has_no_avoid_line(self):
        self.assertNotIn("Do not repeat", build_prompt(REMAINING, 4, []))

    def test_history_renders_tried_groups(self):
        history = [{"items": ["A1", "A2", "A3", "A4"]}]
        prompt = build_prompt(REMAINING, 4, history)
        self.assertIn("Do not repeat", prompt)
        self.assertIn(json.dumps([["A1", "A2", "A3", "A4"]]), prompt)


class ParseGuessTest(unittest.TestCase):
    def test_nested_guess_shape(self):
        text = '{"guess":{"label":"Group A","items":["A1","A2","A3","A4"]},"notes":"x"}'
        self.assertEqual(
            parse_guess(text, REMAINING), {"label": "Group A", "items": ["A1", "A2", "A3", "A4"]}
        )

    def test_flat_items_shape(self):
        text = '{"items":["B1","B2","B3","B4"],"label":"Group B"}'
        self.assertEqual(
            parse_guess(text, REMAINING), {"label": "Group B", "items": ["B1", "B2", "B3", "B4"]}
        )

    def test_strips_think_tags(self):
        text = '<think>let me reason A1 A2</think>{"guess":{"label":"G","items":["A1","A2","A3","A4"]}}'
        result = parse_guess(text, REMAINING)
        self.assertEqual(result["items"], ["A1", "A2", "A3", "A4"])

    def test_strips_code_fence(self):
        text = '```json\n{"guess":{"label":"G","items":["A1","A2","A3","A4"]}}\n```'
        result = parse_guess(text, REMAINING)
        self.assertEqual(result["items"], ["A1", "A2", "A3", "A4"])

    def test_item_not_remaining_rejected(self):
        text = '{"guess":{"label":"G","items":["A1","A2","A3","ZZ"]}}'
        self.assertIsNone(parse_guess(text, REMAINING))

    def test_fewer_than_four_rejected(self):
        text = '{"guess":{"label":"G","items":["A1","A2","A3"]}}'
        self.assertIsNone(parse_guess(text, REMAINING))

    def test_non_json_returns_none(self):
        self.assertIsNone(parse_guess("I think the answer is A1, A2, A3, A4", REMAINING))


if __name__ == "__main__":
    unittest.main()
