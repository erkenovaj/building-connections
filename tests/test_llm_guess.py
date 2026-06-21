"""Tests for single-group guess normalization in the agent loop."""

from __future__ import annotations

import unittest

from api.llm._ollamafree import normalize_one_guess

REMAINING = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]


class NormalizeOneGuessTest(unittest.TestCase):
    def test_valid_guess(self):
        parsed = {"guess": {"label": "Group A", "items": ["A1", "A2", "A3", "A4"]}}
        result = normalize_one_guess(parsed, REMAINING)
        self.assertEqual(result, {"label": "Group A", "items": ["A1", "A2", "A3", "A4"]})

    def test_term_not_remaining_rejected(self):
        parsed = {"guess": {"label": "X", "items": ["A1", "A2", "A3", "ZZ"]}}
        self.assertIsNone(normalize_one_guess(parsed, REMAINING))

    def test_fewer_than_four_returns_none(self):
        parsed = {"guess": {"label": "X", "items": ["A1", "A2", "A3"]}}
        self.assertIsNone(normalize_one_guess(parsed, REMAINING))

    def test_missing_guess_returns_none(self):
        self.assertIsNone(normalize_one_guess({"notes": "hmm"}, REMAINING))

    def test_flat_items_shape_accepted(self):
        parsed = {"items": ["B1", "B2", "B3", "B4"], "label": "Group B"}
        result = normalize_one_guess(parsed, REMAINING)
        self.assertEqual(result, {"label": "Group B", "items": ["B1", "B2", "B3", "B4"]})


if __name__ == "__main__":
    unittest.main()
